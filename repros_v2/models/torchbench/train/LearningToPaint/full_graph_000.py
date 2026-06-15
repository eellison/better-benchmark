class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[64, 9, 3, 3][81, 9, 3, 1]cuda:0", primals_2: "f32[96, 9, 128, 128][147456, 16384, 128, 1]cuda:0", primals_3: "i64[][]cuda:0", primals_4: "f32[64][1]cuda:0", primals_5: "f32[64][1]cuda:0", primals_6: "f32[64][1]cuda:0", primals_7: "f32[64][1]cuda:0", primals_8: "f32[64, 64, 3, 3][576, 9, 3, 1]cuda:0", primals_9: "i64[][]cuda:0", primals_10: "f32[64][1]cuda:0", primals_11: "f32[64][1]cuda:0", primals_12: "f32[64][1]cuda:0", primals_13: "f32[64][1]cuda:0", primals_14: "f32[64, 64, 3, 3][576, 9, 3, 1]cuda:0", primals_15: "i64[][]cuda:0", primals_16: "f32[64][1]cuda:0", primals_17: "f32[64][1]cuda:0", primals_18: "f32[64][1]cuda:0", primals_19: "f32[64][1]cuda:0", primals_20: "f32[64, 64, 1, 1][64, 1, 1, 1]cuda:0", primals_21: "i64[][]cuda:0", primals_22: "f32[64][1]cuda:0", primals_23: "f32[64][1]cuda:0", primals_24: "f32[64][1]cuda:0", primals_25: "f32[64][1]cuda:0", primals_26: "f32[64, 64, 3, 3][576, 9, 3, 1]cuda:0", primals_27: "i64[][]cuda:0", primals_28: "f32[64][1]cuda:0", primals_29: "f32[64][1]cuda:0", primals_30: "f32[64][1]cuda:0", primals_31: "f32[64][1]cuda:0", primals_32: "f32[64, 64, 3, 3][576, 9, 3, 1]cuda:0", primals_33: "i64[][]cuda:0", primals_34: "f32[64][1]cuda:0", primals_35: "f32[64][1]cuda:0", primals_36: "f32[64][1]cuda:0", primals_37: "f32[64][1]cuda:0", primals_38: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0", primals_39: "i64[][]cuda:0", primals_40: "f32[128][1]cuda:0", primals_41: "f32[128][1]cuda:0", primals_42: "f32[128][1]cuda:0", primals_43: "f32[128][1]cuda:0", primals_44: "f32[128, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_45: "i64[][]cuda:0", primals_46: "f32[128][1]cuda:0", primals_47: "f32[128][1]cuda:0", primals_48: "f32[128][1]cuda:0", primals_49: "f32[128][1]cuda:0", primals_50: "f32[128, 64, 1, 1][64, 1, 1, 1]cuda:0", primals_51: "i64[][]cuda:0", primals_52: "f32[128][1]cuda:0", primals_53: "f32[128][1]cuda:0", primals_54: "f32[128][1]cuda:0", primals_55: "f32[128][1]cuda:0", primals_56: "f32[128, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_57: "i64[][]cuda:0", primals_58: "f32[128][1]cuda:0", primals_59: "f32[128][1]cuda:0", primals_60: "f32[128][1]cuda:0", primals_61: "f32[128][1]cuda:0", primals_62: "f32[128, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_63: "i64[][]cuda:0", primals_64: "f32[128][1]cuda:0", primals_65: "f32[128][1]cuda:0", primals_66: "f32[128][1]cuda:0", primals_67: "f32[128][1]cuda:0", primals_68: "f32[256, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_69: "i64[][]cuda:0", primals_70: "f32[256][1]cuda:0", primals_71: "f32[256][1]cuda:0", primals_72: "f32[256][1]cuda:0", primals_73: "f32[256][1]cuda:0", primals_74: "f32[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", primals_75: "i64[][]cuda:0", primals_76: "f32[256][1]cuda:0", primals_77: "f32[256][1]cuda:0", primals_78: "f32[256][1]cuda:0", primals_79: "f32[256][1]cuda:0", primals_80: "f32[256, 128, 1, 1][128, 1, 1, 1]cuda:0", primals_81: "i64[][]cuda:0", primals_82: "f32[256][1]cuda:0", primals_83: "f32[256][1]cuda:0", primals_84: "f32[256][1]cuda:0", primals_85: "f32[256][1]cuda:0", primals_86: "f32[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", primals_87: "i64[][]cuda:0", primals_88: "f32[256][1]cuda:0", primals_89: "f32[256][1]cuda:0", primals_90: "f32[256][1]cuda:0", primals_91: "f32[256][1]cuda:0", primals_92: "f32[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", primals_93: "i64[][]cuda:0", primals_94: "f32[256][1]cuda:0", primals_95: "f32[256][1]cuda:0", primals_96: "f32[256][1]cuda:0", primals_97: "f32[256][1]cuda:0", primals_98: "f32[512, 256, 3, 3][2304, 9, 3, 1]cuda:0", primals_99: "i64[][]cuda:0", primals_100: "f32[512][1]cuda:0", primals_101: "f32[512][1]cuda:0", primals_102: "f32[512][1]cuda:0", primals_103: "f32[512][1]cuda:0", primals_104: "f32[512, 512, 3, 3][4608, 9, 3, 1]cuda:0", primals_105: "i64[][]cuda:0", primals_106: "f32[512][1]cuda:0", primals_107: "f32[512][1]cuda:0", primals_108: "f32[512][1]cuda:0", primals_109: "f32[512][1]cuda:0", primals_110: "f32[512, 256, 1, 1][256, 1, 1, 1]cuda:0", primals_111: "i64[][]cuda:0", primals_112: "f32[512][1]cuda:0", primals_113: "f32[512][1]cuda:0", primals_114: "f32[512][1]cuda:0", primals_115: "f32[512][1]cuda:0", primals_116: "f32[512, 512, 3, 3][4608, 9, 3, 1]cuda:0", primals_117: "i64[][]cuda:0", primals_118: "f32[512][1]cuda:0", primals_119: "f32[512][1]cuda:0", primals_120: "f32[512][1]cuda:0", primals_121: "f32[512][1]cuda:0", primals_122: "f32[512, 512, 3, 3][4608, 9, 3, 1]cuda:0", primals_123: "i64[][]cuda:0", primals_124: "f32[512][1]cuda:0", primals_125: "f32[512][1]cuda:0", primals_126: "f32[512][1]cuda:0", primals_127: "f32[512][1]cuda:0", primals_128: "f32[65, 512][512, 1]cuda:0", primals_129: "f32[65][1]cuda:0"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:128 in forward, code: x = F.relu(self.bn1(self.conv1(x)))
        convert_element_type: "bf16[64, 9, 3, 3][81, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_1, torch.bfloat16);  primals_1 = None
        convert_element_type_1: "bf16[96, 9, 128, 128][147456, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16);  primals_2 = None
        convolution: "bf16[96, 64, 64, 64][262144, 4096, 64, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_1, convert_element_type, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
        add: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_3, 1)
        convert_element_type_2: "f32[96, 64, 64, 64][262144, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_2, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_2 = None
        getitem: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        add_1: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[96, 64, 64, 64][262144, 4096, 64, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul: "f32[96, 64, 64, 64][262144, 4096, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_1: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_1: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
        mul_2: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_4, 0.9)
        add_2: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_2: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_3: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_2, 1.000002543137978);  squeeze_2 = None
        mul_4: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
        mul_5: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_5, 0.9)
        add_3: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_6, -1)
        unsqueeze_1: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[96, 64, 64, 64][262144, 4096, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_7, -1);  primals_7 = None
        unsqueeze_3: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_4: "f32[96, 64, 64, 64][262144, 4096, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        convert_element_type_3: "bf16[96, 64, 64, 64][262144, 4096, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None
        relu: "bf16[96, 64, 64, 64][262144, 4096, 64, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_3);  convert_element_type_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        convert_element_type_4: "bf16[64, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_8, torch.bfloat16);  primals_8 = None
        convolution_1: "bf16[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.convolution.default(relu, convert_element_type_4, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
        add_5: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_9, 1)
        convert_element_type_5: "f32[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_5, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_5 = None
        getitem_2: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_1[0]
        getitem_3: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        add_6: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
        rsqrt_1: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_1: "f32[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
        mul_7: "f32[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        squeeze_3: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        squeeze_4: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_8: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
        mul_9: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_10, 0.9)
        add_7: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        squeeze_5: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_10: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_5, 1.000010172629523);  squeeze_5 = None
        mul_11: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
        mul_12: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_11, 0.9)
        add_8: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        unsqueeze_4: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_12, -1)
        unsqueeze_5: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_13: "f32[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
        unsqueeze_6: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_13, -1);  primals_13 = None
        unsqueeze_7: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_9: "f32[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
        convert_element_type_6: "bf16[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_9, torch.bfloat16);  add_9 = None
        relu_1: "bf16[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_6);  convert_element_type_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        convert_element_type_7: "bf16[64, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
        convolution_2: "bf16[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.convolution.default(relu_1, convert_element_type_7, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_10: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_15, 1)
        convert_element_type_8: "f32[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_2, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_8, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_8 = None
        getitem_4: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_2[0]
        getitem_5: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        add_11: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
        rsqrt_2: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_2: "f32[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
        mul_14: "f32[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        squeeze_6: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        squeeze_7: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_15: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
        mul_16: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_16, 0.9)
        add_12: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
        squeeze_8: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_17: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_8, 1.000010172629523);  squeeze_8 = None
        mul_18: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
        mul_19: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_17, 0.9)
        add_13: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
        unsqueeze_8: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_18, -1)
        unsqueeze_9: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        mul_20: "f32[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
        unsqueeze_10: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_19, -1);  primals_19 = None
        unsqueeze_11: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        add_14: "f32[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
        convert_element_type_9: "bf16[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_14, torch.bfloat16);  add_14 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:56 in forward, code: out += self.shortcut(x)
        convert_element_type_10: "bf16[64, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_20, torch.bfloat16);  primals_20 = None
        convolution_3: "bf16[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.convolution.default(relu, convert_element_type_10, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        add_15: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_21, 1)
        convert_element_type_11: "f32[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_3, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_11, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_11 = None
        getitem_6: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_3[0]
        getitem_7: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        add_16: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
        rsqrt_3: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_3: "f32[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_3, getitem_7)
        mul_21: "f32[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        squeeze_9: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        squeeze_10: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_22: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
        mul_23: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_22, 0.9)
        add_17: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
        squeeze_11: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_24: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_11, 1.000010172629523);  squeeze_11 = None
        mul_25: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
        mul_26: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_23, 0.9)
        add_18: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
        unsqueeze_12: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_24, -1)
        unsqueeze_13: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_27: "f32[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
        unsqueeze_14: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_25, -1);  primals_25 = None
        unsqueeze_15: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_19: "f32[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
        convert_element_type_12: "bf16[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_19, torch.bfloat16);  add_19 = None
        add_20: "bf16[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_9, convert_element_type_12);  convert_element_type_9 = convert_element_type_12 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:57 in forward, code: out = F.relu(out)
        relu_2: "bf16[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.relu.default(add_20);  add_20 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        convert_element_type_13: "bf16[64, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_26, torch.bfloat16);  primals_26 = None
        convolution_4: "bf16[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.convolution.default(relu_2, convert_element_type_13, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_21: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_27, 1)
        convert_element_type_14: "f32[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_4, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_14, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_14 = None
        getitem_8: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_4[0]
        getitem_9: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        add_22: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
        rsqrt_4: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        sub_4: "f32[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_4, getitem_9)
        mul_28: "f32[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        squeeze_12: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        squeeze_13: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_29: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
        mul_30: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_28, 0.9)
        add_23: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
        squeeze_14: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_31: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_14, 1.000010172629523);  squeeze_14 = None
        mul_32: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
        mul_33: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_29, 0.9)
        add_24: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
        unsqueeze_16: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_30, -1)
        unsqueeze_17: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        mul_34: "f32[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
        unsqueeze_18: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_31, -1);  primals_31 = None
        unsqueeze_19: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        add_25: "f32[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None
        convert_element_type_15: "bf16[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_25, torch.bfloat16);  add_25 = None
        relu_3: "bf16[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_15);  convert_element_type_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        convert_element_type_16: "bf16[64, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_32, torch.bfloat16);  primals_32 = None
        convolution_5: "bf16[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.convolution.default(relu_3, convert_element_type_16, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_26: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_33, 1)
        convert_element_type_17: "f32[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_5, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_17, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_17 = None
        getitem_10: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_5[0]
        getitem_11: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        add_27: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
        rsqrt_5: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        sub_5: "f32[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_5, getitem_11)
        mul_35: "f32[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        squeeze_15: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        squeeze_16: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_36: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
        mul_37: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_34, 0.9)
        add_28: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
        squeeze_17: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_38: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_17, 1.000010172629523);  squeeze_17 = None
        mul_39: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
        mul_40: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_35, 0.9)
        add_29: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
        unsqueeze_20: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_36, -1)
        unsqueeze_21: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_41: "f32[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
        unsqueeze_22: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_37, -1);  primals_37 = None
        unsqueeze_23: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_30: "f32[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
        convert_element_type_18: "bf16[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_30, torch.bfloat16);  add_30 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:56 in forward, code: out += self.shortcut(x)
        add_31: "bf16[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_18, relu_2);  convert_element_type_18 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:57 in forward, code: out = F.relu(out)
        relu_4: "bf16[96, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.relu.default(add_31);  add_31 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        convert_element_type_19: "bf16[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_38, torch.bfloat16);  primals_38 = None
        convolution_6: "bf16[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.convolution.default(relu_4, convert_element_type_19, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
        add_32: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_39, 1)
        convert_element_type_20: "f32[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_6, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_20, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_20 = None
        getitem_12: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_6[0]
        getitem_13: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        add_33: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
        rsqrt_6: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        sub_6: "f32[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_6, getitem_13)
        mul_42: "f32[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        squeeze_18: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        squeeze_19: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_43: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
        mul_44: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_40, 0.9)
        add_34: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
        squeeze_20: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
        mul_45: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0000406917599187);  squeeze_20 = None
        mul_46: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_45, 0.1);  mul_45 = None
        mul_47: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_41, 0.9)
        add_35: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
        unsqueeze_24: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_42, -1)
        unsqueeze_25: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        mul_48: "f32[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_25);  mul_42 = unsqueeze_25 = None
        unsqueeze_26: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_43, -1);  primals_43 = None
        unsqueeze_27: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        add_36: "f32[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_48, unsqueeze_27);  mul_48 = unsqueeze_27 = None
        convert_element_type_21: "bf16[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_36, torch.bfloat16);  add_36 = None
        relu_5: "bf16[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_21);  convert_element_type_21 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        convert_element_type_22: "bf16[128, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16);  primals_44 = None
        convolution_7: "bf16[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.convolution.default(relu_5, convert_element_type_22, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_37: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_45, 1)
        convert_element_type_23: "f32[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_7, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_23, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_23 = None
        getitem_14: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_7[0]
        getitem_15: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        add_38: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
        rsqrt_7: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        sub_7: "f32[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_7, getitem_15)
        mul_49: "f32[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        squeeze_21: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        squeeze_22: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
        mul_50: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
        mul_51: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_46, 0.9)
        add_39: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
        squeeze_23: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
        mul_52: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0000406917599187);  squeeze_23 = None
        mul_53: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_52, 0.1);  mul_52 = None
        mul_54: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_47, 0.9)
        add_40: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
        unsqueeze_28: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_48, -1)
        unsqueeze_29: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_55: "f32[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
        unsqueeze_30: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_49, -1);  primals_49 = None
        unsqueeze_31: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_41: "f32[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None
        convert_element_type_24: "bf16[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_41, torch.bfloat16);  add_41 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:56 in forward, code: out += self.shortcut(x)
        convert_element_type_25: "bf16[128, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_50, torch.bfloat16);  primals_50 = None
        convolution_8: "bf16[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.convolution.default(relu_4, convert_element_type_25, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        add_42: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_51, 1)
        convert_element_type_26: "f32[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_8, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_26, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_26 = None
        getitem_16: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_8[0]
        getitem_17: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        add_43: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
        rsqrt_8: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        sub_8: "f32[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_8, getitem_17)
        mul_56: "f32[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        squeeze_24: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
        squeeze_25: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_57: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
        mul_58: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_52, 0.9)
        add_44: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
        squeeze_26: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
        mul_59: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0000406917599187);  squeeze_26 = None
        mul_60: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
        mul_61: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_53, 0.9)
        add_45: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
        unsqueeze_32: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_54, -1)
        unsqueeze_33: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        mul_62: "f32[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
        unsqueeze_34: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_55, -1);  primals_55 = None
        unsqueeze_35: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        add_46: "f32[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None
        convert_element_type_27: "bf16[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_46, torch.bfloat16);  add_46 = None
        add_47: "bf16[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_24, convert_element_type_27);  convert_element_type_24 = convert_element_type_27 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:57 in forward, code: out = F.relu(out)
        relu_6: "bf16[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.relu.default(add_47);  add_47 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        convert_element_type_28: "bf16[128, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_56, torch.bfloat16);  primals_56 = None
        convolution_9: "bf16[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.convolution.default(relu_6, convert_element_type_28, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_48: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_57, 1)
        convert_element_type_29: "f32[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_9, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_29, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_29 = None
        getitem_18: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_9[0]
        getitem_19: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        add_49: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
        rsqrt_9: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        sub_9: "f32[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_9, getitem_19)
        mul_63: "f32[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        squeeze_27: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        squeeze_28: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_64: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
        mul_65: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_58, 0.9)
        add_50: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
        squeeze_29: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
        mul_66: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0000406917599187);  squeeze_29 = None
        mul_67: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, 0.1);  mul_66 = None
        mul_68: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_59, 0.9)
        add_51: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
        unsqueeze_36: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_60, -1)
        unsqueeze_37: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_69: "f32[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_37);  mul_63 = unsqueeze_37 = None
        unsqueeze_38: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_61, -1);  primals_61 = None
        unsqueeze_39: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_52: "f32[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_39);  mul_69 = unsqueeze_39 = None
        convert_element_type_30: "bf16[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_52, torch.bfloat16);  add_52 = None
        relu_7: "bf16[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_30);  convert_element_type_30 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        convert_element_type_31: "bf16[128, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_62, torch.bfloat16);  primals_62 = None
        convolution_10: "bf16[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.convolution.default(relu_7, convert_element_type_31, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_53: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_63, 1)
        convert_element_type_32: "f32[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_10, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_32, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_32 = None
        getitem_20: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_10[0]
        getitem_21: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        add_54: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
        rsqrt_10: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        sub_10: "f32[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_10, getitem_21)
        mul_70: "f32[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        squeeze_30: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        squeeze_31: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_71: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
        mul_72: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_64, 0.9)
        add_55: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_71, mul_72);  mul_71 = mul_72 = None
        squeeze_32: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        mul_73: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0000406917599187);  squeeze_32 = None
        mul_74: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_73, 0.1);  mul_73 = None
        mul_75: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_65, 0.9)
        add_56: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
        unsqueeze_40: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_66, -1)
        unsqueeze_41: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        mul_76: "f32[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_41);  mul_70 = unsqueeze_41 = None
        unsqueeze_42: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_67, -1);  primals_67 = None
        unsqueeze_43: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        add_57: "f32[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_76, unsqueeze_43);  mul_76 = unsqueeze_43 = None
        convert_element_type_33: "bf16[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_57, torch.bfloat16);  add_57 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:56 in forward, code: out += self.shortcut(x)
        add_58: "bf16[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_33, relu_6);  convert_element_type_33 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:57 in forward, code: out = F.relu(out)
        relu_8: "bf16[96, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.relu.default(add_58);  add_58 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        convert_element_type_34: "bf16[256, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_68, torch.bfloat16);  primals_68 = None
        convolution_11: "bf16[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.convolution.default(relu_8, convert_element_type_34, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
        add_59: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_69, 1)
        convert_element_type_35: "f32[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_11, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_35, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_35 = None
        getitem_22: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_11[0]
        getitem_23: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        add_60: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
        rsqrt_11: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_11: "f32[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_11, getitem_23)
        mul_77: "f32[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        squeeze_33: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
        squeeze_34: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
        mul_78: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
        mul_79: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_70, 0.9)
        add_61: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
        squeeze_35: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
        mul_80: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0001627869119323);  squeeze_35 = None
        mul_81: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_80, 0.1);  mul_80 = None
        mul_82: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_71, 0.9)
        add_62: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
        unsqueeze_44: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_72, -1)
        unsqueeze_45: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_83: "f32[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_45);  mul_77 = unsqueeze_45 = None
        unsqueeze_46: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_73, -1);  primals_73 = None
        unsqueeze_47: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_63: "f32[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_47);  mul_83 = unsqueeze_47 = None
        convert_element_type_36: "bf16[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_63, torch.bfloat16);  add_63 = None
        relu_9: "bf16[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_36);  convert_element_type_36 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        convert_element_type_37: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_74, torch.bfloat16);  primals_74 = None
        convolution_12: "bf16[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.convolution.default(relu_9, convert_element_type_37, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_64: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_75, 1)
        convert_element_type_38: "f32[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_12, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_38, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_38 = None
        getitem_24: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_12[0]
        getitem_25: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        add_65: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
        rsqrt_12: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        sub_12: "f32[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_12, getitem_25)
        mul_84: "f32[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        squeeze_36: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        squeeze_37: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_85: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
        mul_86: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_76, 0.9)
        add_66: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
        squeeze_38: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
        mul_87: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0001627869119323);  squeeze_38 = None
        mul_88: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_87, 0.1);  mul_87 = None
        mul_89: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_77, 0.9)
        add_67: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
        unsqueeze_48: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_78, -1)
        unsqueeze_49: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        mul_90: "f32[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_49);  mul_84 = unsqueeze_49 = None
        unsqueeze_50: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_79, -1);  primals_79 = None
        unsqueeze_51: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        add_68: "f32[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_51);  mul_90 = unsqueeze_51 = None
        convert_element_type_39: "bf16[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_68, torch.bfloat16);  add_68 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:56 in forward, code: out += self.shortcut(x)
        convert_element_type_40: "bf16[256, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_80, torch.bfloat16);  primals_80 = None
        convolution_13: "bf16[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.convolution.default(relu_8, convert_element_type_40, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        add_69: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_81, 1)
        convert_element_type_41: "f32[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_13, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_41, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_41 = None
        getitem_26: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_13[0]
        getitem_27: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        add_70: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
        rsqrt_13: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        sub_13: "f32[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_13, getitem_27)
        mul_91: "f32[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        squeeze_39: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        squeeze_40: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
        mul_92: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
        mul_93: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_82, 0.9)
        add_71: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
        squeeze_41: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        mul_94: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0001627869119323);  squeeze_41 = None
        mul_95: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_94, 0.1);  mul_94 = None
        mul_96: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_83, 0.9)
        add_72: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None
        unsqueeze_52: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_84, -1)
        unsqueeze_53: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_97: "f32[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_53);  mul_91 = unsqueeze_53 = None
        unsqueeze_54: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_85, -1);  primals_85 = None
        unsqueeze_55: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_73: "f32[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_55);  mul_97 = unsqueeze_55 = None
        convert_element_type_42: "bf16[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_73, torch.bfloat16);  add_73 = None
        add_74: "bf16[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_39, convert_element_type_42);  convert_element_type_39 = convert_element_type_42 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:57 in forward, code: out = F.relu(out)
        relu_10: "bf16[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.relu.default(add_74);  add_74 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        convert_element_type_43: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_86, torch.bfloat16);  primals_86 = None
        convolution_14: "bf16[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.convolution.default(relu_10, convert_element_type_43, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_75: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_87, 1)
        convert_element_type_44: "f32[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_14, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_44, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_44 = None
        getitem_28: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_14[0]
        getitem_29: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        add_76: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
        rsqrt_14: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        sub_14: "f32[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_14, getitem_29)
        mul_98: "f32[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        squeeze_42: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        squeeze_43: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_99: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
        mul_100: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_88, 0.9)
        add_77: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
        squeeze_44: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        mul_101: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0001627869119323);  squeeze_44 = None
        mul_102: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_101, 0.1);  mul_101 = None
        mul_103: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_89, 0.9)
        add_78: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
        unsqueeze_56: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_90, -1)
        unsqueeze_57: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        mul_104: "f32[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_57);  mul_98 = unsqueeze_57 = None
        unsqueeze_58: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_91, -1);  primals_91 = None
        unsqueeze_59: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        add_79: "f32[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_59);  mul_104 = unsqueeze_59 = None
        convert_element_type_45: "bf16[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_79, torch.bfloat16);  add_79 = None
        relu_11: "bf16[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_45);  convert_element_type_45 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        convert_element_type_46: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_92, torch.bfloat16);  primals_92 = None
        convolution_15: "bf16[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.convolution.default(relu_11, convert_element_type_46, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_80: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_93, 1)
        convert_element_type_47: "f32[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_15, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_47, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_47 = None
        getitem_30: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_15[0]
        getitem_31: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        add_81: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
        rsqrt_15: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
        sub_15: "f32[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_15, getitem_31)
        mul_105: "f32[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        squeeze_45: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
        squeeze_46: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_106: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
        mul_107: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_94, 0.9)
        add_82: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
        squeeze_47: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
        mul_108: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0001627869119323);  squeeze_47 = None
        mul_109: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_108, 0.1);  mul_108 = None
        mul_110: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_95, 0.9)
        add_83: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
        unsqueeze_60: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_96, -1)
        unsqueeze_61: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_111: "f32[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_61);  mul_105 = unsqueeze_61 = None
        unsqueeze_62: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_97, -1);  primals_97 = None
        unsqueeze_63: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_84: "f32[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_63);  mul_111 = unsqueeze_63 = None
        convert_element_type_48: "bf16[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_84, torch.bfloat16);  add_84 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:56 in forward, code: out += self.shortcut(x)
        add_85: "bf16[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_48, relu_10);  convert_element_type_48 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:57 in forward, code: out = F.relu(out)
        relu_12: "bf16[96, 256, 8, 8][16384, 64, 8, 1]cuda:0" = torch.ops.aten.relu.default(add_85);  add_85 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        convert_element_type_49: "bf16[512, 256, 3, 3][2304, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_98, torch.bfloat16);  primals_98 = None
        convolution_16: "bf16[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.convolution.default(relu_12, convert_element_type_49, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
        add_86: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_99, 1)
        convert_element_type_50: "f32[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_16, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_50, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_50 = None
        getitem_32: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_16[0]
        getitem_33: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        add_87: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
        rsqrt_16: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
        sub_16: "f32[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_16, getitem_33)
        mul_112: "f32[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        squeeze_48: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        squeeze_49: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_113: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
        mul_114: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_100, 0.9)
        add_88: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
        squeeze_50: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
        mul_115: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0006514657980456);  squeeze_50 = None
        mul_116: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_115, 0.1);  mul_115 = None
        mul_117: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_101, 0.9)
        add_89: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
        unsqueeze_64: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_102, -1)
        unsqueeze_65: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        mul_118: "f32[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_65);  mul_112 = unsqueeze_65 = None
        unsqueeze_66: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_103, -1);  primals_103 = None
        unsqueeze_67: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        add_90: "f32[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_67);  mul_118 = unsqueeze_67 = None
        convert_element_type_51: "bf16[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_90, torch.bfloat16);  add_90 = None
        relu_13: "bf16[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_51);  convert_element_type_51 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        convert_element_type_52: "bf16[512, 512, 3, 3][4608, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_104, torch.bfloat16);  primals_104 = None
        convolution_17: "bf16[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.convolution.default(relu_13, convert_element_type_52, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_91: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_105, 1)
        convert_element_type_53: "f32[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_17, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_53, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_53 = None
        getitem_34: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_17[0]
        getitem_35: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        add_92: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
        rsqrt_17: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        sub_17: "f32[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_17, getitem_35)
        mul_119: "f32[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        squeeze_51: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
        squeeze_52: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
        mul_120: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
        mul_121: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_106, 0.9)
        add_93: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_120, mul_121);  mul_120 = mul_121 = None
        squeeze_53: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
        mul_122: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0006514657980456);  squeeze_53 = None
        mul_123: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_122, 0.1);  mul_122 = None
        mul_124: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_107, 0.9)
        add_94: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
        unsqueeze_68: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_108, -1)
        unsqueeze_69: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_125: "f32[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_69);  mul_119 = unsqueeze_69 = None
        unsqueeze_70: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_109, -1);  primals_109 = None
        unsqueeze_71: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_95: "f32[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_71);  mul_125 = unsqueeze_71 = None
        convert_element_type_54: "bf16[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_95, torch.bfloat16);  add_95 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:56 in forward, code: out += self.shortcut(x)
        convert_element_type_55: "bf16[512, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_110, torch.bfloat16);  primals_110 = None
        convolution_18: "bf16[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.convolution.default(relu_12, convert_element_type_55, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        add_96: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_111, 1)
        convert_element_type_56: "f32[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_18, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_56, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_56 = None
        getitem_36: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_18[0]
        getitem_37: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        add_97: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
        rsqrt_18: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
        sub_18: "f32[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_18, getitem_37)
        mul_126: "f32[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        squeeze_54: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
        squeeze_55: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
        mul_127: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
        mul_128: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_112, 0.9)
        add_98: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
        squeeze_56: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
        mul_129: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0006514657980456);  squeeze_56 = None
        mul_130: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_129, 0.1);  mul_129 = None
        mul_131: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_113, 0.9)
        add_99: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
        unsqueeze_72: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_114, -1)
        unsqueeze_73: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        mul_132: "f32[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_126, unsqueeze_73);  mul_126 = unsqueeze_73 = None
        unsqueeze_74: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_115, -1);  primals_115 = None
        unsqueeze_75: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        add_100: "f32[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_132, unsqueeze_75);  mul_132 = unsqueeze_75 = None
        convert_element_type_57: "bf16[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_100, torch.bfloat16);  add_100 = None
        add_101: "bf16[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_54, convert_element_type_57);  convert_element_type_54 = convert_element_type_57 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:57 in forward, code: out = F.relu(out)
        relu_14: "bf16[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.relu.default(add_101);  add_101 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        convert_element_type_58: "bf16[512, 512, 3, 3][4608, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_116, torch.bfloat16);  primals_116 = None
        convolution_19: "bf16[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.convolution.default(relu_14, convert_element_type_58, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_102: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_117, 1)
        convert_element_type_59: "f32[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_19, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_59, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_59 = None
        getitem_38: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_19[0]
        getitem_39: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        add_103: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_38, 1e-05)
        rsqrt_19: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
        sub_19: "f32[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_19, getitem_39)
        mul_133: "f32[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        squeeze_57: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
        squeeze_58: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
        mul_134: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
        mul_135: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_118, 0.9)
        add_104: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
        squeeze_59: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
        mul_136: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0006514657980456);  squeeze_59 = None
        mul_137: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_136, 0.1);  mul_136 = None
        mul_138: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_119, 0.9)
        add_105: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
        unsqueeze_76: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_120, -1)
        unsqueeze_77: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_139: "f32[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_77);  mul_133 = unsqueeze_77 = None
        unsqueeze_78: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_121, -1);  primals_121 = None
        unsqueeze_79: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_106: "f32[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_139, unsqueeze_79);  mul_139 = unsqueeze_79 = None
        convert_element_type_60: "bf16[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_106, torch.bfloat16);  add_106 = None
        relu_15: "bf16[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_60);  convert_element_type_60 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        convert_element_type_61: "bf16[512, 512, 3, 3][4608, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_122, torch.bfloat16);  primals_122 = None
        convolution_20: "bf16[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.convolution.default(relu_15, convert_element_type_61, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_107: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_123, 1)
        convert_element_type_62: "f32[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_20, torch.float32)
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_62, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_62 = None
        getitem_40: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_20[0]
        getitem_41: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None
        add_108: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_40, 1e-05)
        rsqrt_20: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
        sub_20: "f32[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_20, getitem_41)
        mul_140: "f32[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
        squeeze_60: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
        squeeze_61: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
        mul_141: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
        mul_142: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_124, 0.9)
        add_109: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None
        squeeze_62: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
        mul_143: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0006514657980456);  squeeze_62 = None
        mul_144: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_143, 0.1);  mul_143 = None
        mul_145: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_125, 0.9)
        add_110: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
        unsqueeze_80: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_126, -1)
        unsqueeze_81: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        mul_146: "f32[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_140, unsqueeze_81);  mul_140 = unsqueeze_81 = None
        unsqueeze_82: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_127, -1);  primals_127 = None
        unsqueeze_83: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        add_111: "f32[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_83);  mul_146 = unsqueeze_83 = None
        convert_element_type_63: "bf16[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_111, torch.bfloat16);  add_111 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:56 in forward, code: out += self.shortcut(x)
        add_112: "bf16[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_63, relu_14);  convert_element_type_63 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:57 in forward, code: out = F.relu(out)
        relu_16: "bf16[96, 512, 4, 4][8192, 16, 4, 1]cuda:0" = torch.ops.aten.relu.default(add_112);  add_112 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:133 in forward, code: x = F.avg_pool2d(x, 4)
        avg_pool2d: "bf16[96, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.avg_pool2d.default(relu_16, [4, 4])

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:134 in forward, code: x = x.view(x.size(0), -1)
        view: "bf16[96, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(avg_pool2d, [96, -1]);  avg_pool2d = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:135 in forward, code: x = self.fc(x)
        convert_element_type_64: "bf16[65][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_129, torch.bfloat16);  primals_129 = None
        convert_element_type_65: "bf16[65, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_128, torch.bfloat16);  primals_128 = None
        permute: "bf16[512, 65][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_65, [1, 0]);  convert_element_type_65 = None
        addmm: "bf16[96, 65][65, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_64, view, permute);  convert_element_type_64 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:136 in forward, code: x = torch.sigmoid(x)
        sigmoid: "bf16[96, 65][65, 1]cuda:0" = torch.ops.aten.sigmoid.default(addmm);  addmm = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:135 in forward, code: x = self.fc(x)
        permute_1: "bf16[65, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        unsqueeze_84: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
        unsqueeze_85: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_84, 2);  unsqueeze_84 = None
        unsqueeze_86: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_85, 3);  unsqueeze_85 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        unsqueeze_96: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
        unsqueeze_97: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_96, 2);  unsqueeze_96 = None
        unsqueeze_98: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_97, 3);  unsqueeze_97 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:56 in forward, code: out += self.shortcut(x)
        unsqueeze_108: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
        unsqueeze_109: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_108, 2);  unsqueeze_108 = None
        unsqueeze_110: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_109, 3);  unsqueeze_109 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        unsqueeze_120: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
        unsqueeze_121: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_120, 2);  unsqueeze_120 = None
        unsqueeze_122: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_121, 3);  unsqueeze_121 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        unsqueeze_132: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
        unsqueeze_133: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_132, 2);  unsqueeze_132 = None
        unsqueeze_134: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_133, 3);  unsqueeze_133 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        unsqueeze_144: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
        unsqueeze_145: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_144, 2);  unsqueeze_144 = None
        unsqueeze_146: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_145, 3);  unsqueeze_145 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        unsqueeze_156: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
        unsqueeze_157: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_156, 2);  unsqueeze_156 = None
        unsqueeze_158: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_157, 3);  unsqueeze_157 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:56 in forward, code: out += self.shortcut(x)
        unsqueeze_168: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
        unsqueeze_169: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_168, 2);  unsqueeze_168 = None
        unsqueeze_170: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_169, 3);  unsqueeze_169 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        unsqueeze_180: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
        unsqueeze_181: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_180, 2);  unsqueeze_180 = None
        unsqueeze_182: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_181, 3);  unsqueeze_181 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        unsqueeze_192: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
        unsqueeze_193: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_192, 2);  unsqueeze_192 = None
        unsqueeze_194: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_193, 3);  unsqueeze_193 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        unsqueeze_204: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
        unsqueeze_205: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_204, 2);  unsqueeze_204 = None
        unsqueeze_206: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 3);  unsqueeze_205 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        unsqueeze_216: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
        unsqueeze_217: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_216, 2);  unsqueeze_216 = None
        unsqueeze_218: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 3);  unsqueeze_217 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:56 in forward, code: out += self.shortcut(x)
        unsqueeze_228: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
        unsqueeze_229: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_228, 2);  unsqueeze_228 = None
        unsqueeze_230: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 3);  unsqueeze_229 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        unsqueeze_240: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
        unsqueeze_241: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_240, 2);  unsqueeze_240 = None
        unsqueeze_242: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 3);  unsqueeze_241 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        unsqueeze_252: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
        unsqueeze_253: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_252, 2);  unsqueeze_252 = None
        unsqueeze_254: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 3);  unsqueeze_253 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        unsqueeze_264: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
        unsqueeze_265: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 2);  unsqueeze_264 = None
        unsqueeze_266: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 3);  unsqueeze_265 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        unsqueeze_276: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
        unsqueeze_277: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 2);  unsqueeze_276 = None
        unsqueeze_278: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 3);  unsqueeze_277 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:56 in forward, code: out += self.shortcut(x)
        unsqueeze_288: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
        unsqueeze_289: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 2);  unsqueeze_288 = None
        unsqueeze_290: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 3);  unsqueeze_289 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        unsqueeze_300: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
        unsqueeze_301: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 2);  unsqueeze_300 = None
        unsqueeze_302: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 3);  unsqueeze_301 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        unsqueeze_312: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
        unsqueeze_313: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 2);  unsqueeze_312 = None
        unsqueeze_314: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 3);  unsqueeze_313 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:128 in forward, code: x = F.relu(self.bn1(self.conv1(x)))
        unsqueeze_324: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_325: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_324, 2);  unsqueeze_324 = None
        unsqueeze_326: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 3);  unsqueeze_325 = None

        # No stacktrace found for following nodes
        copy_: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_3, add);  primals_3 = add = copy_ = None
        copy__1: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_4, add_2);  primals_4 = add_2 = copy__1 = None
        copy__2: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_5, add_3);  primals_5 = add_3 = copy__2 = None
        copy__3: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_9, add_5);  primals_9 = add_5 = copy__3 = None
        copy__4: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_10, add_7);  primals_10 = add_7 = copy__4 = None
        copy__5: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_11, add_8);  primals_11 = add_8 = copy__5 = None
        copy__6: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_15, add_10);  primals_15 = add_10 = copy__6 = None
        copy__7: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_16, add_12);  primals_16 = add_12 = copy__7 = None
        copy__8: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_17, add_13);  primals_17 = add_13 = copy__8 = None
        copy__9: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_21, add_15);  primals_21 = add_15 = copy__9 = None
        copy__10: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_22, add_17);  primals_22 = add_17 = copy__10 = None
        copy__11: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_23, add_18);  primals_23 = add_18 = copy__11 = None
        copy__12: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_27, add_21);  primals_27 = add_21 = copy__12 = None
        copy__13: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_28, add_23);  primals_28 = add_23 = copy__13 = None
        copy__14: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_29, add_24);  primals_29 = add_24 = copy__14 = None
        copy__15: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_33, add_26);  primals_33 = add_26 = copy__15 = None
        copy__16: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_34, add_28);  primals_34 = add_28 = copy__16 = None
        copy__17: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_35, add_29);  primals_35 = add_29 = copy__17 = None
        copy__18: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_39, add_32);  primals_39 = add_32 = copy__18 = None
        copy__19: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_40, add_34);  primals_40 = add_34 = copy__19 = None
        copy__20: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_41, add_35);  primals_41 = add_35 = copy__20 = None
        copy__21: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_45, add_37);  primals_45 = add_37 = copy__21 = None
        copy__22: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_46, add_39);  primals_46 = add_39 = copy__22 = None
        copy__23: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_47, add_40);  primals_47 = add_40 = copy__23 = None
        copy__24: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_51, add_42);  primals_51 = add_42 = copy__24 = None
        copy__25: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_52, add_44);  primals_52 = add_44 = copy__25 = None
        copy__26: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_53, add_45);  primals_53 = add_45 = copy__26 = None
        copy__27: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_57, add_48);  primals_57 = add_48 = copy__27 = None
        copy__28: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_58, add_50);  primals_58 = add_50 = copy__28 = None
        copy__29: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_59, add_51);  primals_59 = add_51 = copy__29 = None
        copy__30: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_63, add_53);  primals_63 = add_53 = copy__30 = None
        copy__31: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_64, add_55);  primals_64 = add_55 = copy__31 = None
        copy__32: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_65, add_56);  primals_65 = add_56 = copy__32 = None
        copy__33: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_69, add_59);  primals_69 = add_59 = copy__33 = None
        copy__34: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_70, add_61);  primals_70 = add_61 = copy__34 = None
        copy__35: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_71, add_62);  primals_71 = add_62 = copy__35 = None
        copy__36: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_75, add_64);  primals_75 = add_64 = copy__36 = None
        copy__37: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_76, add_66);  primals_76 = add_66 = copy__37 = None
        copy__38: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_77, add_67);  primals_77 = add_67 = copy__38 = None
        copy__39: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_81, add_69);  primals_81 = add_69 = copy__39 = None
        copy__40: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_82, add_71);  primals_82 = add_71 = copy__40 = None
        copy__41: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_83, add_72);  primals_83 = add_72 = copy__41 = None
        copy__42: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_87, add_75);  primals_87 = add_75 = copy__42 = None
        copy__43: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_88, add_77);  primals_88 = add_77 = copy__43 = None
        copy__44: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_89, add_78);  primals_89 = add_78 = copy__44 = None
        copy__45: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_93, add_80);  primals_93 = add_80 = copy__45 = None
        copy__46: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_94, add_82);  primals_94 = add_82 = copy__46 = None
        copy__47: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_95, add_83);  primals_95 = add_83 = copy__47 = None
        copy__48: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_99, add_86);  primals_99 = add_86 = copy__48 = None
        copy__49: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_100, add_88);  primals_100 = add_88 = copy__49 = None
        copy__50: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_101, add_89);  primals_101 = add_89 = copy__50 = None
        copy__51: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_105, add_91);  primals_105 = add_91 = copy__51 = None
        copy__52: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_106, add_93);  primals_106 = add_93 = copy__52 = None
        copy__53: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_107, add_94);  primals_107 = add_94 = copy__53 = None
        copy__54: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_111, add_96);  primals_111 = add_96 = copy__54 = None
        copy__55: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_112, add_98);  primals_112 = add_98 = copy__55 = None
        copy__56: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_113, add_99);  primals_113 = add_99 = copy__56 = None
        copy__57: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_117, add_102);  primals_117 = add_102 = copy__57 = None
        copy__58: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_118, add_104);  primals_118 = add_104 = copy__58 = None
        copy__59: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_119, add_105);  primals_119 = add_105 = copy__59 = None
        copy__60: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_123, add_107);  primals_123 = add_107 = copy__60 = None
        copy__61: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_124, add_109);  primals_124 = add_109 = copy__61 = None
        copy__62: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_125, add_110);  primals_125 = add_110 = copy__62 = None
        return (sigmoid, primals_6, primals_12, primals_18, primals_24, primals_30, primals_36, primals_42, primals_48, primals_54, primals_60, primals_66, primals_72, primals_78, primals_84, primals_90, primals_96, primals_102, primals_108, primals_114, primals_120, primals_126, convert_element_type, convert_element_type_1, convolution, squeeze_1, relu, convert_element_type_4, convolution_1, squeeze_4, relu_1, convert_element_type_7, convolution_2, squeeze_7, convert_element_type_10, convolution_3, squeeze_10, relu_2, convert_element_type_13, convolution_4, squeeze_13, relu_3, convert_element_type_16, convolution_5, squeeze_16, relu_4, convert_element_type_19, convolution_6, squeeze_19, relu_5, convert_element_type_22, convolution_7, squeeze_22, convert_element_type_25, convolution_8, squeeze_25, relu_6, convert_element_type_28, convolution_9, squeeze_28, relu_7, convert_element_type_31, convolution_10, squeeze_31, relu_8, convert_element_type_34, convolution_11, squeeze_34, relu_9, convert_element_type_37, convolution_12, squeeze_37, convert_element_type_40, convolution_13, squeeze_40, relu_10, convert_element_type_43, convolution_14, squeeze_43, relu_11, convert_element_type_46, convolution_15, squeeze_46, relu_12, convert_element_type_49, convolution_16, squeeze_49, relu_13, convert_element_type_52, convolution_17, squeeze_52, convert_element_type_55, convolution_18, squeeze_55, relu_14, convert_element_type_58, convolution_19, squeeze_58, relu_15, convert_element_type_61, convolution_20, squeeze_61, relu_16, view, sigmoid, permute_1, unsqueeze_86, unsqueeze_98, unsqueeze_110, unsqueeze_122, unsqueeze_134, unsqueeze_146, unsqueeze_158, unsqueeze_170, unsqueeze_182, unsqueeze_194, unsqueeze_206, unsqueeze_218, unsqueeze_230, unsqueeze_242, unsqueeze_254, unsqueeze_266, unsqueeze_278, unsqueeze_290, unsqueeze_302, unsqueeze_314, unsqueeze_326)
