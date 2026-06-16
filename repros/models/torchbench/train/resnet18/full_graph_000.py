class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[64, 3, 7, 7][147, 49, 7, 1]cuda:0", primals_2: "f32[16, 3, 224, 224][150528, 50176, 224, 1]cuda:0", primals_3: "i64[][]cuda:0", primals_4: "f32[64][1]cuda:0", primals_5: "f32[64][1]cuda:0", primals_6: "f32[64][1]cuda:0", primals_7: "f32[64][1]cuda:0", primals_8: "f32[64, 64, 3, 3][576, 9, 3, 1]cuda:0", primals_9: "i64[][]cuda:0", primals_10: "f32[64][1]cuda:0", primals_11: "f32[64][1]cuda:0", primals_12: "f32[64][1]cuda:0", primals_13: "f32[64][1]cuda:0", primals_14: "f32[64, 64, 3, 3][576, 9, 3, 1]cuda:0", primals_15: "i64[][]cuda:0", primals_16: "f32[64][1]cuda:0", primals_17: "f32[64][1]cuda:0", primals_18: "f32[64][1]cuda:0", primals_19: "f32[64][1]cuda:0", primals_20: "f32[64, 64, 3, 3][576, 9, 3, 1]cuda:0", primals_21: "i64[][]cuda:0", primals_22: "f32[64][1]cuda:0", primals_23: "f32[64][1]cuda:0", primals_24: "f32[64][1]cuda:0", primals_25: "f32[64][1]cuda:0", primals_26: "f32[64, 64, 3, 3][576, 9, 3, 1]cuda:0", primals_27: "i64[][]cuda:0", primals_28: "f32[64][1]cuda:0", primals_29: "f32[64][1]cuda:0", primals_30: "f32[64][1]cuda:0", primals_31: "f32[64][1]cuda:0", primals_32: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0", primals_33: "i64[][]cuda:0", primals_34: "f32[128][1]cuda:0", primals_35: "f32[128][1]cuda:0", primals_36: "f32[128][1]cuda:0", primals_37: "f32[128][1]cuda:0", primals_38: "f32[128, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_39: "i64[][]cuda:0", primals_40: "f32[128][1]cuda:0", primals_41: "f32[128][1]cuda:0", primals_42: "f32[128][1]cuda:0", primals_43: "f32[128][1]cuda:0", primals_44: "f32[128, 64, 1, 1][64, 1, 1, 1]cuda:0", primals_45: "i64[][]cuda:0", primals_46: "f32[128][1]cuda:0", primals_47: "f32[128][1]cuda:0", primals_48: "f32[128][1]cuda:0", primals_49: "f32[128][1]cuda:0", primals_50: "f32[128, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_51: "i64[][]cuda:0", primals_52: "f32[128][1]cuda:0", primals_53: "f32[128][1]cuda:0", primals_54: "f32[128][1]cuda:0", primals_55: "f32[128][1]cuda:0", primals_56: "f32[128, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_57: "i64[][]cuda:0", primals_58: "f32[128][1]cuda:0", primals_59: "f32[128][1]cuda:0", primals_60: "f32[128][1]cuda:0", primals_61: "f32[128][1]cuda:0", primals_62: "f32[256, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_63: "i64[][]cuda:0", primals_64: "f32[256][1]cuda:0", primals_65: "f32[256][1]cuda:0", primals_66: "f32[256][1]cuda:0", primals_67: "f32[256][1]cuda:0", primals_68: "f32[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", primals_69: "i64[][]cuda:0", primals_70: "f32[256][1]cuda:0", primals_71: "f32[256][1]cuda:0", primals_72: "f32[256][1]cuda:0", primals_73: "f32[256][1]cuda:0", primals_74: "f32[256, 128, 1, 1][128, 1, 1, 1]cuda:0", primals_75: "i64[][]cuda:0", primals_76: "f32[256][1]cuda:0", primals_77: "f32[256][1]cuda:0", primals_78: "f32[256][1]cuda:0", primals_79: "f32[256][1]cuda:0", primals_80: "f32[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", primals_81: "i64[][]cuda:0", primals_82: "f32[256][1]cuda:0", primals_83: "f32[256][1]cuda:0", primals_84: "f32[256][1]cuda:0", primals_85: "f32[256][1]cuda:0", primals_86: "f32[256, 256, 3, 3][2304, 9, 3, 1]cuda:0", primals_87: "i64[][]cuda:0", primals_88: "f32[256][1]cuda:0", primals_89: "f32[256][1]cuda:0", primals_90: "f32[256][1]cuda:0", primals_91: "f32[256][1]cuda:0", primals_92: "f32[512, 256, 3, 3][2304, 9, 3, 1]cuda:0", primals_93: "i64[][]cuda:0", primals_94: "f32[512][1]cuda:0", primals_95: "f32[512][1]cuda:0", primals_96: "f32[512][1]cuda:0", primals_97: "f32[512][1]cuda:0", primals_98: "f32[512, 512, 3, 3][4608, 9, 3, 1]cuda:0", primals_99: "i64[][]cuda:0", primals_100: "f32[512][1]cuda:0", primals_101: "f32[512][1]cuda:0", primals_102: "f32[512][1]cuda:0", primals_103: "f32[512][1]cuda:0", primals_104: "f32[512, 256, 1, 1][256, 1, 1, 1]cuda:0", primals_105: "i64[][]cuda:0", primals_106: "f32[512][1]cuda:0", primals_107: "f32[512][1]cuda:0", primals_108: "f32[512][1]cuda:0", primals_109: "f32[512][1]cuda:0", primals_110: "f32[512, 512, 3, 3][4608, 9, 3, 1]cuda:0", primals_111: "i64[][]cuda:0", primals_112: "f32[512][1]cuda:0", primals_113: "f32[512][1]cuda:0", primals_114: "f32[512][1]cuda:0", primals_115: "f32[512][1]cuda:0", primals_116: "f32[512, 512, 3, 3][4608, 9, 3, 1]cuda:0", primals_117: "i64[][]cuda:0", primals_118: "f32[512][1]cuda:0", primals_119: "f32[512][1]cuda:0", primals_120: "f32[512][1]cuda:0", primals_121: "f32[512][1]cuda:0", primals_122: "f32[1000, 512][512, 1]cuda:0", primals_123: "f32[1000][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:268 in _forward_impl, code: x = self.conv1(x)
        convert_element_type: "bf16[64, 3, 7, 7][147, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_1, torch.bfloat16);  primals_1 = None
        convert_element_type_1: "bf16[16, 3, 224, 224][150528, 50176, 224, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16);  primals_2 = None
        convolution: "bf16[16, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_1, convert_element_type, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:269 in _forward_impl, code: x = self.bn1(x)
        add: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_3, 1)
        convert_element_type_2: "f32[16, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_2, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_2 = None
        getitem: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        add_1: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[16, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul: "f32[16, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3])
        mul_1: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze, 0.1);  squeeze = None
        mul_2: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_4, 0.9)
        add_2: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_2: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_3: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_2, 1.0000049824865598);  squeeze_2 = None
        mul_4: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
        mul_5: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_5, 0.9)
        add_3: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_6, -1)
        unsqueeze_1: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[16, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_7, -1)
        unsqueeze_3: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_4: "f32[16, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        convert_element_type_3: "bf16[16, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:270 in _forward_impl, code: x = self.relu(x)
        relu: "bf16[16, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_3);  convert_element_type_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:271 in _forward_impl, code: x = self.maxpool(x)
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu, [3, 3], [2, 2], [1, 1], [1, 1], False);  relu = None
        getitem_2: "bf16[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = _low_memory_max_pool_with_offsets[0]
        getitem_3: "i8[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = _low_memory_max_pool_with_offsets[1];  _low_memory_max_pool_with_offsets = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convert_element_type_4: "bf16[64, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_8, torch.bfloat16);  primals_8 = None
        convolution_1: "bf16[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(getitem_2, convert_element_type_4, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        add_5: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_9, 1)
        convert_element_type_5: "f32[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_5, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_5 = None
        getitem_4: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_1[0]
        getitem_5: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        add_6: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
        rsqrt_1: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_1: "f32[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_1, getitem_5)
        mul_7: "f32[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        squeeze_3: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        squeeze_4: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_8: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
        mul_9: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_10, 0.9)
        add_7: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        squeeze_5: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_10: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000199302441455);  squeeze_5 = None
        mul_11: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
        mul_12: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_11, 0.9)
        add_8: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        unsqueeze_4: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_12, -1)
        unsqueeze_5: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_13: "f32[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
        unsqueeze_6: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_13, -1);  primals_13 = None
        unsqueeze_7: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_9: "f32[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
        convert_element_type_6: "bf16[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_9, torch.bfloat16);  add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        relu_1: "bf16[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_6);  convert_element_type_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convert_element_type_7: "bf16[64, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
        convolution_2: "bf16[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_1, convert_element_type_7, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        add_10: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_15, 1)
        convert_element_type_8: "f32[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_2, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_8, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_8 = None
        getitem_6: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_2[0]
        getitem_7: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        add_11: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
        rsqrt_2: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_2: "f32[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_2, getitem_7)
        mul_14: "f32[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        squeeze_6: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        squeeze_7: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_15: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
        mul_16: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_16, 0.9)
        add_12: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
        squeeze_8: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_17: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0000199302441455);  squeeze_8 = None
        mul_18: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
        mul_19: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_17, 0.9)
        add_13: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
        unsqueeze_8: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_18, -1)
        unsqueeze_9: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        mul_20: "f32[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
        unsqueeze_10: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_19, -1);  primals_19 = None
        unsqueeze_11: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        add_14: "f32[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
        convert_element_type_9: "bf16[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_14, torch.bfloat16);  add_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
        add_15: "bf16[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_9, getitem_2);  convert_element_type_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        relu_2: "bf16[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(add_15);  add_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convert_element_type_10: "bf16[64, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_20, torch.bfloat16);  primals_20 = None
        convolution_3: "bf16[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_2, convert_element_type_10, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        add_16: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_21, 1)
        convert_element_type_11: "f32[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_3, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_11, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_11 = None
        getitem_8: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_3[0]
        getitem_9: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        add_17: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
        rsqrt_3: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        sub_3: "f32[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_3, getitem_9)
        mul_21: "f32[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        squeeze_9: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        squeeze_10: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_22: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
        mul_23: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_22, 0.9)
        add_18: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
        squeeze_11: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_24: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0000199302441455);  squeeze_11 = None
        mul_25: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
        mul_26: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_23, 0.9)
        add_19: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
        unsqueeze_12: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_24, -1)
        unsqueeze_13: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_27: "f32[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
        unsqueeze_14: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_25, -1);  primals_25 = None
        unsqueeze_15: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_20: "f32[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
        convert_element_type_12: "bf16[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_20, torch.bfloat16);  add_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        relu_3: "bf16[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_12);  convert_element_type_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convert_element_type_13: "bf16[64, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_26, torch.bfloat16);  primals_26 = None
        convolution_4: "bf16[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_3, convert_element_type_13, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        add_21: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_27, 1)
        convert_element_type_14: "f32[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_4, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_14, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_14 = None
        getitem_10: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_4[0]
        getitem_11: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        add_22: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
        rsqrt_4: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        sub_4: "f32[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_4, getitem_11)
        mul_28: "f32[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        squeeze_12: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        squeeze_13: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_29: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
        mul_30: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_28, 0.9)
        add_23: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
        squeeze_14: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_31: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000199302441455);  squeeze_14 = None
        mul_32: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
        mul_33: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_29, 0.9)
        add_24: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
        unsqueeze_16: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_30, -1)
        unsqueeze_17: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        mul_34: "f32[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
        unsqueeze_18: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_31, -1);  primals_31 = None
        unsqueeze_19: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        add_25: "f32[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None
        convert_element_type_15: "bf16[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_25, torch.bfloat16);  add_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
        add_26: "bf16[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_15, relu_2);  convert_element_type_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        relu_4: "bf16[16, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(add_26);  add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convert_element_type_16: "bf16[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_32, torch.bfloat16);  primals_32 = None
        convolution_5: "bf16[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_4, convert_element_type_16, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        add_27: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_33, 1)
        convert_element_type_17: "f32[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_5, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_17, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_17 = None
        getitem_12: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_5[0]
        getitem_13: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        add_28: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
        rsqrt_5: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_5: "f32[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_5, getitem_13)
        mul_35: "f32[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        squeeze_15: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        squeeze_16: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_36: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
        mul_37: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_34, 0.9)
        add_29: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
        squeeze_17: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
        mul_38: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0000797257434426);  squeeze_17 = None
        mul_39: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
        mul_40: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_35, 0.9)
        add_30: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
        unsqueeze_20: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_36, -1)
        unsqueeze_21: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_41: "f32[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
        unsqueeze_22: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_37, -1);  primals_37 = None
        unsqueeze_23: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_31: "f32[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
        convert_element_type_18: "bf16[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_31, torch.bfloat16);  add_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        relu_5: "bf16[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_18);  convert_element_type_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convert_element_type_19: "bf16[128, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_38, torch.bfloat16);  primals_38 = None
        convolution_6: "bf16[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_5, convert_element_type_19, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        add_32: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_39, 1)
        convert_element_type_20: "f32[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_6, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_20, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_20 = None
        getitem_14: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_6[0]
        getitem_15: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        add_33: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
        rsqrt_6: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        sub_6: "f32[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_6, getitem_15)
        mul_42: "f32[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        squeeze_18: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        squeeze_19: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_43: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
        mul_44: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_40, 0.9)
        add_34: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
        squeeze_20: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
        mul_45: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0000797257434426);  squeeze_20 = None
        mul_46: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_45, 0.1);  mul_45 = None
        mul_47: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_41, 0.9)
        add_35: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
        unsqueeze_24: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_42, -1)
        unsqueeze_25: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        mul_48: "f32[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_25);  mul_42 = unsqueeze_25 = None
        unsqueeze_26: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_43, -1);  primals_43 = None
        unsqueeze_27: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        add_36: "f32[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_48, unsqueeze_27);  mul_48 = unsqueeze_27 = None
        convert_element_type_21: "bf16[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_36, torch.bfloat16);  add_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:100 in forward, code: identity = self.downsample(x)
        convert_element_type_22: "bf16[128, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16);  primals_44 = None
        convolution_7: "bf16[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_4, convert_element_type_22, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        add_37: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_45, 1)
        convert_element_type_23: "f32[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_7, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_23, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_23 = None
        getitem_16: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_7[0]
        getitem_17: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        add_38: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
        rsqrt_7: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        sub_7: "f32[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_7, getitem_17)
        mul_49: "f32[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        squeeze_21: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
        squeeze_22: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
        mul_50: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
        mul_51: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_46, 0.9)
        add_39: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
        squeeze_23: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
        mul_52: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0000797257434426);  squeeze_23 = None
        mul_53: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_52, 0.1);  mul_52 = None
        mul_54: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_47, 0.9)
        add_40: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
        unsqueeze_28: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_48, -1)
        unsqueeze_29: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_55: "f32[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
        unsqueeze_30: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_49, -1);  primals_49 = None
        unsqueeze_31: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_41: "f32[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None
        convert_element_type_24: "bf16[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_41, torch.bfloat16);  add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
        add_42: "bf16[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_21, convert_element_type_24);  convert_element_type_21 = convert_element_type_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        relu_6: "bf16[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(add_42);  add_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convert_element_type_25: "bf16[128, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_50, torch.bfloat16);  primals_50 = None
        convolution_8: "bf16[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_6, convert_element_type_25, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        add_43: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_51, 1)
        convert_element_type_26: "f32[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_8, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_26, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_26 = None
        getitem_18: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_8[0]
        getitem_19: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        add_44: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
        rsqrt_8: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_8: "f32[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_8, getitem_19)
        mul_56: "f32[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        squeeze_24: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        squeeze_25: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_57: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
        mul_58: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_52, 0.9)
        add_45: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
        squeeze_26: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
        mul_59: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0000797257434426);  squeeze_26 = None
        mul_60: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
        mul_61: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_53, 0.9)
        add_46: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
        unsqueeze_32: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_54, -1)
        unsqueeze_33: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        mul_62: "f32[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
        unsqueeze_34: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_55, -1);  primals_55 = None
        unsqueeze_35: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        add_47: "f32[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None
        convert_element_type_27: "bf16[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_47, torch.bfloat16);  add_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        relu_7: "bf16[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_27);  convert_element_type_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convert_element_type_28: "bf16[128, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_56, torch.bfloat16);  primals_56 = None
        convolution_9: "bf16[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_7, convert_element_type_28, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        add_48: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_57, 1)
        convert_element_type_29: "f32[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_9, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_29, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_29 = None
        getitem_20: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_9[0]
        getitem_21: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        add_49: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
        rsqrt_9: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        sub_9: "f32[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_9, getitem_21)
        mul_63: "f32[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        squeeze_27: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        squeeze_28: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_64: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
        mul_65: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_58, 0.9)
        add_50: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
        squeeze_29: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        mul_66: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0000797257434426);  squeeze_29 = None
        mul_67: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, 0.1);  mul_66 = None
        mul_68: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_59, 0.9)
        add_51: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
        unsqueeze_36: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_60, -1)
        unsqueeze_37: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_69: "f32[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_37);  mul_63 = unsqueeze_37 = None
        unsqueeze_38: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_61, -1);  primals_61 = None
        unsqueeze_39: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_52: "f32[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_39);  mul_69 = unsqueeze_39 = None
        convert_element_type_30: "bf16[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_52, torch.bfloat16);  add_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
        add_53: "bf16[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_30, relu_6);  convert_element_type_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        relu_8: "bf16[16, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(add_53);  add_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convert_element_type_31: "bf16[256, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_62, torch.bfloat16);  primals_62 = None
        convolution_10: "bf16[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_8, convert_element_type_31, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        add_54: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_63, 1)
        convert_element_type_32: "f32[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_10, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_32, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_32 = None
        getitem_22: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_10[0]
        getitem_23: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        add_55: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
        rsqrt_10: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
        sub_10: "f32[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_10, getitem_23)
        mul_70: "f32[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        squeeze_30: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
        squeeze_31: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_71: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
        mul_72: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_64, 0.9)
        add_56: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_71, mul_72);  mul_71 = mul_72 = None
        squeeze_32: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
        mul_73: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0003189792663476);  squeeze_32 = None
        mul_74: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_73, 0.1);  mul_73 = None
        mul_75: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_65, 0.9)
        add_57: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
        unsqueeze_40: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_66, -1)
        unsqueeze_41: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        mul_76: "f32[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_41);  mul_70 = unsqueeze_41 = None
        unsqueeze_42: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_67, -1);  primals_67 = None
        unsqueeze_43: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        add_58: "f32[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_76, unsqueeze_43);  mul_76 = unsqueeze_43 = None
        convert_element_type_33: "bf16[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_58, torch.bfloat16);  add_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        relu_9: "bf16[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_33);  convert_element_type_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convert_element_type_34: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_68, torch.bfloat16);  primals_68 = None
        convolution_11: "bf16[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_9, convert_element_type_34, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        add_59: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_69, 1)
        convert_element_type_35: "f32[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_11, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_35, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_35 = None
        getitem_24: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_11[0]
        getitem_25: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        add_60: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
        rsqrt_11: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_11: "f32[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_11, getitem_25)
        mul_77: "f32[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        squeeze_33: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        squeeze_34: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
        mul_78: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
        mul_79: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_70, 0.9)
        add_61: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
        squeeze_35: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
        mul_80: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0003189792663476);  squeeze_35 = None
        mul_81: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_80, 0.1);  mul_80 = None
        mul_82: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_71, 0.9)
        add_62: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
        unsqueeze_44: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_72, -1)
        unsqueeze_45: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_83: "f32[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_45);  mul_77 = unsqueeze_45 = None
        unsqueeze_46: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_73, -1);  primals_73 = None
        unsqueeze_47: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_63: "f32[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_47);  mul_83 = unsqueeze_47 = None
        convert_element_type_36: "bf16[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_63, torch.bfloat16);  add_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:100 in forward, code: identity = self.downsample(x)
        convert_element_type_37: "bf16[256, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_74, torch.bfloat16);  primals_74 = None
        convolution_12: "bf16[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_8, convert_element_type_37, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        add_64: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_75, 1)
        convert_element_type_38: "f32[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_12, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_38, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_38 = None
        getitem_26: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_12[0]
        getitem_27: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        add_65: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
        rsqrt_12: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        sub_12: "f32[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_12, getitem_27)
        mul_84: "f32[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        squeeze_36: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        squeeze_37: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_85: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
        mul_86: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_76, 0.9)
        add_66: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
        squeeze_38: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        mul_87: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0003189792663476);  squeeze_38 = None
        mul_88: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_87, 0.1);  mul_87 = None
        mul_89: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_77, 0.9)
        add_67: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
        unsqueeze_48: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_78, -1)
        unsqueeze_49: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        mul_90: "f32[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_49);  mul_84 = unsqueeze_49 = None
        unsqueeze_50: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_79, -1);  primals_79 = None
        unsqueeze_51: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        add_68: "f32[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_51);  mul_90 = unsqueeze_51 = None
        convert_element_type_39: "bf16[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_68, torch.bfloat16);  add_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
        add_69: "bf16[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_36, convert_element_type_39);  convert_element_type_36 = convert_element_type_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        relu_10: "bf16[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(add_69);  add_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convert_element_type_40: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_80, torch.bfloat16);  primals_80 = None
        convolution_13: "bf16[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_10, convert_element_type_40, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        add_70: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_81, 1)
        convert_element_type_41: "f32[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_13, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_41, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_41 = None
        getitem_28: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_13[0]
        getitem_29: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        add_71: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
        rsqrt_13: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        sub_13: "f32[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_13, getitem_29)
        mul_91: "f32[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        squeeze_39: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        squeeze_40: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
        mul_92: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
        mul_93: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_82, 0.9)
        add_72: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
        squeeze_41: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        mul_94: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0003189792663476);  squeeze_41 = None
        mul_95: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_94, 0.1);  mul_94 = None
        mul_96: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_83, 0.9)
        add_73: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None
        unsqueeze_52: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_84, -1)
        unsqueeze_53: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_97: "f32[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_53);  mul_91 = unsqueeze_53 = None
        unsqueeze_54: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_85, -1);  primals_85 = None
        unsqueeze_55: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_74: "f32[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_55);  mul_97 = unsqueeze_55 = None
        convert_element_type_42: "bf16[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_74, torch.bfloat16);  add_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        relu_11: "bf16[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_42);  convert_element_type_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convert_element_type_43: "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_86, torch.bfloat16);  primals_86 = None
        convolution_14: "bf16[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_11, convert_element_type_43, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        add_75: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_87, 1)
        convert_element_type_44: "f32[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_14, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_44, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_44 = None
        getitem_30: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_14[0]
        getitem_31: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        add_76: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
        rsqrt_14: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        sub_14: "f32[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_14, getitem_31)
        mul_98: "f32[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        squeeze_42: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
        squeeze_43: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_99: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
        mul_100: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_88, 0.9)
        add_77: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
        squeeze_44: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
        mul_101: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0003189792663476);  squeeze_44 = None
        mul_102: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_101, 0.1);  mul_101 = None
        mul_103: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_89, 0.9)
        add_78: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
        unsqueeze_56: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_90, -1)
        unsqueeze_57: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        mul_104: "f32[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_57);  mul_98 = unsqueeze_57 = None
        unsqueeze_58: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_91, -1);  primals_91 = None
        unsqueeze_59: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        add_79: "f32[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_59);  mul_104 = unsqueeze_59 = None
        convert_element_type_45: "bf16[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_79, torch.bfloat16);  add_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
        add_80: "bf16[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_45, relu_10);  convert_element_type_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        relu_12: "bf16[16, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(add_80);  add_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convert_element_type_46: "bf16[512, 256, 3, 3][2304, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_92, torch.bfloat16);  primals_92 = None
        convolution_15: "bf16[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_12, convert_element_type_46, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        add_81: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_93, 1)
        convert_element_type_47: "f32[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_15, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_47, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_47 = None
        getitem_32: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_15[0]
        getitem_33: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        add_82: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
        rsqrt_15: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_15: "f32[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_15, getitem_33)
        mul_105: "f32[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        squeeze_45: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        squeeze_46: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_106: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
        mul_107: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_94, 0.9)
        add_83: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
        squeeze_47: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
        mul_108: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0012771392081736);  squeeze_47 = None
        mul_109: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_108, 0.1);  mul_108 = None
        mul_110: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_95, 0.9)
        add_84: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
        unsqueeze_60: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_96, -1)
        unsqueeze_61: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_111: "f32[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_61);  mul_105 = unsqueeze_61 = None
        unsqueeze_62: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_97, -1);  primals_97 = None
        unsqueeze_63: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_85: "f32[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_63);  mul_111 = unsqueeze_63 = None
        convert_element_type_48: "bf16[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_85, torch.bfloat16);  add_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        relu_13: "bf16[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_48);  convert_element_type_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convert_element_type_49: "bf16[512, 512, 3, 3][4608, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_98, torch.bfloat16);  primals_98 = None
        convolution_16: "bf16[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_13, convert_element_type_49, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        add_86: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_99, 1)
        convert_element_type_50: "f32[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_16, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_50, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_50 = None
        getitem_34: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_16[0]
        getitem_35: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        add_87: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
        rsqrt_16: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
        sub_16: "f32[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_16, getitem_35)
        mul_112: "f32[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        squeeze_48: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
        squeeze_49: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_113: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
        mul_114: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_100, 0.9)
        add_88: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
        squeeze_50: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
        mul_115: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0012771392081736);  squeeze_50 = None
        mul_116: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_115, 0.1);  mul_115 = None
        mul_117: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_101, 0.9)
        add_89: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
        unsqueeze_64: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_102, -1)
        unsqueeze_65: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        mul_118: "f32[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_65);  mul_112 = unsqueeze_65 = None
        unsqueeze_66: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_103, -1);  primals_103 = None
        unsqueeze_67: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        add_90: "f32[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_67);  mul_118 = unsqueeze_67 = None
        convert_element_type_51: "bf16[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_90, torch.bfloat16);  add_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:100 in forward, code: identity = self.downsample(x)
        convert_element_type_52: "bf16[512, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_104, torch.bfloat16);  primals_104 = None
        convolution_17: "bf16[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_12, convert_element_type_52, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        add_91: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_105, 1)
        convert_element_type_53: "f32[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_17, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_53, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_53 = None
        getitem_36: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_17[0]
        getitem_37: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        add_92: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
        rsqrt_17: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        sub_17: "f32[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_17, getitem_37)
        mul_119: "f32[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        squeeze_51: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
        squeeze_52: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
        mul_120: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
        mul_121: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_106, 0.9)
        add_93: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_120, mul_121);  mul_120 = mul_121 = None
        squeeze_53: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
        mul_122: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0012771392081736);  squeeze_53 = None
        mul_123: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_122, 0.1);  mul_122 = None
        mul_124: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_107, 0.9)
        add_94: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
        unsqueeze_68: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_108, -1)
        unsqueeze_69: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_125: "f32[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_69);  mul_119 = unsqueeze_69 = None
        unsqueeze_70: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_109, -1);  primals_109 = None
        unsqueeze_71: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_95: "f32[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_71);  mul_125 = unsqueeze_71 = None
        convert_element_type_54: "bf16[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_95, torch.bfloat16);  add_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
        add_96: "bf16[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_51, convert_element_type_54);  convert_element_type_51 = convert_element_type_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        relu_14: "bf16[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(add_96);  add_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convert_element_type_55: "bf16[512, 512, 3, 3][4608, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_110, torch.bfloat16);  primals_110 = None
        convolution_18: "bf16[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_14, convert_element_type_55, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        add_97: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_111, 1)
        convert_element_type_56: "f32[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_18, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_56, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_56 = None
        getitem_38: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_18[0]
        getitem_39: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        add_98: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_38, 1e-05)
        rsqrt_18: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        sub_18: "f32[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_18, getitem_39)
        mul_126: "f32[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        squeeze_54: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
        squeeze_55: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
        mul_127: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
        mul_128: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_112, 0.9)
        add_99: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
        squeeze_56: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
        mul_129: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0012771392081736);  squeeze_56 = None
        mul_130: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_129, 0.1);  mul_129 = None
        mul_131: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_113, 0.9)
        add_100: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
        unsqueeze_72: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_114, -1)
        unsqueeze_73: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        mul_132: "f32[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_126, unsqueeze_73);  mul_126 = unsqueeze_73 = None
        unsqueeze_74: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_115, -1);  primals_115 = None
        unsqueeze_75: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        add_101: "f32[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_132, unsqueeze_75);  mul_132 = unsqueeze_75 = None
        convert_element_type_57: "bf16[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_101, torch.bfloat16);  add_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        relu_15: "bf16[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_57);  convert_element_type_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convert_element_type_58: "bf16[512, 512, 3, 3][4608, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_116, torch.bfloat16);  primals_116 = None
        convolution_19: "bf16[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_15, convert_element_type_58, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        add_102: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_117, 1)
        convert_element_type_59: "f32[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_19, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_59, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_59 = None
        getitem_40: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_19[0]
        getitem_41: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        add_103: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_40, 1e-05)
        rsqrt_19: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
        sub_19: "f32[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_19, getitem_41)
        mul_133: "f32[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        squeeze_57: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
        squeeze_58: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
        mul_134: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
        mul_135: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_118, 0.9)
        add_104: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
        squeeze_59: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
        mul_136: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0012771392081736);  squeeze_59 = None
        mul_137: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_136, 0.1);  mul_136 = None
        mul_138: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_119, 0.9)
        add_105: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
        unsqueeze_76: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_120, -1)
        unsqueeze_77: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_139: "f32[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_77);  mul_133 = unsqueeze_77 = None
        unsqueeze_78: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_121, -1);  primals_121 = None
        unsqueeze_79: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_106: "f32[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_139, unsqueeze_79);  mul_139 = unsqueeze_79 = None
        convert_element_type_60: "bf16[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_106, torch.bfloat16);  add_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
        add_107: "bf16[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_60, relu_14);  convert_element_type_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        relu_16: "bf16[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(add_107);  add_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:278 in _forward_impl, code: x = self.avgpool(x)
        mean: "bf16[16, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(relu_16, [-1, -2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:279 in _forward_impl, code: x = torch.flatten(x, 1)
        view: "bf16[16, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mean, [16, 512]);  mean = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:280 in _forward_impl, code: x = self.fc(x)
        convert_element_type_61: "bf16[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_123, torch.bfloat16);  primals_123 = None
        convert_element_type_62: "bf16[1000, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_122, torch.bfloat16);  primals_122 = None
        permute: "bf16[512, 1000][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_62, [1, 0]);  convert_element_type_62 = None
        addmm: "bf16[16, 1000][1000, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_61, view, permute);  convert_element_type_61 = None
        permute_1: "bf16[1000, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        le: "b8[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        unsqueeze_80: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
        unsqueeze_81: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_80, 2);  unsqueeze_80 = None
        unsqueeze_82: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_81, 3);  unsqueeze_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        unsqueeze_92: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
        unsqueeze_93: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_92, 2);  unsqueeze_92 = None
        unsqueeze_94: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_93, 3);  unsqueeze_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:100 in forward, code: identity = self.downsample(x)
        unsqueeze_104: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
        unsqueeze_105: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_104, 2);  unsqueeze_104 = None
        unsqueeze_106: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_105, 3);  unsqueeze_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        unsqueeze_116: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
        unsqueeze_117: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_116, 2);  unsqueeze_116 = None
        unsqueeze_118: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_117, 3);  unsqueeze_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        unsqueeze_128: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
        unsqueeze_129: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_128, 2);  unsqueeze_128 = None
        unsqueeze_130: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_129, 3);  unsqueeze_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        unsqueeze_140: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
        unsqueeze_141: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_140, 2);  unsqueeze_140 = None
        unsqueeze_142: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_141, 3);  unsqueeze_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        unsqueeze_152: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
        unsqueeze_153: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_152, 2);  unsqueeze_152 = None
        unsqueeze_154: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_153, 3);  unsqueeze_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:100 in forward, code: identity = self.downsample(x)
        unsqueeze_164: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
        unsqueeze_165: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_164, 2);  unsqueeze_164 = None
        unsqueeze_166: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_165, 3);  unsqueeze_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        unsqueeze_176: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
        unsqueeze_177: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_176, 2);  unsqueeze_176 = None
        unsqueeze_178: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_177, 3);  unsqueeze_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        unsqueeze_188: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
        unsqueeze_189: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_188, 2);  unsqueeze_188 = None
        unsqueeze_190: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_189, 3);  unsqueeze_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        unsqueeze_200: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
        unsqueeze_201: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_200, 2);  unsqueeze_200 = None
        unsqueeze_202: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_201, 3);  unsqueeze_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        unsqueeze_212: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
        unsqueeze_213: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 2);  unsqueeze_212 = None
        unsqueeze_214: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 3);  unsqueeze_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:100 in forward, code: identity = self.downsample(x)
        unsqueeze_224: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
        unsqueeze_225: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 2);  unsqueeze_224 = None
        unsqueeze_226: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_225, 3);  unsqueeze_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        unsqueeze_236: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
        unsqueeze_237: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 2);  unsqueeze_236 = None
        unsqueeze_238: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_237, 3);  unsqueeze_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        unsqueeze_248: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
        unsqueeze_249: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 2);  unsqueeze_248 = None
        unsqueeze_250: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_249, 3);  unsqueeze_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        unsqueeze_260: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
        unsqueeze_261: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 2);  unsqueeze_260 = None
        unsqueeze_262: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 3);  unsqueeze_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        unsqueeze_272: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
        unsqueeze_273: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 2);  unsqueeze_272 = None
        unsqueeze_274: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 3);  unsqueeze_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        unsqueeze_284: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
        unsqueeze_285: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 2);  unsqueeze_284 = None
        unsqueeze_286: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 3);  unsqueeze_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        unsqueeze_296: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
        unsqueeze_297: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 2);  unsqueeze_296 = None
        unsqueeze_298: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 3);  unsqueeze_297 = None

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
        copy__9: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_21, add_16);  primals_21 = add_16 = copy__9 = None
        copy__10: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_22, add_18);  primals_22 = add_18 = copy__10 = None
        copy__11: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_23, add_19);  primals_23 = add_19 = copy__11 = None
        copy__12: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_27, add_21);  primals_27 = add_21 = copy__12 = None
        copy__13: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_28, add_23);  primals_28 = add_23 = copy__13 = None
        copy__14: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_29, add_24);  primals_29 = add_24 = copy__14 = None
        copy__15: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_33, add_27);  primals_33 = add_27 = copy__15 = None
        copy__16: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_34, add_29);  primals_34 = add_29 = copy__16 = None
        copy__17: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_35, add_30);  primals_35 = add_30 = copy__17 = None
        copy__18: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_39, add_32);  primals_39 = add_32 = copy__18 = None
        copy__19: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_40, add_34);  primals_40 = add_34 = copy__19 = None
        copy__20: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_41, add_35);  primals_41 = add_35 = copy__20 = None
        copy__21: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_45, add_37);  primals_45 = add_37 = copy__21 = None
        copy__22: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_46, add_39);  primals_46 = add_39 = copy__22 = None
        copy__23: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_47, add_40);  primals_47 = add_40 = copy__23 = None
        copy__24: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_51, add_43);  primals_51 = add_43 = copy__24 = None
        copy__25: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_52, add_45);  primals_52 = add_45 = copy__25 = None
        copy__26: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_53, add_46);  primals_53 = add_46 = copy__26 = None
        copy__27: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_57, add_48);  primals_57 = add_48 = copy__27 = None
        copy__28: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_58, add_50);  primals_58 = add_50 = copy__28 = None
        copy__29: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_59, add_51);  primals_59 = add_51 = copy__29 = None
        copy__30: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_63, add_54);  primals_63 = add_54 = copy__30 = None
        copy__31: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_64, add_56);  primals_64 = add_56 = copy__31 = None
        copy__32: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_65, add_57);  primals_65 = add_57 = copy__32 = None
        copy__33: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_69, add_59);  primals_69 = add_59 = copy__33 = None
        copy__34: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_70, add_61);  primals_70 = add_61 = copy__34 = None
        copy__35: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_71, add_62);  primals_71 = add_62 = copy__35 = None
        copy__36: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_75, add_64);  primals_75 = add_64 = copy__36 = None
        copy__37: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_76, add_66);  primals_76 = add_66 = copy__37 = None
        copy__38: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_77, add_67);  primals_77 = add_67 = copy__38 = None
        copy__39: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_81, add_70);  primals_81 = add_70 = copy__39 = None
        copy__40: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_82, add_72);  primals_82 = add_72 = copy__40 = None
        copy__41: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_83, add_73);  primals_83 = add_73 = copy__41 = None
        copy__42: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_87, add_75);  primals_87 = add_75 = copy__42 = None
        copy__43: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_88, add_77);  primals_88 = add_77 = copy__43 = None
        copy__44: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_89, add_78);  primals_89 = add_78 = copy__44 = None
        copy__45: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_93, add_81);  primals_93 = add_81 = copy__45 = None
        copy__46: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_94, add_83);  primals_94 = add_83 = copy__46 = None
        copy__47: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_95, add_84);  primals_95 = add_84 = copy__47 = None
        copy__48: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_99, add_86);  primals_99 = add_86 = copy__48 = None
        copy__49: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_100, add_88);  primals_100 = add_88 = copy__49 = None
        copy__50: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_101, add_89);  primals_101 = add_89 = copy__50 = None
        copy__51: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_105, add_91);  primals_105 = add_91 = copy__51 = None
        copy__52: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_106, add_93);  primals_106 = add_93 = copy__52 = None
        copy__53: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_107, add_94);  primals_107 = add_94 = copy__53 = None
        copy__54: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_111, add_97);  primals_111 = add_97 = copy__54 = None
        copy__55: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_112, add_99);  primals_112 = add_99 = copy__55 = None
        copy__56: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_113, add_100);  primals_113 = add_100 = copy__56 = None
        copy__57: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_117, add_102);  primals_117 = add_102 = copy__57 = None
        copy__58: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_118, add_104);  primals_118 = add_104 = copy__58 = None
        copy__59: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_119, add_105);  primals_119 = add_105 = copy__59 = None
        return (addmm, primals_6, primals_7, primals_12, primals_18, primals_24, primals_30, primals_36, primals_42, primals_48, primals_54, primals_60, primals_66, primals_72, primals_78, primals_84, primals_90, primals_96, primals_102, primals_108, primals_114, primals_120, convert_element_type, convert_element_type_1, convolution, getitem_1, rsqrt, getitem_2, getitem_3, convert_element_type_4, convolution_1, squeeze_4, relu_1, convert_element_type_7, convolution_2, squeeze_7, relu_2, convert_element_type_10, convolution_3, squeeze_10, relu_3, convert_element_type_13, convolution_4, squeeze_13, relu_4, convert_element_type_16, convolution_5, squeeze_16, relu_5, convert_element_type_19, convolution_6, squeeze_19, convert_element_type_22, convolution_7, squeeze_22, relu_6, convert_element_type_25, convolution_8, squeeze_25, relu_7, convert_element_type_28, convolution_9, squeeze_28, relu_8, convert_element_type_31, convolution_10, squeeze_31, relu_9, convert_element_type_34, convolution_11, squeeze_34, convert_element_type_37, convolution_12, squeeze_37, relu_10, convert_element_type_40, convolution_13, squeeze_40, relu_11, convert_element_type_43, convolution_14, squeeze_43, relu_12, convert_element_type_46, convolution_15, squeeze_46, relu_13, convert_element_type_49, convolution_16, squeeze_49, convert_element_type_52, convolution_17, squeeze_52, relu_14, convert_element_type_55, convolution_18, squeeze_55, relu_15, convert_element_type_58, convolution_19, squeeze_58, view, permute_1, le, unsqueeze_82, unsqueeze_94, unsqueeze_106, unsqueeze_118, unsqueeze_130, unsqueeze_142, unsqueeze_154, unsqueeze_166, unsqueeze_178, unsqueeze_190, unsqueeze_202, unsqueeze_214, unsqueeze_226, unsqueeze_238, unsqueeze_250, unsqueeze_262, unsqueeze_274, unsqueeze_286, unsqueeze_298)
