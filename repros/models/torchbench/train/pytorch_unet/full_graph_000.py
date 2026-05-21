class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[64, 3, 3, 3]", primals_2: "f32[64]", primals_3: "f32[8, 3, 640, 959]", primals_4: "i64[]", primals_5: "f32[64]", primals_6: "f32[64]", primals_7: "f32[64]", primals_8: "f32[64]", primals_9: "f32[64, 64, 3, 3]", primals_10: "f32[64]", primals_11: "i64[]", primals_12: "f32[64]", primals_13: "f32[64]", primals_14: "f32[64]", primals_15: "f32[64]", primals_16: "f32[128, 64, 3, 3]", primals_17: "f32[128]", primals_18: "i64[]", primals_19: "f32[128]", primals_20: "f32[128]", primals_21: "f32[128]", primals_22: "f32[128]", primals_23: "f32[128, 128, 3, 3]", primals_24: "f32[128]", primals_25: "i64[]", primals_26: "f32[128]", primals_27: "f32[128]", primals_28: "f32[128]", primals_29: "f32[128]", primals_30: "f32[256, 128, 3, 3]", primals_31: "f32[256]", primals_32: "i64[]", primals_33: "f32[256]", primals_34: "f32[256]", primals_35: "f32[256]", primals_36: "f32[256]", primals_37: "f32[256, 256, 3, 3]", primals_38: "f32[256]", primals_39: "i64[]", primals_40: "f32[256]", primals_41: "f32[256]", primals_42: "f32[256]", primals_43: "f32[256]", primals_44: "f32[512, 256, 3, 3]", primals_45: "f32[512]", primals_46: "i64[]", primals_47: "f32[512]", primals_48: "f32[512]", primals_49: "f32[512]", primals_50: "f32[512]", primals_51: "f32[512, 512, 3, 3]", primals_52: "f32[512]", primals_53: "i64[]", primals_54: "f32[512]", primals_55: "f32[512]", primals_56: "f32[512]", primals_57: "f32[512]", primals_58: "f32[512, 512, 3, 3]", primals_59: "f32[512]", primals_60: "i64[]", primals_61: "f32[512]", primals_62: "f32[512]", primals_63: "f32[512]", primals_64: "f32[512]", primals_65: "f32[512, 512, 3, 3]", primals_66: "f32[512]", primals_67: "i64[]", primals_68: "f32[512]", primals_69: "f32[512]", primals_70: "f32[512]", primals_71: "f32[512]", primals_72: "f32[512, 1024, 3, 3]", primals_73: "f32[512]", primals_74: "i64[]", primals_75: "f32[512]", primals_76: "f32[512]", primals_77: "f32[512]", primals_78: "f32[512]", primals_79: "f32[256, 512, 3, 3]", primals_80: "f32[256]", primals_81: "i64[]", primals_82: "f32[256]", primals_83: "f32[256]", primals_84: "f32[256]", primals_85: "f32[256]", primals_86: "f32[256, 512, 3, 3]", primals_87: "f32[256]", primals_88: "i64[]", primals_89: "f32[256]", primals_90: "f32[256]", primals_91: "f32[256]", primals_92: "f32[256]", primals_93: "f32[128, 256, 3, 3]", primals_94: "f32[128]", primals_95: "i64[]", primals_96: "f32[128]", primals_97: "f32[128]", primals_98: "f32[128]", primals_99: "f32[128]", primals_100: "f32[128, 256, 3, 3]", primals_101: "f32[128]", primals_102: "i64[]", primals_103: "f32[128]", primals_104: "f32[128]", primals_105: "f32[128]", primals_106: "f32[128]", primals_107: "f32[64, 128, 3, 3]", primals_108: "f32[64]", primals_109: "i64[]", primals_110: "f32[64]", primals_111: "f32[64]", primals_112: "f32[64]", primals_113: "f32[64]", primals_114: "f32[64, 128, 3, 3]", primals_115: "f32[64]", primals_116: "i64[]", primals_117: "f32[64]", primals_118: "f32[64]", primals_119: "f32[64]", primals_120: "f32[64]", primals_121: "f32[64, 64, 3, 3]", primals_122: "f32[64]", primals_123: "i64[]", primals_124: "f32[64]", primals_125: "f32[64]", primals_126: "f32[64]", primals_127: "f32[64]", primals_128: "f32[2, 64, 1, 1]", primals_129: "f32[2]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25 in forward, code: return self.double_conv(x)
        convolution: "f32[8, 64, 640, 959]" = torch.ops.aten.convolution.default(primals_3, primals_1, primals_2, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_2 = None
        add: "i64[]" = torch.ops.aten.add.Tensor(primals_4, 1)
        var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 64, 1, 1]" = var_mean[0]
        getitem_1: "f32[1, 64, 1, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[8, 64, 640, 959]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_1: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_1: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
        mul_2: "f32[64]" = torch.ops.aten.mul.Tensor(primals_5, 0.9)
        add_2: "f32[64]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_2: "f32[64]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_3: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.000000203662711);  squeeze_2 = None
        mul_4: "f32[64]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
        mul_5: "f32[64]" = torch.ops.aten.mul.Tensor(primals_6, 0.9)
        add_3: "f32[64]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1)
        unsqueeze_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1);  primals_8 = None
        unsqueeze_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_4: "f32[8, 64, 640, 959]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        relu: "f32[8, 64, 640, 959]" = torch.ops.aten.relu.default(add_4);  add_4 = None
        convolution_1: "f32[8, 64, 640, 959]" = torch.ops.aten.convolution.default(relu, primals_9, primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_10 = None
        add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_11, 1)
        var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[1, 64, 1, 1]" = var_mean_1[0]
        getitem_3: "f32[1, 64, 1, 1]" = var_mean_1[1];  var_mean_1 = None
        add_6: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
        rsqrt_1: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_1: "f32[8, 64, 640, 959]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
        mul_7: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        squeeze_3: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3])
        mul_8: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1);  squeeze_3 = None
        mul_9: "f32[64]" = torch.ops.aten.mul.Tensor(primals_12, 0.9)
        add_7: "f32[64]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        squeeze_5: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_10: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.000000203662711);  squeeze_5 = None
        mul_11: "f32[64]" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
        mul_12: "f32[64]" = torch.ops.aten.mul.Tensor(primals_13, 0.9)
        add_8: "f32[64]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        unsqueeze_4: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1)
        unsqueeze_5: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_13: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
        unsqueeze_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1)
        unsqueeze_7: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_9: "f32[8, 64, 640, 959]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
        relu_1: "f32[8, 64, 640, 959]" = torch.ops.aten.relu.default(add_9);  add_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:38 in forward, code: return self.maxpool_conv(x)
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem_4: "f32[8, 64, 320, 479]" = _low_memory_max_pool_with_offsets[0]
        getitem_5: "i8[8, 64, 320, 479]" = _low_memory_max_pool_with_offsets[1];  _low_memory_max_pool_with_offsets = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25 in forward, code: return self.double_conv(x)
        convolution_2: "f32[8, 128, 320, 479]" = torch.ops.aten.convolution.default(getitem_4, primals_16, primals_17, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_17 = None
        add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_18, 1)
        var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
        getitem_6: "f32[1, 128, 1, 1]" = var_mean_2[0]
        getitem_7: "f32[1, 128, 1, 1]" = var_mean_2[1];  var_mean_2 = None
        add_11: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
        rsqrt_2: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_2: "f32[8, 128, 320, 479]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_7)
        mul_14: "f32[8, 128, 320, 479]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        squeeze_6: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        squeeze_7: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_15: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
        mul_16: "f32[128]" = torch.ops.aten.mul.Tensor(primals_19, 0.9)
        add_12: "f32[128]" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
        squeeze_8: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_17: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0000008155017088);  squeeze_8 = None
        mul_18: "f32[128]" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
        mul_19: "f32[128]" = torch.ops.aten.mul.Tensor(primals_20, 0.9)
        add_13: "f32[128]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
        unsqueeze_8: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1)
        unsqueeze_9: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        mul_20: "f32[8, 128, 320, 479]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
        unsqueeze_10: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_22, -1);  primals_22 = None
        unsqueeze_11: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        add_14: "f32[8, 128, 320, 479]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
        relu_2: "f32[8, 128, 320, 479]" = torch.ops.aten.relu.default(add_14);  add_14 = None
        convolution_3: "f32[8, 128, 320, 479]" = torch.ops.aten.convolution.default(relu_2, primals_23, primals_24, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_24 = None
        add_15: "i64[]" = torch.ops.aten.add.Tensor(primals_25, 1)
        var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
        getitem_8: "f32[1, 128, 1, 1]" = var_mean_3[0]
        getitem_9: "f32[1, 128, 1, 1]" = var_mean_3[1];  var_mean_3 = None
        add_16: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
        rsqrt_3: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_3: "f32[8, 128, 320, 479]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_9)
        mul_21: "f32[8, 128, 320, 479]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        squeeze_9: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3])
        mul_22: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1);  squeeze_9 = None
        mul_23: "f32[128]" = torch.ops.aten.mul.Tensor(primals_26, 0.9)
        add_17: "f32[128]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
        squeeze_11: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_24: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0000008155017088);  squeeze_11 = None
        mul_25: "f32[128]" = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
        mul_26: "f32[128]" = torch.ops.aten.mul.Tensor(primals_27, 0.9)
        add_18: "f32[128]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
        unsqueeze_12: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1)
        unsqueeze_13: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_27: "f32[8, 128, 320, 479]" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
        unsqueeze_14: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
        unsqueeze_15: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_19: "f32[8, 128, 320, 479]" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
        relu_3: "f32[8, 128, 320, 479]" = torch.ops.aten.relu.default(add_19);  add_19 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:38 in forward, code: return self.maxpool_conv(x)
        _low_memory_max_pool_with_offsets_1 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem_10: "f32[8, 128, 160, 239]" = _low_memory_max_pool_with_offsets_1[0]
        getitem_11: "i8[8, 128, 160, 239]" = _low_memory_max_pool_with_offsets_1[1];  _low_memory_max_pool_with_offsets_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25 in forward, code: return self.double_conv(x)
        convolution_4: "f32[8, 256, 160, 239]" = torch.ops.aten.convolution.default(getitem_10, primals_30, primals_31, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_31 = None
        add_20: "i64[]" = torch.ops.aten.add.Tensor(primals_32, 1)
        var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
        getitem_12: "f32[1, 256, 1, 1]" = var_mean_4[0]
        getitem_13: "f32[1, 256, 1, 1]" = var_mean_4[1];  var_mean_4 = None
        add_21: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
        rsqrt_4: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_4: "f32[8, 256, 160, 239]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_13)
        mul_28: "f32[8, 256, 160, 239]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        squeeze_12: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        squeeze_13: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_29: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
        mul_30: "f32[256]" = torch.ops.aten.mul.Tensor(primals_33, 0.9)
        add_22: "f32[256]" = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
        squeeze_14: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
        mul_31: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000032688391371);  squeeze_14 = None
        mul_32: "f32[256]" = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
        mul_33: "f32[256]" = torch.ops.aten.mul.Tensor(primals_34, 0.9)
        add_23: "f32[256]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
        unsqueeze_16: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
        unsqueeze_17: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        mul_34: "f32[8, 256, 160, 239]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
        unsqueeze_18: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
        unsqueeze_19: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        add_24: "f32[8, 256, 160, 239]" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None
        relu_4: "f32[8, 256, 160, 239]" = torch.ops.aten.relu.default(add_24);  add_24 = None
        convolution_5: "f32[8, 256, 160, 239]" = torch.ops.aten.convolution.default(relu_4, primals_37, primals_38, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_38 = None
        add_25: "i64[]" = torch.ops.aten.add.Tensor(primals_39, 1)
        var_mean_5 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
        getitem_14: "f32[1, 256, 1, 1]" = var_mean_5[0]
        getitem_15: "f32[1, 256, 1, 1]" = var_mean_5[1];  var_mean_5 = None
        add_26: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
        rsqrt_5: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_5: "f32[8, 256, 160, 239]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_15)
        mul_35: "f32[8, 256, 160, 239]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        squeeze_15: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3])
        mul_36: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1);  squeeze_15 = None
        mul_37: "f32[256]" = torch.ops.aten.mul.Tensor(primals_40, 0.9)
        add_27: "f32[256]" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
        squeeze_17: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
        mul_38: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0000032688391371);  squeeze_17 = None
        mul_39: "f32[256]" = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
        mul_40: "f32[256]" = torch.ops.aten.mul.Tensor(primals_41, 0.9)
        add_28: "f32[256]" = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
        unsqueeze_20: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1)
        unsqueeze_21: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_41: "f32[8, 256, 160, 239]" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
        unsqueeze_22: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1)
        unsqueeze_23: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_29: "f32[8, 256, 160, 239]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
        relu_5: "f32[8, 256, 160, 239]" = torch.ops.aten.relu.default(add_29);  add_29 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:38 in forward, code: return self.maxpool_conv(x)
        _low_memory_max_pool_with_offsets_2 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_5, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem_16: "f32[8, 256, 80, 119]" = _low_memory_max_pool_with_offsets_2[0]
        getitem_17: "i8[8, 256, 80, 119]" = _low_memory_max_pool_with_offsets_2[1];  _low_memory_max_pool_with_offsets_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25 in forward, code: return self.double_conv(x)
        convolution_6: "f32[8, 512, 80, 119]" = torch.ops.aten.convolution.default(getitem_16, primals_44, primals_45, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_45 = None
        add_30: "i64[]" = torch.ops.aten.add.Tensor(primals_46, 1)
        var_mean_6 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
        getitem_18: "f32[1, 512, 1, 1]" = var_mean_6[0]
        getitem_19: "f32[1, 512, 1, 1]" = var_mean_6[1];  var_mean_6 = None
        add_31: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
        rsqrt_6: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        sub_6: "f32[8, 512, 80, 119]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_19)
        mul_42: "f32[8, 512, 80, 119]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        squeeze_18: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        squeeze_19: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_43: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
        mul_44: "f32[512]" = torch.ops.aten.mul.Tensor(primals_47, 0.9)
        add_32: "f32[512]" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
        squeeze_20: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
        mul_45: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0000131304245066);  squeeze_20 = None
        mul_46: "f32[512]" = torch.ops.aten.mul.Tensor(mul_45, 0.1);  mul_45 = None
        mul_47: "f32[512]" = torch.ops.aten.mul.Tensor(primals_48, 0.9)
        add_33: "f32[512]" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
        unsqueeze_24: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1)
        unsqueeze_25: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        mul_48: "f32[8, 512, 80, 119]" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_25);  mul_42 = unsqueeze_25 = None
        unsqueeze_26: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1);  primals_50 = None
        unsqueeze_27: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        add_34: "f32[8, 512, 80, 119]" = torch.ops.aten.add.Tensor(mul_48, unsqueeze_27);  mul_48 = unsqueeze_27 = None
        relu_6: "f32[8, 512, 80, 119]" = torch.ops.aten.relu.default(add_34);  add_34 = None
        convolution_7: "f32[8, 512, 80, 119]" = torch.ops.aten.convolution.default(relu_6, primals_51, primals_52, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_52 = None
        add_35: "i64[]" = torch.ops.aten.add.Tensor(primals_53, 1)
        var_mean_7 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
        getitem_20: "f32[1, 512, 1, 1]" = var_mean_7[0]
        getitem_21: "f32[1, 512, 1, 1]" = var_mean_7[1];  var_mean_7 = None
        add_36: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
        rsqrt_7: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_7: "f32[8, 512, 80, 119]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_21)
        mul_49: "f32[8, 512, 80, 119]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        squeeze_21: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3])
        mul_50: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1);  squeeze_21 = None
        mul_51: "f32[512]" = torch.ops.aten.mul.Tensor(primals_54, 0.9)
        add_37: "f32[512]" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
        squeeze_23: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        mul_52: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0000131304245066);  squeeze_23 = None
        mul_53: "f32[512]" = torch.ops.aten.mul.Tensor(mul_52, 0.1);  mul_52 = None
        mul_54: "f32[512]" = torch.ops.aten.mul.Tensor(primals_55, 0.9)
        add_38: "f32[512]" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
        unsqueeze_28: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1)
        unsqueeze_29: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_55: "f32[8, 512, 80, 119]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
        unsqueeze_30: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1)
        unsqueeze_31: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_39: "f32[8, 512, 80, 119]" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None
        relu_7: "f32[8, 512, 80, 119]" = torch.ops.aten.relu.default(add_39);  add_39 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:38 in forward, code: return self.maxpool_conv(x)
        _low_memory_max_pool_with_offsets_3 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_7, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem_22: "f32[8, 512, 40, 59]" = _low_memory_max_pool_with_offsets_3[0]
        getitem_23: "i8[8, 512, 40, 59]" = _low_memory_max_pool_with_offsets_3[1];  _low_memory_max_pool_with_offsets_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25 in forward, code: return self.double_conv(x)
        convolution_8: "f32[8, 512, 40, 59]" = torch.ops.aten.convolution.default(getitem_22, primals_58, primals_59, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_59 = None
        add_40: "i64[]" = torch.ops.aten.add.Tensor(primals_60, 1)
        var_mean_8 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
        getitem_24: "f32[1, 512, 1, 1]" = var_mean_8[0]
        getitem_25: "f32[1, 512, 1, 1]" = var_mean_8[1];  var_mean_8 = None
        add_41: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
        rsqrt_8: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
        sub_8: "f32[8, 512, 40, 59]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_25)
        mul_56: "f32[8, 512, 40, 59]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        squeeze_24: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        squeeze_25: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_57: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
        mul_58: "f32[512]" = torch.ops.aten.mul.Tensor(primals_61, 0.9)
        add_42: "f32[512]" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
        squeeze_26: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
        mul_59: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0000529689072515);  squeeze_26 = None
        mul_60: "f32[512]" = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
        mul_61: "f32[512]" = torch.ops.aten.mul.Tensor(primals_62, 0.9)
        add_43: "f32[512]" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
        unsqueeze_32: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1)
        unsqueeze_33: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        mul_62: "f32[8, 512, 40, 59]" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
        unsqueeze_34: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_64, -1);  primals_64 = None
        unsqueeze_35: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        add_44: "f32[8, 512, 40, 59]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None
        relu_8: "f32[8, 512, 40, 59]" = torch.ops.aten.relu.default(add_44);  add_44 = None
        convolution_9: "f32[8, 512, 40, 59]" = torch.ops.aten.convolution.default(relu_8, primals_65, primals_66, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_66 = None
        add_45: "i64[]" = torch.ops.aten.add.Tensor(primals_67, 1)
        var_mean_9 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
        getitem_26: "f32[1, 512, 1, 1]" = var_mean_9[0]
        getitem_27: "f32[1, 512, 1, 1]" = var_mean_9[1];  var_mean_9 = None
        add_46: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
        rsqrt_9: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_9: "f32[8, 512, 40, 59]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_27)
        mul_63: "f32[8, 512, 40, 59]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        squeeze_27: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3])
        mul_64: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1);  squeeze_27 = None
        mul_65: "f32[512]" = torch.ops.aten.mul.Tensor(primals_68, 0.9)
        add_47: "f32[512]" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
        squeeze_29: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        mul_66: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0000529689072515);  squeeze_29 = None
        mul_67: "f32[512]" = torch.ops.aten.mul.Tensor(mul_66, 0.1);  mul_66 = None
        mul_68: "f32[512]" = torch.ops.aten.mul.Tensor(primals_69, 0.9)
        add_48: "f32[512]" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
        unsqueeze_36: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_70, -1)
        unsqueeze_37: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_69: "f32[8, 512, 40, 59]" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_37);  mul_63 = unsqueeze_37 = None
        unsqueeze_38: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
        unsqueeze_39: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_49: "f32[8, 512, 40, 59]" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_39);  mul_69 = unsqueeze_39 = None
        relu_9: "f32[8, 512, 40, 59]" = torch.ops.aten.relu.default(add_49);  add_49 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:58 in forward, code: x1 = self.up(x1)
        iota: "i64[80]" = torch.ops.prims.iota.default(80, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type: "f32[80]" = torch.ops.prims.convert_element_type.default(iota, torch.float32);  iota = None
        mul_70: "f32[80]" = torch.ops.aten.mul.Tensor(convert_element_type, 0.4936708860759494);  convert_element_type = None
        clamp_min: "f32[80]" = torch.ops.aten.clamp_min.default(mul_70, 0.0);  mul_70 = None
        view: "f32[80, 1]" = torch.ops.aten.reshape.default(clamp_min, [80, 1]);  clamp_min = None
        convert_element_type_1: "i64[80, 1]" = torch.ops.prims.convert_element_type.default(view, torch.int64)
        add_50: "i64[80, 1]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1)
        clamp_max: "i64[80, 1]" = torch.ops.aten.clamp_max.default(add_50, 39);  add_50 = None
        iota_1: "i64[118]" = torch.ops.prims.iota.default(118, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_2: "f32[118]" = torch.ops.prims.convert_element_type.default(iota_1, torch.float32);  iota_1 = None
        mul_71: "f32[118]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 0.49572649572649574);  convert_element_type_2 = None
        clamp_min_1: "f32[118]" = torch.ops.aten.clamp_min.default(mul_71, 0.0);  mul_71 = None
        convert_element_type_3: "i64[118]" = torch.ops.prims.convert_element_type.default(clamp_min_1, torch.int64)
        add_51: "i64[118]" = torch.ops.aten.add.Tensor(convert_element_type_3, 1)
        clamp_max_1: "i64[118]" = torch.ops.aten.clamp_max.default(add_51, 58);  add_51 = None
        _unsafe_index: "f32[8, 512, 80, 118]" = torch.ops.aten._unsafe_index.Tensor(relu_9, [None, None, convert_element_type_1, convert_element_type_3])
        _unsafe_index_1: "f32[8, 512, 80, 118]" = torch.ops.aten._unsafe_index.Tensor(relu_9, [None, None, convert_element_type_1, clamp_max_1])
        _unsafe_index_2: "f32[8, 512, 80, 118]" = torch.ops.aten._unsafe_index.Tensor(relu_9, [None, None, clamp_max, convert_element_type_3])
        _unsafe_index_3: "f32[8, 512, 80, 118]" = torch.ops.aten._unsafe_index.Tensor(relu_9, [None, None, clamp_max, clamp_max_1]);  relu_9 = None
        sub_10: "f32[118]" = torch.ops.aten.sub.Tensor(clamp_min_1, convert_element_type_3);  clamp_min_1 = None
        clamp_min_2: "f32[118]" = torch.ops.aten.clamp_min.default(sub_10, 0.0);  sub_10 = None
        clamp_max_2: "f32[118]" = torch.ops.aten.clamp_max.default(clamp_min_2, 1.0);  clamp_min_2 = None
        sub_11: "f32[8, 512, 80, 118]" = torch.ops.aten.sub.Tensor(_unsafe_index_1, _unsafe_index);  _unsafe_index_1 = None
        mul_72: "f32[8, 512, 80, 118]" = torch.ops.aten.mul.Tensor(sub_11, clamp_max_2);  sub_11 = None
        add_52: "f32[8, 512, 80, 118]" = torch.ops.aten.add.Tensor(_unsafe_index, mul_72);  _unsafe_index = mul_72 = None
        sub_12: "f32[8, 512, 80, 118]" = torch.ops.aten.sub.Tensor(_unsafe_index_3, _unsafe_index_2);  _unsafe_index_3 = None
        mul_73: "f32[8, 512, 80, 118]" = torch.ops.aten.mul.Tensor(sub_12, clamp_max_2);  sub_12 = None
        add_53: "f32[8, 512, 80, 118]" = torch.ops.aten.add.Tensor(_unsafe_index_2, mul_73);  _unsafe_index_2 = mul_73 = None
        sub_13: "f32[80, 1]" = torch.ops.aten.sub.Tensor(view, convert_element_type_1);  view = None
        clamp_min_3: "f32[80, 1]" = torch.ops.aten.clamp_min.default(sub_13, 0.0);  sub_13 = None
        clamp_max_3: "f32[80, 1]" = torch.ops.aten.clamp_max.default(clamp_min_3, 1.0);  clamp_min_3 = None
        sub_14: "f32[8, 512, 80, 118]" = torch.ops.aten.sub.Tensor(add_53, add_52);  add_53 = None
        mul_74: "f32[8, 512, 80, 118]" = torch.ops.aten.mul.Tensor(sub_14, clamp_max_3);  sub_14 = None
        add_54: "f32[8, 512, 80, 118]" = torch.ops.aten.add.Tensor(add_52, mul_74);  add_52 = mul_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "f32[8, 512, 80, 119]" = torch.ops.aten.constant_pad_nd.default(add_54, [0, 1, 0, 0], 0.0);  add_54 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:67 in forward, code: x = torch.cat([x2, x1], dim=1)
        cat: "f32[8, 1024, 80, 119]" = torch.ops.aten.cat.default([relu_7, constant_pad_nd], 1);  relu_7 = constant_pad_nd = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25 in forward, code: return self.double_conv(x)
        convolution_10: "f32[8, 512, 80, 119]" = torch.ops.aten.convolution.default(cat, primals_72, primals_73, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_73 = None
        add_55: "i64[]" = torch.ops.aten.add.Tensor(primals_74, 1)
        var_mean_10 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
        getitem_28: "f32[1, 512, 1, 1]" = var_mean_10[0]
        getitem_29: "f32[1, 512, 1, 1]" = var_mean_10[1];  var_mean_10 = None
        add_56: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
        rsqrt_10: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        sub_15: "f32[8, 512, 80, 119]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_29)
        mul_75: "f32[8, 512, 80, 119]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = None
        squeeze_30: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        squeeze_31: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_76: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
        mul_77: "f32[512]" = torch.ops.aten.mul.Tensor(primals_75, 0.9)
        add_57: "f32[512]" = torch.ops.aten.add.Tensor(mul_76, mul_77);  mul_76 = mul_77 = None
        squeeze_32: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        mul_78: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0000131304245066);  squeeze_32 = None
        mul_79: "f32[512]" = torch.ops.aten.mul.Tensor(mul_78, 0.1);  mul_78 = None
        mul_80: "f32[512]" = torch.ops.aten.mul.Tensor(primals_76, 0.9)
        add_58: "f32[512]" = torch.ops.aten.add.Tensor(mul_79, mul_80);  mul_79 = mul_80 = None
        unsqueeze_40: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1)
        unsqueeze_41: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        mul_81: "f32[8, 512, 80, 119]" = torch.ops.aten.mul.Tensor(mul_75, unsqueeze_41);  mul_75 = unsqueeze_41 = None
        unsqueeze_42: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
        unsqueeze_43: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        add_59: "f32[8, 512, 80, 119]" = torch.ops.aten.add.Tensor(mul_81, unsqueeze_43);  mul_81 = unsqueeze_43 = None
        relu_10: "f32[8, 512, 80, 119]" = torch.ops.aten.relu.default(add_59);  add_59 = None
        convolution_11: "f32[8, 256, 80, 119]" = torch.ops.aten.convolution.default(relu_10, primals_79, primals_80, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_80 = None
        add_60: "i64[]" = torch.ops.aten.add.Tensor(primals_81, 1)
        var_mean_11 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
        getitem_30: "f32[1, 256, 1, 1]" = var_mean_11[0]
        getitem_31: "f32[1, 256, 1, 1]" = var_mean_11[1];  var_mean_11 = None
        add_61: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
        rsqrt_11: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        sub_16: "f32[8, 256, 80, 119]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_31)
        mul_82: "f32[8, 256, 80, 119]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_11);  sub_16 = None
        squeeze_33: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3])
        mul_83: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1);  squeeze_33 = None
        mul_84: "f32[256]" = torch.ops.aten.mul.Tensor(primals_82, 0.9)
        add_62: "f32[256]" = torch.ops.aten.add.Tensor(mul_83, mul_84);  mul_83 = mul_84 = None
        squeeze_35: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
        mul_85: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0000131304245066);  squeeze_35 = None
        mul_86: "f32[256]" = torch.ops.aten.mul.Tensor(mul_85, 0.1);  mul_85 = None
        mul_87: "f32[256]" = torch.ops.aten.mul.Tensor(primals_83, 0.9)
        add_63: "f32[256]" = torch.ops.aten.add.Tensor(mul_86, mul_87);  mul_86 = mul_87 = None
        unsqueeze_44: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1)
        unsqueeze_45: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_88: "f32[8, 256, 80, 119]" = torch.ops.aten.mul.Tensor(mul_82, unsqueeze_45);  mul_82 = unsqueeze_45 = None
        unsqueeze_46: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_85, -1)
        unsqueeze_47: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_64: "f32[8, 256, 80, 119]" = torch.ops.aten.add.Tensor(mul_88, unsqueeze_47);  mul_88 = unsqueeze_47 = None
        relu_11: "f32[8, 256, 80, 119]" = torch.ops.aten.relu.default(add_64);  add_64 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:58 in forward, code: x1 = self.up(x1)
        iota_2: "i64[160]" = torch.ops.prims.iota.default(160, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_4: "f32[160]" = torch.ops.prims.convert_element_type.default(iota_2, torch.float32);  iota_2 = None
        mul_89: "f32[160]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 0.4968553459119497);  convert_element_type_4 = None
        clamp_min_4: "f32[160]" = torch.ops.aten.clamp_min.default(mul_89, 0.0);  mul_89 = None
        view_2: "f32[160, 1]" = torch.ops.aten.reshape.default(clamp_min_4, [160, 1]);  clamp_min_4 = None
        convert_element_type_5: "i64[160, 1]" = torch.ops.prims.convert_element_type.default(view_2, torch.int64)
        add_65: "i64[160, 1]" = torch.ops.aten.add.Tensor(convert_element_type_5, 1)
        clamp_max_4: "i64[160, 1]" = torch.ops.aten.clamp_max.default(add_65, 79);  add_65 = None
        iota_3: "i64[238]" = torch.ops.prims.iota.default(238, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_6: "f32[238]" = torch.ops.prims.convert_element_type.default(iota_3, torch.float32);  iota_3 = None
        mul_90: "f32[238]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.4978902953586498);  convert_element_type_6 = None
        clamp_min_5: "f32[238]" = torch.ops.aten.clamp_min.default(mul_90, 0.0);  mul_90 = None
        convert_element_type_7: "i64[238]" = torch.ops.prims.convert_element_type.default(clamp_min_5, torch.int64)
        add_66: "i64[238]" = torch.ops.aten.add.Tensor(convert_element_type_7, 1)
        clamp_max_5: "i64[238]" = torch.ops.aten.clamp_max.default(add_66, 118);  add_66 = None
        _unsafe_index_4: "f32[8, 256, 160, 238]" = torch.ops.aten._unsafe_index.Tensor(relu_11, [None, None, convert_element_type_5, convert_element_type_7])
        _unsafe_index_5: "f32[8, 256, 160, 238]" = torch.ops.aten._unsafe_index.Tensor(relu_11, [None, None, convert_element_type_5, clamp_max_5])
        _unsafe_index_6: "f32[8, 256, 160, 238]" = torch.ops.aten._unsafe_index.Tensor(relu_11, [None, None, clamp_max_4, convert_element_type_7])
        _unsafe_index_7: "f32[8, 256, 160, 238]" = torch.ops.aten._unsafe_index.Tensor(relu_11, [None, None, clamp_max_4, clamp_max_5]);  relu_11 = None
        sub_17: "f32[238]" = torch.ops.aten.sub.Tensor(clamp_min_5, convert_element_type_7);  clamp_min_5 = None
        clamp_min_6: "f32[238]" = torch.ops.aten.clamp_min.default(sub_17, 0.0);  sub_17 = None
        clamp_max_6: "f32[238]" = torch.ops.aten.clamp_max.default(clamp_min_6, 1.0);  clamp_min_6 = None
        sub_18: "f32[8, 256, 160, 238]" = torch.ops.aten.sub.Tensor(_unsafe_index_5, _unsafe_index_4);  _unsafe_index_5 = None
        mul_91: "f32[8, 256, 160, 238]" = torch.ops.aten.mul.Tensor(sub_18, clamp_max_6);  sub_18 = None
        add_67: "f32[8, 256, 160, 238]" = torch.ops.aten.add.Tensor(_unsafe_index_4, mul_91);  _unsafe_index_4 = mul_91 = None
        sub_19: "f32[8, 256, 160, 238]" = torch.ops.aten.sub.Tensor(_unsafe_index_7, _unsafe_index_6);  _unsafe_index_7 = None
        mul_92: "f32[8, 256, 160, 238]" = torch.ops.aten.mul.Tensor(sub_19, clamp_max_6);  sub_19 = None
        add_68: "f32[8, 256, 160, 238]" = torch.ops.aten.add.Tensor(_unsafe_index_6, mul_92);  _unsafe_index_6 = mul_92 = None
        sub_20: "f32[160, 1]" = torch.ops.aten.sub.Tensor(view_2, convert_element_type_5);  view_2 = None
        clamp_min_7: "f32[160, 1]" = torch.ops.aten.clamp_min.default(sub_20, 0.0);  sub_20 = None
        clamp_max_7: "f32[160, 1]" = torch.ops.aten.clamp_max.default(clamp_min_7, 1.0);  clamp_min_7 = None
        sub_21: "f32[8, 256, 160, 238]" = torch.ops.aten.sub.Tensor(add_68, add_67);  add_68 = None
        mul_93: "f32[8, 256, 160, 238]" = torch.ops.aten.mul.Tensor(sub_21, clamp_max_7);  sub_21 = None
        add_69: "f32[8, 256, 160, 238]" = torch.ops.aten.add.Tensor(add_67, mul_93);  add_67 = mul_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_1: "f32[8, 256, 160, 239]" = torch.ops.aten.constant_pad_nd.default(add_69, [0, 1, 0, 0], 0.0);  add_69 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:67 in forward, code: x = torch.cat([x2, x1], dim=1)
        cat_1: "f32[8, 512, 160, 239]" = torch.ops.aten.cat.default([relu_5, constant_pad_nd_1], 1);  relu_5 = constant_pad_nd_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25 in forward, code: return self.double_conv(x)
        convolution_12: "f32[8, 256, 160, 239]" = torch.ops.aten.convolution.default(cat_1, primals_86, primals_87, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_87 = None
        add_70: "i64[]" = torch.ops.aten.add.Tensor(primals_88, 1)
        var_mean_12 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
        getitem_32: "f32[1, 256, 1, 1]" = var_mean_12[0]
        getitem_33: "f32[1, 256, 1, 1]" = var_mean_12[1];  var_mean_12 = None
        add_71: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
        rsqrt_12: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        sub_22: "f32[8, 256, 160, 239]" = torch.ops.aten.sub.Tensor(convolution_12, getitem_33)
        mul_94: "f32[8, 256, 160, 239]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_12);  sub_22 = None
        squeeze_36: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        squeeze_37: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_95: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
        mul_96: "f32[256]" = torch.ops.aten.mul.Tensor(primals_89, 0.9)
        add_72: "f32[256]" = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None
        squeeze_38: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
        mul_97: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0000032688391371);  squeeze_38 = None
        mul_98: "f32[256]" = torch.ops.aten.mul.Tensor(mul_97, 0.1);  mul_97 = None
        mul_99: "f32[256]" = torch.ops.aten.mul.Tensor(primals_90, 0.9)
        add_73: "f32[256]" = torch.ops.aten.add.Tensor(mul_98, mul_99);  mul_98 = mul_99 = None
        unsqueeze_48: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_91, -1)
        unsqueeze_49: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        mul_100: "f32[8, 256, 160, 239]" = torch.ops.aten.mul.Tensor(mul_94, unsqueeze_49);  mul_94 = unsqueeze_49 = None
        unsqueeze_50: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1);  primals_92 = None
        unsqueeze_51: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        add_74: "f32[8, 256, 160, 239]" = torch.ops.aten.add.Tensor(mul_100, unsqueeze_51);  mul_100 = unsqueeze_51 = None
        relu_12: "f32[8, 256, 160, 239]" = torch.ops.aten.relu.default(add_74);  add_74 = None
        convolution_13: "f32[8, 128, 160, 239]" = torch.ops.aten.convolution.default(relu_12, primals_93, primals_94, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_94 = None
        add_75: "i64[]" = torch.ops.aten.add.Tensor(primals_95, 1)
        var_mean_13 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
        getitem_34: "f32[1, 128, 1, 1]" = var_mean_13[0]
        getitem_35: "f32[1, 128, 1, 1]" = var_mean_13[1];  var_mean_13 = None
        add_76: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
        rsqrt_13: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        sub_23: "f32[8, 128, 160, 239]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_35)
        mul_101: "f32[8, 128, 160, 239]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_13);  sub_23 = None
        squeeze_39: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3])
        mul_102: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1);  squeeze_39 = None
        mul_103: "f32[128]" = torch.ops.aten.mul.Tensor(primals_96, 0.9)
        add_77: "f32[128]" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
        squeeze_41: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
        mul_104: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0000032688391371);  squeeze_41 = None
        mul_105: "f32[128]" = torch.ops.aten.mul.Tensor(mul_104, 0.1);  mul_104 = None
        mul_106: "f32[128]" = torch.ops.aten.mul.Tensor(primals_97, 0.9)
        add_78: "f32[128]" = torch.ops.aten.add.Tensor(mul_105, mul_106);  mul_105 = mul_106 = None
        unsqueeze_52: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1)
        unsqueeze_53: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_107: "f32[8, 128, 160, 239]" = torch.ops.aten.mul.Tensor(mul_101, unsqueeze_53);  mul_101 = unsqueeze_53 = None
        unsqueeze_54: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1)
        unsqueeze_55: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_79: "f32[8, 128, 160, 239]" = torch.ops.aten.add.Tensor(mul_107, unsqueeze_55);  mul_107 = unsqueeze_55 = None
        relu_13: "f32[8, 128, 160, 239]" = torch.ops.aten.relu.default(add_79);  add_79 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:58 in forward, code: x1 = self.up(x1)
        iota_4: "i64[320]" = torch.ops.prims.iota.default(320, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_8: "f32[320]" = torch.ops.prims.convert_element_type.default(iota_4, torch.float32);  iota_4 = None
        mul_108: "f32[320]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 0.49843260188087773);  convert_element_type_8 = None
        clamp_min_8: "f32[320]" = torch.ops.aten.clamp_min.default(mul_108, 0.0);  mul_108 = None
        view_4: "f32[320, 1]" = torch.ops.aten.reshape.default(clamp_min_8, [320, 1]);  clamp_min_8 = None
        convert_element_type_9: "i64[320, 1]" = torch.ops.prims.convert_element_type.default(view_4, torch.int64)
        add_80: "i64[320, 1]" = torch.ops.aten.add.Tensor(convert_element_type_9, 1)
        clamp_max_8: "i64[320, 1]" = torch.ops.aten.clamp_max.default(add_80, 159);  add_80 = None
        iota_5: "i64[478]" = torch.ops.prims.iota.default(478, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_10: "f32[478]" = torch.ops.prims.convert_element_type.default(iota_5, torch.float32);  iota_5 = None
        mul_109: "f32[478]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 0.4989517819706499);  convert_element_type_10 = None
        clamp_min_9: "f32[478]" = torch.ops.aten.clamp_min.default(mul_109, 0.0);  mul_109 = None
        convert_element_type_11: "i64[478]" = torch.ops.prims.convert_element_type.default(clamp_min_9, torch.int64)
        add_81: "i64[478]" = torch.ops.aten.add.Tensor(convert_element_type_11, 1)
        clamp_max_9: "i64[478]" = torch.ops.aten.clamp_max.default(add_81, 238);  add_81 = None
        _unsafe_index_8: "f32[8, 128, 320, 478]" = torch.ops.aten._unsafe_index.Tensor(relu_13, [None, None, convert_element_type_9, convert_element_type_11])
        _unsafe_index_9: "f32[8, 128, 320, 478]" = torch.ops.aten._unsafe_index.Tensor(relu_13, [None, None, convert_element_type_9, clamp_max_9])
        _unsafe_index_10: "f32[8, 128, 320, 478]" = torch.ops.aten._unsafe_index.Tensor(relu_13, [None, None, clamp_max_8, convert_element_type_11])
        _unsafe_index_11: "f32[8, 128, 320, 478]" = torch.ops.aten._unsafe_index.Tensor(relu_13, [None, None, clamp_max_8, clamp_max_9]);  relu_13 = None
        sub_24: "f32[478]" = torch.ops.aten.sub.Tensor(clamp_min_9, convert_element_type_11);  clamp_min_9 = None
        clamp_min_10: "f32[478]" = torch.ops.aten.clamp_min.default(sub_24, 0.0);  sub_24 = None
        clamp_max_10: "f32[478]" = torch.ops.aten.clamp_max.default(clamp_min_10, 1.0);  clamp_min_10 = None
        sub_25: "f32[8, 128, 320, 478]" = torch.ops.aten.sub.Tensor(_unsafe_index_9, _unsafe_index_8);  _unsafe_index_9 = None
        mul_110: "f32[8, 128, 320, 478]" = torch.ops.aten.mul.Tensor(sub_25, clamp_max_10);  sub_25 = None
        add_82: "f32[8, 128, 320, 478]" = torch.ops.aten.add.Tensor(_unsafe_index_8, mul_110);  _unsafe_index_8 = mul_110 = None
        sub_26: "f32[8, 128, 320, 478]" = torch.ops.aten.sub.Tensor(_unsafe_index_11, _unsafe_index_10);  _unsafe_index_11 = None
        mul_111: "f32[8, 128, 320, 478]" = torch.ops.aten.mul.Tensor(sub_26, clamp_max_10);  sub_26 = None
        add_83: "f32[8, 128, 320, 478]" = torch.ops.aten.add.Tensor(_unsafe_index_10, mul_111);  _unsafe_index_10 = mul_111 = None
        sub_27: "f32[320, 1]" = torch.ops.aten.sub.Tensor(view_4, convert_element_type_9);  view_4 = None
        clamp_min_11: "f32[320, 1]" = torch.ops.aten.clamp_min.default(sub_27, 0.0);  sub_27 = None
        clamp_max_11: "f32[320, 1]" = torch.ops.aten.clamp_max.default(clamp_min_11, 1.0);  clamp_min_11 = None
        sub_28: "f32[8, 128, 320, 478]" = torch.ops.aten.sub.Tensor(add_83, add_82);  add_83 = None
        mul_112: "f32[8, 128, 320, 478]" = torch.ops.aten.mul.Tensor(sub_28, clamp_max_11);  sub_28 = None
        add_84: "f32[8, 128, 320, 478]" = torch.ops.aten.add.Tensor(add_82, mul_112);  add_82 = mul_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_2: "f32[8, 128, 320, 479]" = torch.ops.aten.constant_pad_nd.default(add_84, [0, 1, 0, 0], 0.0);  add_84 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:67 in forward, code: x = torch.cat([x2, x1], dim=1)
        cat_2: "f32[8, 256, 320, 479]" = torch.ops.aten.cat.default([relu_3, constant_pad_nd_2], 1);  relu_3 = constant_pad_nd_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25 in forward, code: return self.double_conv(x)
        convolution_14: "f32[8, 128, 320, 479]" = torch.ops.aten.convolution.default(cat_2, primals_100, primals_101, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_101 = None
        add_85: "i64[]" = torch.ops.aten.add.Tensor(primals_102, 1)
        var_mean_14 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
        getitem_36: "f32[1, 128, 1, 1]" = var_mean_14[0]
        getitem_37: "f32[1, 128, 1, 1]" = var_mean_14[1];  var_mean_14 = None
        add_86: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
        rsqrt_14: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        sub_29: "f32[8, 128, 320, 479]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_37)
        mul_113: "f32[8, 128, 320, 479]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_14);  sub_29 = None
        squeeze_42: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
        squeeze_43: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_114: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
        mul_115: "f32[128]" = torch.ops.aten.mul.Tensor(primals_103, 0.9)
        add_87: "f32[128]" = torch.ops.aten.add.Tensor(mul_114, mul_115);  mul_114 = mul_115 = None
        squeeze_44: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
        mul_116: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0000008155017088);  squeeze_44 = None
        mul_117: "f32[128]" = torch.ops.aten.mul.Tensor(mul_116, 0.1);  mul_116 = None
        mul_118: "f32[128]" = torch.ops.aten.mul.Tensor(primals_104, 0.9)
        add_88: "f32[128]" = torch.ops.aten.add.Tensor(mul_117, mul_118);  mul_117 = mul_118 = None
        unsqueeze_56: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_105, -1)
        unsqueeze_57: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        mul_119: "f32[8, 128, 320, 479]" = torch.ops.aten.mul.Tensor(mul_113, unsqueeze_57);  mul_113 = unsqueeze_57 = None
        unsqueeze_58: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_106, -1);  primals_106 = None
        unsqueeze_59: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        add_89: "f32[8, 128, 320, 479]" = torch.ops.aten.add.Tensor(mul_119, unsqueeze_59);  mul_119 = unsqueeze_59 = None
        relu_14: "f32[8, 128, 320, 479]" = torch.ops.aten.relu.default(add_89);  add_89 = None
        convolution_15: "f32[8, 64, 320, 479]" = torch.ops.aten.convolution.default(relu_14, primals_107, primals_108, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_108 = None
        add_90: "i64[]" = torch.ops.aten.add.Tensor(primals_109, 1)
        var_mean_15 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
        getitem_38: "f32[1, 64, 1, 1]" = var_mean_15[0]
        getitem_39: "f32[1, 64, 1, 1]" = var_mean_15[1];  var_mean_15 = None
        add_91: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05)
        rsqrt_15: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
        sub_30: "f32[8, 64, 320, 479]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_39)
        mul_120: "f32[8, 64, 320, 479]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_15);  sub_30 = None
        squeeze_45: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3])
        mul_121: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1);  squeeze_45 = None
        mul_122: "f32[64]" = torch.ops.aten.mul.Tensor(primals_110, 0.9)
        add_92: "f32[64]" = torch.ops.aten.add.Tensor(mul_121, mul_122);  mul_121 = mul_122 = None
        squeeze_47: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
        mul_123: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0000008155017088);  squeeze_47 = None
        mul_124: "f32[64]" = torch.ops.aten.mul.Tensor(mul_123, 0.1);  mul_123 = None
        mul_125: "f32[64]" = torch.ops.aten.mul.Tensor(primals_111, 0.9)
        add_93: "f32[64]" = torch.ops.aten.add.Tensor(mul_124, mul_125);  mul_124 = mul_125 = None
        unsqueeze_60: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_112, -1)
        unsqueeze_61: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_126: "f32[8, 64, 320, 479]" = torch.ops.aten.mul.Tensor(mul_120, unsqueeze_61);  mul_120 = unsqueeze_61 = None
        unsqueeze_62: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_113, -1)
        unsqueeze_63: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_94: "f32[8, 64, 320, 479]" = torch.ops.aten.add.Tensor(mul_126, unsqueeze_63);  mul_126 = unsqueeze_63 = None
        relu_15: "f32[8, 64, 320, 479]" = torch.ops.aten.relu.default(add_94);  add_94 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:58 in forward, code: x1 = self.up(x1)
        iota_6: "i64[640]" = torch.ops.prims.iota.default(640, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_12: "f32[640]" = torch.ops.prims.convert_element_type.default(iota_6, torch.float32);  iota_6 = None
        mul_127: "f32[640]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 0.49921752738654146);  convert_element_type_12 = None
        clamp_min_12: "f32[640]" = torch.ops.aten.clamp_min.default(mul_127, 0.0);  mul_127 = None
        view_6: "f32[640, 1]" = torch.ops.aten.reshape.default(clamp_min_12, [640, 1]);  clamp_min_12 = None
        convert_element_type_13: "i64[640, 1]" = torch.ops.prims.convert_element_type.default(view_6, torch.int64)
        add_95: "i64[640, 1]" = torch.ops.aten.add.Tensor(convert_element_type_13, 1)
        clamp_max_12: "i64[640, 1]" = torch.ops.aten.clamp_max.default(add_95, 319);  add_95 = None
        iota_7: "i64[958]" = torch.ops.prims.iota.default(958, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_14: "f32[958]" = torch.ops.prims.convert_element_type.default(iota_7, torch.float32);  iota_7 = None
        mul_128: "f32[958]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 0.4994775339602926);  convert_element_type_14 = None
        clamp_min_13: "f32[958]" = torch.ops.aten.clamp_min.default(mul_128, 0.0);  mul_128 = None
        convert_element_type_15: "i64[958]" = torch.ops.prims.convert_element_type.default(clamp_min_13, torch.int64)
        add_96: "i64[958]" = torch.ops.aten.add.Tensor(convert_element_type_15, 1)
        clamp_max_13: "i64[958]" = torch.ops.aten.clamp_max.default(add_96, 478);  add_96 = None
        _unsafe_index_12: "f32[8, 64, 640, 958]" = torch.ops.aten._unsafe_index.Tensor(relu_15, [None, None, convert_element_type_13, convert_element_type_15])
        _unsafe_index_13: "f32[8, 64, 640, 958]" = torch.ops.aten._unsafe_index.Tensor(relu_15, [None, None, convert_element_type_13, clamp_max_13])
        _unsafe_index_14: "f32[8, 64, 640, 958]" = torch.ops.aten._unsafe_index.Tensor(relu_15, [None, None, clamp_max_12, convert_element_type_15])
        _unsafe_index_15: "f32[8, 64, 640, 958]" = torch.ops.aten._unsafe_index.Tensor(relu_15, [None, None, clamp_max_12, clamp_max_13]);  relu_15 = None
        sub_31: "f32[958]" = torch.ops.aten.sub.Tensor(clamp_min_13, convert_element_type_15);  clamp_min_13 = None
        clamp_min_14: "f32[958]" = torch.ops.aten.clamp_min.default(sub_31, 0.0);  sub_31 = None
        clamp_max_14: "f32[958]" = torch.ops.aten.clamp_max.default(clamp_min_14, 1.0);  clamp_min_14 = None
        sub_32: "f32[8, 64, 640, 958]" = torch.ops.aten.sub.Tensor(_unsafe_index_13, _unsafe_index_12);  _unsafe_index_13 = None
        mul_129: "f32[8, 64, 640, 958]" = torch.ops.aten.mul.Tensor(sub_32, clamp_max_14);  sub_32 = None
        add_97: "f32[8, 64, 640, 958]" = torch.ops.aten.add.Tensor(_unsafe_index_12, mul_129);  _unsafe_index_12 = mul_129 = None
        sub_33: "f32[8, 64, 640, 958]" = torch.ops.aten.sub.Tensor(_unsafe_index_15, _unsafe_index_14);  _unsafe_index_15 = None
        mul_130: "f32[8, 64, 640, 958]" = torch.ops.aten.mul.Tensor(sub_33, clamp_max_14);  sub_33 = None
        add_98: "f32[8, 64, 640, 958]" = torch.ops.aten.add.Tensor(_unsafe_index_14, mul_130);  _unsafe_index_14 = mul_130 = None
        sub_34: "f32[640, 1]" = torch.ops.aten.sub.Tensor(view_6, convert_element_type_13);  view_6 = None
        clamp_min_15: "f32[640, 1]" = torch.ops.aten.clamp_min.default(sub_34, 0.0);  sub_34 = None
        clamp_max_15: "f32[640, 1]" = torch.ops.aten.clamp_max.default(clamp_min_15, 1.0);  clamp_min_15 = None
        sub_35: "f32[8, 64, 640, 958]" = torch.ops.aten.sub.Tensor(add_98, add_97);  add_98 = None
        mul_131: "f32[8, 64, 640, 958]" = torch.ops.aten.mul.Tensor(sub_35, clamp_max_15);  sub_35 = None
        add_99: "f32[8, 64, 640, 958]" = torch.ops.aten.add.Tensor(add_97, mul_131);  add_97 = mul_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_3: "f32[8, 64, 640, 959]" = torch.ops.aten.constant_pad_nd.default(add_99, [0, 1, 0, 0], 0.0);  add_99 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:67 in forward, code: x = torch.cat([x2, x1], dim=1)
        cat_3: "f32[8, 128, 640, 959]" = torch.ops.aten.cat.default([relu_1, constant_pad_nd_3], 1);  relu_1 = constant_pad_nd_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25 in forward, code: return self.double_conv(x)
        convolution_16: "f32[8, 64, 640, 959]" = torch.ops.aten.convolution.default(cat_3, primals_114, primals_115, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_115 = None
        add_100: "i64[]" = torch.ops.aten.add.Tensor(primals_116, 1)
        var_mean_16 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
        getitem_40: "f32[1, 64, 1, 1]" = var_mean_16[0]
        getitem_41: "f32[1, 64, 1, 1]" = var_mean_16[1];  var_mean_16 = None
        add_101: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05)
        rsqrt_16: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
        sub_36: "f32[8, 64, 640, 959]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_41)
        mul_132: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_16);  sub_36 = None
        squeeze_48: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
        squeeze_49: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_133: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
        mul_134: "f32[64]" = torch.ops.aten.mul.Tensor(primals_117, 0.9)
        add_102: "f32[64]" = torch.ops.aten.add.Tensor(mul_133, mul_134);  mul_133 = mul_134 = None
        squeeze_50: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
        mul_135: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.000000203662711);  squeeze_50 = None
        mul_136: "f32[64]" = torch.ops.aten.mul.Tensor(mul_135, 0.1);  mul_135 = None
        mul_137: "f32[64]" = torch.ops.aten.mul.Tensor(primals_118, 0.9)
        add_103: "f32[64]" = torch.ops.aten.add.Tensor(mul_136, mul_137);  mul_136 = mul_137 = None
        unsqueeze_64: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_119, -1)
        unsqueeze_65: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        mul_138: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(mul_132, unsqueeze_65);  mul_132 = unsqueeze_65 = None
        unsqueeze_66: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_120, -1);  primals_120 = None
        unsqueeze_67: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        add_104: "f32[8, 64, 640, 959]" = torch.ops.aten.add.Tensor(mul_138, unsqueeze_67);  mul_138 = unsqueeze_67 = None
        relu_16: "f32[8, 64, 640, 959]" = torch.ops.aten.relu.default(add_104);  add_104 = None
        convolution_17: "f32[8, 64, 640, 959]" = torch.ops.aten.convolution.default(relu_16, primals_121, primals_122, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_122 = None
        add_105: "i64[]" = torch.ops.aten.add.Tensor(primals_123, 1)
        var_mean_17 = torch.ops.aten.var_mean.correction(convolution_17, [0, 2, 3], correction = 0, keepdim = True)
        getitem_42: "f32[1, 64, 1, 1]" = var_mean_17[0]
        getitem_43: "f32[1, 64, 1, 1]" = var_mean_17[1];  var_mean_17 = None
        add_106: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05)
        rsqrt_17: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        sub_37: "f32[8, 64, 640, 959]" = torch.ops.aten.sub.Tensor(convolution_17, getitem_43)
        mul_139: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_17);  sub_37 = None
        squeeze_51: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
        squeeze_52: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
        mul_140: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
        mul_141: "f32[64]" = torch.ops.aten.mul.Tensor(primals_124, 0.9)
        add_107: "f32[64]" = torch.ops.aten.add.Tensor(mul_140, mul_141);  mul_140 = mul_141 = None
        squeeze_53: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
        mul_142: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.000000203662711);  squeeze_53 = None
        mul_143: "f32[64]" = torch.ops.aten.mul.Tensor(mul_142, 0.1);  mul_142 = None
        mul_144: "f32[64]" = torch.ops.aten.mul.Tensor(primals_125, 0.9)
        add_108: "f32[64]" = torch.ops.aten.add.Tensor(mul_143, mul_144);  mul_143 = mul_144 = None
        unsqueeze_68: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_126, -1)
        unsqueeze_69: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_145: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(mul_139, unsqueeze_69);  mul_139 = unsqueeze_69 = None
        unsqueeze_70: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_127, -1);  primals_127 = None
        unsqueeze_71: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_109: "f32[8, 64, 640, 959]" = torch.ops.aten.add.Tensor(mul_145, unsqueeze_71);  mul_145 = unsqueeze_71 = None
        relu_17: "f32[8, 64, 640, 959]" = torch.ops.aten.relu.default(add_109);  add_109 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:77 in forward, code: return self.conv(x)
        convolution_18: "f32[8, 2, 640, 959]" = torch.ops.aten.convolution.default(relu_17, primals_128, primals_129, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_129 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25 in forward, code: return self.double_conv(x)
        unsqueeze_72: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
        unsqueeze_73: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, 2);  unsqueeze_72 = None
        unsqueeze_74: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_73, 3);  unsqueeze_73 = None
        unsqueeze_84: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
        unsqueeze_85: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, 2);  unsqueeze_84 = None
        unsqueeze_86: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_85, 3);  unsqueeze_85 = None
        unsqueeze_108: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
        unsqueeze_109: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, 2);  unsqueeze_108 = None
        unsqueeze_110: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_109, 3);  unsqueeze_109 = None
        unsqueeze_132: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
        unsqueeze_133: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, 2);  unsqueeze_132 = None
        unsqueeze_134: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_133, 3);  unsqueeze_133 = None
        unsqueeze_156: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
        unsqueeze_157: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, 2);  unsqueeze_156 = None
        unsqueeze_158: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_157, 3);  unsqueeze_157 = None
        unsqueeze_180: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
        unsqueeze_181: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, 2);  unsqueeze_180 = None
        unsqueeze_182: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_181, 3);  unsqueeze_181 = None
        unsqueeze_204: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
        unsqueeze_205: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, 2);  unsqueeze_204 = None
        unsqueeze_206: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 3);  unsqueeze_205 = None
        unsqueeze_228: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
        unsqueeze_229: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, 2);  unsqueeze_228 = None
        unsqueeze_230: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 3);  unsqueeze_229 = None
        unsqueeze_252: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
        unsqueeze_253: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, 2);  unsqueeze_252 = None
        unsqueeze_254: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 3);  unsqueeze_253 = None
        unsqueeze_276: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_277: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 2);  unsqueeze_276 = None
        unsqueeze_278: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 3);  unsqueeze_277 = None

        # No stacktrace found for following nodes
        copy_: "i64[]" = torch.ops.aten.copy_.default(primals_4, add);  primals_4 = add = copy_ = None
        copy__1: "f32[64]" = torch.ops.aten.copy_.default(primals_5, add_2);  primals_5 = add_2 = copy__1 = None
        copy__2: "f32[64]" = torch.ops.aten.copy_.default(primals_6, add_3);  primals_6 = add_3 = copy__2 = None
        copy__3: "i64[]" = torch.ops.aten.copy_.default(primals_11, add_5);  primals_11 = add_5 = copy__3 = None
        copy__4: "f32[64]" = torch.ops.aten.copy_.default(primals_12, add_7);  primals_12 = add_7 = copy__4 = None
        copy__5: "f32[64]" = torch.ops.aten.copy_.default(primals_13, add_8);  primals_13 = add_8 = copy__5 = None
        copy__6: "i64[]" = torch.ops.aten.copy_.default(primals_18, add_10);  primals_18 = add_10 = copy__6 = None
        copy__7: "f32[128]" = torch.ops.aten.copy_.default(primals_19, add_12);  primals_19 = add_12 = copy__7 = None
        copy__8: "f32[128]" = torch.ops.aten.copy_.default(primals_20, add_13);  primals_20 = add_13 = copy__8 = None
        copy__9: "i64[]" = torch.ops.aten.copy_.default(primals_25, add_15);  primals_25 = add_15 = copy__9 = None
        copy__10: "f32[128]" = torch.ops.aten.copy_.default(primals_26, add_17);  primals_26 = add_17 = copy__10 = None
        copy__11: "f32[128]" = torch.ops.aten.copy_.default(primals_27, add_18);  primals_27 = add_18 = copy__11 = None
        copy__12: "i64[]" = torch.ops.aten.copy_.default(primals_32, add_20);  primals_32 = add_20 = copy__12 = None
        copy__13: "f32[256]" = torch.ops.aten.copy_.default(primals_33, add_22);  primals_33 = add_22 = copy__13 = None
        copy__14: "f32[256]" = torch.ops.aten.copy_.default(primals_34, add_23);  primals_34 = add_23 = copy__14 = None
        copy__15: "i64[]" = torch.ops.aten.copy_.default(primals_39, add_25);  primals_39 = add_25 = copy__15 = None
        copy__16: "f32[256]" = torch.ops.aten.copy_.default(primals_40, add_27);  primals_40 = add_27 = copy__16 = None
        copy__17: "f32[256]" = torch.ops.aten.copy_.default(primals_41, add_28);  primals_41 = add_28 = copy__17 = None
        copy__18: "i64[]" = torch.ops.aten.copy_.default(primals_46, add_30);  primals_46 = add_30 = copy__18 = None
        copy__19: "f32[512]" = torch.ops.aten.copy_.default(primals_47, add_32);  primals_47 = add_32 = copy__19 = None
        copy__20: "f32[512]" = torch.ops.aten.copy_.default(primals_48, add_33);  primals_48 = add_33 = copy__20 = None
        copy__21: "i64[]" = torch.ops.aten.copy_.default(primals_53, add_35);  primals_53 = add_35 = copy__21 = None
        copy__22: "f32[512]" = torch.ops.aten.copy_.default(primals_54, add_37);  primals_54 = add_37 = copy__22 = None
        copy__23: "f32[512]" = torch.ops.aten.copy_.default(primals_55, add_38);  primals_55 = add_38 = copy__23 = None
        copy__24: "i64[]" = torch.ops.aten.copy_.default(primals_60, add_40);  primals_60 = add_40 = copy__24 = None
        copy__25: "f32[512]" = torch.ops.aten.copy_.default(primals_61, add_42);  primals_61 = add_42 = copy__25 = None
        copy__26: "f32[512]" = torch.ops.aten.copy_.default(primals_62, add_43);  primals_62 = add_43 = copy__26 = None
        copy__27: "i64[]" = torch.ops.aten.copy_.default(primals_67, add_45);  primals_67 = add_45 = copy__27 = None
        copy__28: "f32[512]" = torch.ops.aten.copy_.default(primals_68, add_47);  primals_68 = add_47 = copy__28 = None
        copy__29: "f32[512]" = torch.ops.aten.copy_.default(primals_69, add_48);  primals_69 = add_48 = copy__29 = None
        copy__30: "i64[]" = torch.ops.aten.copy_.default(primals_74, add_55);  primals_74 = add_55 = copy__30 = None
        copy__31: "f32[512]" = torch.ops.aten.copy_.default(primals_75, add_57);  primals_75 = add_57 = copy__31 = None
        copy__32: "f32[512]" = torch.ops.aten.copy_.default(primals_76, add_58);  primals_76 = add_58 = copy__32 = None
        copy__33: "i64[]" = torch.ops.aten.copy_.default(primals_81, add_60);  primals_81 = add_60 = copy__33 = None
        copy__34: "f32[256]" = torch.ops.aten.copy_.default(primals_82, add_62);  primals_82 = add_62 = copy__34 = None
        copy__35: "f32[256]" = torch.ops.aten.copy_.default(primals_83, add_63);  primals_83 = add_63 = copy__35 = None
        copy__36: "i64[]" = torch.ops.aten.copy_.default(primals_88, add_70);  primals_88 = add_70 = copy__36 = None
        copy__37: "f32[256]" = torch.ops.aten.copy_.default(primals_89, add_72);  primals_89 = add_72 = copy__37 = None
        copy__38: "f32[256]" = torch.ops.aten.copy_.default(primals_90, add_73);  primals_90 = add_73 = copy__38 = None
        copy__39: "i64[]" = torch.ops.aten.copy_.default(primals_95, add_75);  primals_95 = add_75 = copy__39 = None
        copy__40: "f32[128]" = torch.ops.aten.copy_.default(primals_96, add_77);  primals_96 = add_77 = copy__40 = None
        copy__41: "f32[128]" = torch.ops.aten.copy_.default(primals_97, add_78);  primals_97 = add_78 = copy__41 = None
        copy__42: "i64[]" = torch.ops.aten.copy_.default(primals_102, add_85);  primals_102 = add_85 = copy__42 = None
        copy__43: "f32[128]" = torch.ops.aten.copy_.default(primals_103, add_87);  primals_103 = add_87 = copy__43 = None
        copy__44: "f32[128]" = torch.ops.aten.copy_.default(primals_104, add_88);  primals_104 = add_88 = copy__44 = None
        copy__45: "i64[]" = torch.ops.aten.copy_.default(primals_109, add_90);  primals_109 = add_90 = copy__45 = None
        copy__46: "f32[64]" = torch.ops.aten.copy_.default(primals_110, add_92);  primals_110 = add_92 = copy__46 = None
        copy__47: "f32[64]" = torch.ops.aten.copy_.default(primals_111, add_93);  primals_111 = add_93 = copy__47 = None
        copy__48: "i64[]" = torch.ops.aten.copy_.default(primals_116, add_100);  primals_116 = add_100 = copy__48 = None
        copy__49: "f32[64]" = torch.ops.aten.copy_.default(primals_117, add_102);  primals_117 = add_102 = copy__49 = None
        copy__50: "f32[64]" = torch.ops.aten.copy_.default(primals_118, add_103);  primals_118 = add_103 = copy__50 = None
        copy__51: "i64[]" = torch.ops.aten.copy_.default(primals_123, add_105);  primals_123 = add_105 = copy__51 = None
        copy__52: "f32[64]" = torch.ops.aten.copy_.default(primals_124, add_107);  primals_124 = add_107 = copy__52 = None
        copy__53: "f32[64]" = torch.ops.aten.copy_.default(primals_125, add_108);  primals_125 = add_108 = copy__53 = None
        return (convolution_18, primals_1, primals_3, primals_7, primals_9, primals_14, primals_15, primals_16, primals_21, primals_23, primals_28, primals_29, primals_30, primals_35, primals_37, primals_42, primals_43, primals_44, primals_49, primals_51, primals_56, primals_57, primals_58, primals_63, primals_65, primals_70, primals_71, primals_72, primals_77, primals_79, primals_84, primals_85, primals_86, primals_91, primals_93, primals_98, primals_99, primals_100, primals_105, primals_107, primals_112, primals_113, primals_114, primals_119, primals_121, primals_126, primals_128, convolution, squeeze_1, relu, convolution_1, getitem_3, rsqrt_1, getitem_4, getitem_5, convolution_2, squeeze_7, relu_2, convolution_3, getitem_9, rsqrt_3, getitem_10, getitem_11, convolution_4, squeeze_13, relu_4, convolution_5, getitem_15, rsqrt_5, getitem_16, getitem_17, convolution_6, squeeze_19, relu_6, convolution_7, getitem_21, rsqrt_7, getitem_22, getitem_23, convolution_8, squeeze_25, relu_8, convolution_9, getitem_27, rsqrt_9, convert_element_type_1, clamp_max, convert_element_type_3, clamp_max_1, clamp_max_2, clamp_max_3, cat, convolution_10, squeeze_31, relu_10, convolution_11, getitem_31, rsqrt_11, convert_element_type_5, clamp_max_4, convert_element_type_7, clamp_max_5, clamp_max_6, clamp_max_7, cat_1, convolution_12, squeeze_37, relu_12, convolution_13, getitem_35, rsqrt_13, convert_element_type_9, clamp_max_8, convert_element_type_11, clamp_max_9, clamp_max_10, clamp_max_11, cat_2, convolution_14, squeeze_43, relu_14, convolution_15, getitem_39, rsqrt_15, convert_element_type_13, clamp_max_12, convert_element_type_15, clamp_max_13, clamp_max_14, clamp_max_15, cat_3, convolution_16, squeeze_49, relu_16, convolution_17, squeeze_52, relu_17, unsqueeze_74, unsqueeze_86, unsqueeze_110, unsqueeze_134, unsqueeze_158, unsqueeze_182, unsqueeze_206, unsqueeze_230, unsqueeze_254, unsqueeze_278)
