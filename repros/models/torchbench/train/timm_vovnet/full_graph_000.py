class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[64, 3, 3, 3]", primals_2: "f32[32, 3, 224, 224]", primals_3: "i64[]", primals_4: "f32[64]", primals_5: "f32[64]", primals_6: "f32[64]", primals_7: "f32[64]", primals_8: "f32[64, 64, 3, 3]", primals_9: "i64[]", primals_10: "f32[64]", primals_11: "f32[64]", primals_12: "f32[64]", primals_13: "f32[64]", primals_14: "f32[128, 64, 3, 3]", primals_15: "i64[]", primals_16: "f32[128]", primals_17: "f32[128]", primals_18: "f32[128]", primals_19: "f32[128]", primals_20: "f32[128, 128, 3, 3]", primals_21: "i64[]", primals_22: "f32[128]", primals_23: "f32[128]", primals_24: "f32[128]", primals_25: "f32[128]", primals_26: "f32[128, 128, 3, 3]", primals_27: "i64[]", primals_28: "f32[128]", primals_29: "f32[128]", primals_30: "f32[128]", primals_31: "f32[128]", primals_32: "f32[128, 128, 3, 3]", primals_33: "i64[]", primals_34: "f32[128]", primals_35: "f32[128]", primals_36: "f32[128]", primals_37: "f32[128]", primals_38: "f32[128, 128, 3, 3]", primals_39: "i64[]", primals_40: "f32[128]", primals_41: "f32[128]", primals_42: "f32[128]", primals_43: "f32[128]", primals_44: "f32[128, 128, 3, 3]", primals_45: "i64[]", primals_46: "f32[128]", primals_47: "f32[128]", primals_48: "f32[128]", primals_49: "f32[128]", primals_50: "f32[256, 768, 1, 1]", primals_51: "i64[]", primals_52: "f32[256]", primals_53: "f32[256]", primals_54: "f32[256]", primals_55: "f32[256]", primals_56: "f32[160, 256, 3, 3]", primals_57: "i64[]", primals_58: "f32[160]", primals_59: "f32[160]", primals_60: "f32[160]", primals_61: "f32[160]", primals_62: "f32[160, 160, 3, 3]", primals_63: "i64[]", primals_64: "f32[160]", primals_65: "f32[160]", primals_66: "f32[160]", primals_67: "f32[160]", primals_68: "f32[160, 160, 3, 3]", primals_69: "i64[]", primals_70: "f32[160]", primals_71: "f32[160]", primals_72: "f32[160]", primals_73: "f32[160]", primals_74: "f32[160, 160, 3, 3]", primals_75: "i64[]", primals_76: "f32[160]", primals_77: "f32[160]", primals_78: "f32[160]", primals_79: "f32[160]", primals_80: "f32[160, 160, 3, 3]", primals_81: "i64[]", primals_82: "f32[160]", primals_83: "f32[160]", primals_84: "f32[160]", primals_85: "f32[160]", primals_86: "f32[512, 1056, 1, 1]", primals_87: "i64[]", primals_88: "f32[512]", primals_89: "f32[512]", primals_90: "f32[512]", primals_91: "f32[512]", primals_92: "f32[192, 512, 3, 3]", primals_93: "i64[]", primals_94: "f32[192]", primals_95: "f32[192]", primals_96: "f32[192]", primals_97: "f32[192]", primals_98: "f32[192, 192, 3, 3]", primals_99: "i64[]", primals_100: "f32[192]", primals_101: "f32[192]", primals_102: "f32[192]", primals_103: "f32[192]", primals_104: "f32[192, 192, 3, 3]", primals_105: "i64[]", primals_106: "f32[192]", primals_107: "f32[192]", primals_108: "f32[192]", primals_109: "f32[192]", primals_110: "f32[192, 192, 3, 3]", primals_111: "i64[]", primals_112: "f32[192]", primals_113: "f32[192]", primals_114: "f32[192]", primals_115: "f32[192]", primals_116: "f32[192, 192, 3, 3]", primals_117: "i64[]", primals_118: "f32[192]", primals_119: "f32[192]", primals_120: "f32[192]", primals_121: "f32[192]", primals_122: "f32[768, 1472, 1, 1]", primals_123: "i64[]", primals_124: "f32[768]", primals_125: "f32[768]", primals_126: "f32[768]", primals_127: "f32[768]", primals_128: "f32[192, 768, 3, 3]", primals_129: "i64[]", primals_130: "f32[192]", primals_131: "f32[192]", primals_132: "f32[192]", primals_133: "f32[192]", primals_134: "f32[192, 192, 3, 3]", primals_135: "i64[]", primals_136: "f32[192]", primals_137: "f32[192]", primals_138: "f32[192]", primals_139: "f32[192]", primals_140: "f32[192, 192, 3, 3]", primals_141: "i64[]", primals_142: "f32[192]", primals_143: "f32[192]", primals_144: "f32[192]", primals_145: "f32[192]", primals_146: "f32[192, 192, 3, 3]", primals_147: "i64[]", primals_148: "f32[192]", primals_149: "f32[192]", primals_150: "f32[192]", primals_151: "f32[192]", primals_152: "f32[192, 192, 3, 3]", primals_153: "i64[]", primals_154: "f32[192]", primals_155: "f32[192]", primals_156: "f32[192]", primals_157: "f32[192]", primals_158: "f32[768, 1728, 1, 1]", primals_159: "i64[]", primals_160: "f32[768]", primals_161: "f32[768]", primals_162: "f32[768]", primals_163: "f32[768]", primals_164: "f32[224, 768, 3, 3]", primals_165: "i64[]", primals_166: "f32[224]", primals_167: "f32[224]", primals_168: "f32[224]", primals_169: "f32[224]", primals_170: "f32[224, 224, 3, 3]", primals_171: "i64[]", primals_172: "f32[224]", primals_173: "f32[224]", primals_174: "f32[224]", primals_175: "f32[224]", primals_176: "f32[224, 224, 3, 3]", primals_177: "i64[]", primals_178: "f32[224]", primals_179: "f32[224]", primals_180: "f32[224]", primals_181: "f32[224]", primals_182: "f32[224, 224, 3, 3]", primals_183: "i64[]", primals_184: "f32[224]", primals_185: "f32[224]", primals_186: "f32[224]", primals_187: "f32[224]", primals_188: "f32[224, 224, 3, 3]", primals_189: "i64[]", primals_190: "f32[224]", primals_191: "f32[224]", primals_192: "f32[224]", primals_193: "f32[224]", primals_194: "f32[1024, 1888, 1, 1]", primals_195: "i64[]", primals_196: "f32[1024]", primals_197: "f32[1024]", primals_198: "f32[1024]", primals_199: "f32[1024]", primals_200: "f32[224, 1024, 3, 3]", primals_201: "i64[]", primals_202: "f32[224]", primals_203: "f32[224]", primals_204: "f32[224]", primals_205: "f32[224]", primals_206: "f32[224, 224, 3, 3]", primals_207: "i64[]", primals_208: "f32[224]", primals_209: "f32[224]", primals_210: "f32[224]", primals_211: "f32[224]", primals_212: "f32[224, 224, 3, 3]", primals_213: "i64[]", primals_214: "f32[224]", primals_215: "f32[224]", primals_216: "f32[224]", primals_217: "f32[224]", primals_218: "f32[224, 224, 3, 3]", primals_219: "i64[]", primals_220: "f32[224]", primals_221: "f32[224]", primals_222: "f32[224]", primals_223: "f32[224]", primals_224: "f32[224, 224, 3, 3]", primals_225: "i64[]", primals_226: "f32[224]", primals_227: "f32[224]", primals_228: "f32[224]", primals_229: "f32[224]", primals_230: "f32[1024, 2144, 1, 1]", primals_231: "i64[]", primals_232: "f32[1024]", primals_233: "f32[1024]", primals_234: "f32[1024]", primals_235: "f32[1024]", primals_236: "f32[1000, 1024]", primals_237: "f32[1000]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution: "f32[32, 64, 112, 112]" = torch.ops.aten.convolution.default(primals_2, primals_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add: "i64[]" = torch.ops.aten.add.Tensor(primals_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 64, 1, 1]" = var_mean[0]
        getitem_1: "f32[1, 64, 1, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[32, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul: "f32[32, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_1: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_1: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
        mul_2: "f32[64]" = torch.ops.aten.mul.Tensor(primals_4, 0.9)
        add_2: "f32[64]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_2: "f32[64]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_3: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.0000024912370735);  squeeze_2 = None
        mul_4: "f32[64]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
        mul_5: "f32[64]" = torch.ops.aten.mul.Tensor(primals_5, 0.9)
        add_3: "f32[64]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1)
        unsqueeze_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[32, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1);  primals_7 = None
        unsqueeze_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_4: "f32[32, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu: "f32[32, 64, 112, 112]" = torch.ops.aten.relu.default(add_4);  add_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_1: "f32[32, 64, 112, 112]" = torch.ops.aten.convolution.default(relu, primals_8, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_9, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[1, 64, 1, 1]" = var_mean_1[0]
        getitem_3: "f32[1, 64, 1, 1]" = var_mean_1[1];  var_mean_1 = None
        add_6: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
        rsqrt_1: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_1: "f32[32, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
        mul_7: "f32[32, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        squeeze_3: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        squeeze_4: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_8: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
        mul_9: "f32[64]" = torch.ops.aten.mul.Tensor(primals_10, 0.9)
        add_7: "f32[64]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        squeeze_5: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_10: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000024912370735);  squeeze_5 = None
        mul_11: "f32[64]" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
        mul_12: "f32[64]" = torch.ops.aten.mul.Tensor(primals_11, 0.9)
        add_8: "f32[64]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        unsqueeze_4: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1)
        unsqueeze_5: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_13: "f32[32, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
        unsqueeze_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1);  primals_13 = None
        unsqueeze_7: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_9: "f32[32, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_1: "f32[32, 64, 112, 112]" = torch.ops.aten.relu.default(add_9);  add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_2: "f32[32, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_1, primals_14, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_15, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
        getitem_4: "f32[1, 128, 1, 1]" = var_mean_2[0]
        getitem_5: "f32[1, 128, 1, 1]" = var_mean_2[1];  var_mean_2 = None
        add_11: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
        rsqrt_2: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_2: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
        mul_14: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        squeeze_6: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        squeeze_7: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_15: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
        mul_16: "f32[128]" = torch.ops.aten.mul.Tensor(primals_16, 0.9)
        add_12: "f32[128]" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
        squeeze_8: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_17: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.00000996502277);  squeeze_8 = None
        mul_18: "f32[128]" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
        mul_19: "f32[128]" = torch.ops.aten.mul.Tensor(primals_17, 0.9)
        add_13: "f32[128]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
        unsqueeze_8: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1)
        unsqueeze_9: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        mul_20: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
        unsqueeze_10: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1);  primals_19 = None
        unsqueeze_11: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        add_14: "f32[32, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_2: "f32[32, 128, 56, 56]" = torch.ops.aten.relu.default(add_14);  add_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_3: "f32[32, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_2, primals_20, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_15: "i64[]" = torch.ops.aten.add.Tensor(primals_21, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
        getitem_6: "f32[1, 128, 1, 1]" = var_mean_3[0]
        getitem_7: "f32[1, 128, 1, 1]" = var_mean_3[1];  var_mean_3 = None
        add_16: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
        rsqrt_3: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_3: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_7)
        mul_21: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        squeeze_9: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        squeeze_10: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_22: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
        mul_23: "f32[128]" = torch.ops.aten.mul.Tensor(primals_22, 0.9)
        add_17: "f32[128]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
        squeeze_11: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_24: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.00000996502277);  squeeze_11 = None
        mul_25: "f32[128]" = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
        mul_26: "f32[128]" = torch.ops.aten.mul.Tensor(primals_23, 0.9)
        add_18: "f32[128]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
        unsqueeze_12: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1)
        unsqueeze_13: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_27: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
        unsqueeze_14: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1);  primals_25 = None
        unsqueeze_15: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_19: "f32[32, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_3: "f32[32, 128, 56, 56]" = torch.ops.aten.relu.default(add_19);  add_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_4: "f32[32, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_3, primals_26, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_20: "i64[]" = torch.ops.aten.add.Tensor(primals_27, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
        getitem_8: "f32[1, 128, 1, 1]" = var_mean_4[0]
        getitem_9: "f32[1, 128, 1, 1]" = var_mean_4[1];  var_mean_4 = None
        add_21: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
        rsqrt_4: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_4: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_9)
        mul_28: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        squeeze_12: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        squeeze_13: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_29: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
        mul_30: "f32[128]" = torch.ops.aten.mul.Tensor(primals_28, 0.9)
        add_22: "f32[128]" = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
        squeeze_14: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_31: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.00000996502277);  squeeze_14 = None
        mul_32: "f32[128]" = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
        mul_33: "f32[128]" = torch.ops.aten.mul.Tensor(primals_29, 0.9)
        add_23: "f32[128]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
        unsqueeze_16: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1)
        unsqueeze_17: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        mul_34: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
        unsqueeze_18: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1);  primals_31 = None
        unsqueeze_19: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        add_24: "f32[32, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_4: "f32[32, 128, 56, 56]" = torch.ops.aten.relu.default(add_24);  add_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_5: "f32[32, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_4, primals_32, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_25: "i64[]" = torch.ops.aten.add.Tensor(primals_33, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_5 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
        getitem_10: "f32[1, 128, 1, 1]" = var_mean_5[0]
        getitem_11: "f32[1, 128, 1, 1]" = var_mean_5[1];  var_mean_5 = None
        add_26: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
        rsqrt_5: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_5: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_11)
        mul_35: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        squeeze_15: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        squeeze_16: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_36: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
        mul_37: "f32[128]" = torch.ops.aten.mul.Tensor(primals_34, 0.9)
        add_27: "f32[128]" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
        squeeze_17: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_38: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.00000996502277);  squeeze_17 = None
        mul_39: "f32[128]" = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
        mul_40: "f32[128]" = torch.ops.aten.mul.Tensor(primals_35, 0.9)
        add_28: "f32[128]" = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
        unsqueeze_20: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1)
        unsqueeze_21: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_41: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
        unsqueeze_22: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1);  primals_37 = None
        unsqueeze_23: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_29: "f32[32, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_5: "f32[32, 128, 56, 56]" = torch.ops.aten.relu.default(add_29);  add_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_6: "f32[32, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_5, primals_38, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_30: "i64[]" = torch.ops.aten.add.Tensor(primals_39, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_6 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
        getitem_12: "f32[1, 128, 1, 1]" = var_mean_6[0]
        getitem_13: "f32[1, 128, 1, 1]" = var_mean_6[1];  var_mean_6 = None
        add_31: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
        rsqrt_6: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        sub_6: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_13)
        mul_42: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        squeeze_18: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        squeeze_19: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_43: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
        mul_44: "f32[128]" = torch.ops.aten.mul.Tensor(primals_40, 0.9)
        add_32: "f32[128]" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
        squeeze_20: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
        mul_45: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.00000996502277);  squeeze_20 = None
        mul_46: "f32[128]" = torch.ops.aten.mul.Tensor(mul_45, 0.1);  mul_45 = None
        mul_47: "f32[128]" = torch.ops.aten.mul.Tensor(primals_41, 0.9)
        add_33: "f32[128]" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
        unsqueeze_24: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1)
        unsqueeze_25: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        mul_48: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_25);  mul_42 = unsqueeze_25 = None
        unsqueeze_26: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1);  primals_43 = None
        unsqueeze_27: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        add_34: "f32[32, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_48, unsqueeze_27);  mul_48 = unsqueeze_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_6: "f32[32, 128, 56, 56]" = torch.ops.aten.relu.default(add_34);  add_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_7: "f32[32, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_6, primals_44, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_35: "i64[]" = torch.ops.aten.add.Tensor(primals_45, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_7 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
        getitem_14: "f32[1, 128, 1, 1]" = var_mean_7[0]
        getitem_15: "f32[1, 128, 1, 1]" = var_mean_7[1];  var_mean_7 = None
        add_36: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
        rsqrt_7: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_7: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_15)
        mul_49: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        squeeze_21: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3])
        mul_50: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1);  squeeze_21 = None
        mul_51: "f32[128]" = torch.ops.aten.mul.Tensor(primals_46, 0.9)
        add_37: "f32[128]" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
        squeeze_23: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
        mul_52: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.00000996502277);  squeeze_23 = None
        mul_53: "f32[128]" = torch.ops.aten.mul.Tensor(mul_52, 0.1);  mul_52 = None
        mul_54: "f32[128]" = torch.ops.aten.mul.Tensor(primals_47, 0.9)
        add_38: "f32[128]" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
        unsqueeze_28: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1)
        unsqueeze_29: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_55: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
        unsqueeze_30: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1)
        unsqueeze_31: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_39: "f32[32, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_7: "f32[32, 128, 56, 56]" = torch.ops.aten.relu.default(add_39);  add_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vovnet.py:40 in forward, code: x = torch.cat(concat_list, dim=1)
        cat: "f32[32, 768, 56, 56]" = torch.ops.aten.cat.default([relu_2, relu_3, relu_4, relu_5, relu_6, relu_7], 1);  relu_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_8: "f32[32, 256, 56, 56]" = torch.ops.aten.convolution.default(cat, primals_50, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_40: "i64[]" = torch.ops.aten.add.Tensor(primals_51, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_8 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
        getitem_16: "f32[1, 256, 1, 1]" = var_mean_8[0]
        getitem_17: "f32[1, 256, 1, 1]" = var_mean_8[1];  var_mean_8 = None
        add_41: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
        rsqrt_8: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
        sub_8: "f32[32, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_17)
        mul_56: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        squeeze_24: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3])
        mul_57: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1);  squeeze_24 = None
        mul_58: "f32[256]" = torch.ops.aten.mul.Tensor(primals_52, 0.9)
        add_42: "f32[256]" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
        squeeze_26: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
        mul_59: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.00000996502277);  squeeze_26 = None
        mul_60: "f32[256]" = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
        mul_61: "f32[256]" = torch.ops.aten.mul.Tensor(primals_53, 0.9)
        add_43: "f32[256]" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
        unsqueeze_32: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1)
        unsqueeze_33: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        mul_62: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
        unsqueeze_34: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1)
        unsqueeze_35: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        add_44: "f32[32, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_8: "f32[32, 256, 56, 56]" = torch.ops.aten.relu.default(add_44);  add_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vovnet.py:161 in forward, code: x = self.pool(x)
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_8, [3, 3], [2, 2], [0, 0], [1, 1], True);  relu_8 = None
        getitem_18: "f32[32, 256, 28, 28]" = _low_memory_max_pool_with_offsets[0]
        getitem_19: "i8[32, 256, 28, 28]" = _low_memory_max_pool_with_offsets[1];  _low_memory_max_pool_with_offsets = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_9: "f32[32, 160, 28, 28]" = torch.ops.aten.convolution.default(getitem_18, primals_56, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_45: "i64[]" = torch.ops.aten.add.Tensor(primals_57, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_9 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
        getitem_20: "f32[1, 160, 1, 1]" = var_mean_9[0]
        getitem_21: "f32[1, 160, 1, 1]" = var_mean_9[1];  var_mean_9 = None
        add_46: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
        rsqrt_9: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_9: "f32[32, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_21)
        mul_63: "f32[32, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        squeeze_27: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        squeeze_28: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_64: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
        mul_65: "f32[160]" = torch.ops.aten.mul.Tensor(primals_58, 0.9)
        add_47: "f32[160]" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
        squeeze_29: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        mul_66: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0000398612827361);  squeeze_29 = None
        mul_67: "f32[160]" = torch.ops.aten.mul.Tensor(mul_66, 0.1);  mul_66 = None
        mul_68: "f32[160]" = torch.ops.aten.mul.Tensor(primals_59, 0.9)
        add_48: "f32[160]" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
        unsqueeze_36: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1)
        unsqueeze_37: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_69: "f32[32, 160, 28, 28]" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_37);  mul_63 = unsqueeze_37 = None
        unsqueeze_38: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_61, -1);  primals_61 = None
        unsqueeze_39: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_49: "f32[32, 160, 28, 28]" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_39);  mul_69 = unsqueeze_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_9: "f32[32, 160, 28, 28]" = torch.ops.aten.relu.default(add_49);  add_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_10: "f32[32, 160, 28, 28]" = torch.ops.aten.convolution.default(relu_9, primals_62, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_50: "i64[]" = torch.ops.aten.add.Tensor(primals_63, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_10 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
        getitem_22: "f32[1, 160, 1, 1]" = var_mean_10[0]
        getitem_23: "f32[1, 160, 1, 1]" = var_mean_10[1];  var_mean_10 = None
        add_51: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
        rsqrt_10: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        sub_10: "f32[32, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_23)
        mul_70: "f32[32, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        squeeze_30: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
        squeeze_31: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_71: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
        mul_72: "f32[160]" = torch.ops.aten.mul.Tensor(primals_64, 0.9)
        add_52: "f32[160]" = torch.ops.aten.add.Tensor(mul_71, mul_72);  mul_71 = mul_72 = None
        squeeze_32: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
        mul_73: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0000398612827361);  squeeze_32 = None
        mul_74: "f32[160]" = torch.ops.aten.mul.Tensor(mul_73, 0.1);  mul_73 = None
        mul_75: "f32[160]" = torch.ops.aten.mul.Tensor(primals_65, 0.9)
        add_53: "f32[160]" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
        unsqueeze_40: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1)
        unsqueeze_41: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        mul_76: "f32[32, 160, 28, 28]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_41);  mul_70 = unsqueeze_41 = None
        unsqueeze_42: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_67, -1);  primals_67 = None
        unsqueeze_43: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        add_54: "f32[32, 160, 28, 28]" = torch.ops.aten.add.Tensor(mul_76, unsqueeze_43);  mul_76 = unsqueeze_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_10: "f32[32, 160, 28, 28]" = torch.ops.aten.relu.default(add_54);  add_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_11: "f32[32, 160, 28, 28]" = torch.ops.aten.convolution.default(relu_10, primals_68, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_55: "i64[]" = torch.ops.aten.add.Tensor(primals_69, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_11 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
        getitem_24: "f32[1, 160, 1, 1]" = var_mean_11[0]
        getitem_25: "f32[1, 160, 1, 1]" = var_mean_11[1];  var_mean_11 = None
        add_56: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
        rsqrt_11: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        sub_11: "f32[32, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_25)
        mul_77: "f32[32, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        squeeze_33: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        squeeze_34: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
        mul_78: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
        mul_79: "f32[160]" = torch.ops.aten.mul.Tensor(primals_70, 0.9)
        add_57: "f32[160]" = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
        squeeze_35: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
        mul_80: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0000398612827361);  squeeze_35 = None
        mul_81: "f32[160]" = torch.ops.aten.mul.Tensor(mul_80, 0.1);  mul_80 = None
        mul_82: "f32[160]" = torch.ops.aten.mul.Tensor(primals_71, 0.9)
        add_58: "f32[160]" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
        unsqueeze_44: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1)
        unsqueeze_45: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_83: "f32[32, 160, 28, 28]" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_45);  mul_77 = unsqueeze_45 = None
        unsqueeze_46: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_73, -1);  primals_73 = None
        unsqueeze_47: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_59: "f32[32, 160, 28, 28]" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_47);  mul_83 = unsqueeze_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_11: "f32[32, 160, 28, 28]" = torch.ops.aten.relu.default(add_59);  add_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_12: "f32[32, 160, 28, 28]" = torch.ops.aten.convolution.default(relu_11, primals_74, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_60: "i64[]" = torch.ops.aten.add.Tensor(primals_75, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_12 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
        getitem_26: "f32[1, 160, 1, 1]" = var_mean_12[0]
        getitem_27: "f32[1, 160, 1, 1]" = var_mean_12[1];  var_mean_12 = None
        add_61: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
        rsqrt_12: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        sub_12: "f32[32, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_12, getitem_27)
        mul_84: "f32[32, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        squeeze_36: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        squeeze_37: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_85: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
        mul_86: "f32[160]" = torch.ops.aten.mul.Tensor(primals_76, 0.9)
        add_62: "f32[160]" = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
        squeeze_38: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        mul_87: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0000398612827361);  squeeze_38 = None
        mul_88: "f32[160]" = torch.ops.aten.mul.Tensor(mul_87, 0.1);  mul_87 = None
        mul_89: "f32[160]" = torch.ops.aten.mul.Tensor(primals_77, 0.9)
        add_63: "f32[160]" = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
        unsqueeze_48: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1)
        unsqueeze_49: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        mul_90: "f32[32, 160, 28, 28]" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_49);  mul_84 = unsqueeze_49 = None
        unsqueeze_50: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_79, -1);  primals_79 = None
        unsqueeze_51: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        add_64: "f32[32, 160, 28, 28]" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_51);  mul_90 = unsqueeze_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_12: "f32[32, 160, 28, 28]" = torch.ops.aten.relu.default(add_64);  add_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_13: "f32[32, 160, 28, 28]" = torch.ops.aten.convolution.default(relu_12, primals_80, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_65: "i64[]" = torch.ops.aten.add.Tensor(primals_81, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_13 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
        getitem_28: "f32[1, 160, 1, 1]" = var_mean_13[0]
        getitem_29: "f32[1, 160, 1, 1]" = var_mean_13[1];  var_mean_13 = None
        add_66: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
        rsqrt_13: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_13: "f32[32, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_29)
        mul_91: "f32[32, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        squeeze_39: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3])
        mul_92: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1);  squeeze_39 = None
        mul_93: "f32[160]" = torch.ops.aten.mul.Tensor(primals_82, 0.9)
        add_67: "f32[160]" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
        squeeze_41: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        mul_94: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0000398612827361);  squeeze_41 = None
        mul_95: "f32[160]" = torch.ops.aten.mul.Tensor(mul_94, 0.1);  mul_94 = None
        mul_96: "f32[160]" = torch.ops.aten.mul.Tensor(primals_83, 0.9)
        add_68: "f32[160]" = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None
        unsqueeze_52: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1)
        unsqueeze_53: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_97: "f32[32, 160, 28, 28]" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_53);  mul_91 = unsqueeze_53 = None
        unsqueeze_54: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_85, -1)
        unsqueeze_55: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_69: "f32[32, 160, 28, 28]" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_55);  mul_97 = unsqueeze_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_13: "f32[32, 160, 28, 28]" = torch.ops.aten.relu.default(add_69);  add_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vovnet.py:40 in forward, code: x = torch.cat(concat_list, dim=1)
        cat_1: "f32[32, 1056, 28, 28]" = torch.ops.aten.cat.default([getitem_18, relu_9, relu_10, relu_11, relu_12, relu_13], 1);  relu_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_14: "f32[32, 512, 28, 28]" = torch.ops.aten.convolution.default(cat_1, primals_86, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_70: "i64[]" = torch.ops.aten.add.Tensor(primals_87, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_14 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
        getitem_30: "f32[1, 512, 1, 1]" = var_mean_14[0]
        getitem_31: "f32[1, 512, 1, 1]" = var_mean_14[1];  var_mean_14 = None
        add_71: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
        rsqrt_14: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        sub_14: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_31)
        mul_98: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        squeeze_42: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3])
        mul_99: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1);  squeeze_42 = None
        mul_100: "f32[512]" = torch.ops.aten.mul.Tensor(primals_88, 0.9)
        add_72: "f32[512]" = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
        squeeze_44: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
        mul_101: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0000398612827361);  squeeze_44 = None
        mul_102: "f32[512]" = torch.ops.aten.mul.Tensor(mul_101, 0.1);  mul_101 = None
        mul_103: "f32[512]" = torch.ops.aten.mul.Tensor(primals_89, 0.9)
        add_73: "f32[512]" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
        unsqueeze_56: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1)
        unsqueeze_57: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        mul_104: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_57);  mul_98 = unsqueeze_57 = None
        unsqueeze_58: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_91, -1)
        unsqueeze_59: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        add_74: "f32[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_59);  mul_104 = unsqueeze_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_14: "f32[32, 512, 28, 28]" = torch.ops.aten.relu.default(add_74);  add_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vovnet.py:161 in forward, code: x = self.pool(x)
        _low_memory_max_pool_with_offsets_1 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_14, [3, 3], [2, 2], [0, 0], [1, 1], True);  relu_14 = None
        getitem_32: "f32[32, 512, 14, 14]" = _low_memory_max_pool_with_offsets_1[0]
        getitem_33: "i8[32, 512, 14, 14]" = _low_memory_max_pool_with_offsets_1[1];  _low_memory_max_pool_with_offsets_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_15: "f32[32, 192, 14, 14]" = torch.ops.aten.convolution.default(getitem_32, primals_92, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_75: "i64[]" = torch.ops.aten.add.Tensor(primals_93, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_15 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
        getitem_34: "f32[1, 192, 1, 1]" = var_mean_15[0]
        getitem_35: "f32[1, 192, 1, 1]" = var_mean_15[1];  var_mean_15 = None
        add_76: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
        rsqrt_15: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        sub_15: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_35)
        mul_105: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        squeeze_45: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
        squeeze_46: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_106: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
        mul_107: "f32[192]" = torch.ops.aten.mul.Tensor(primals_94, 0.9)
        add_77: "f32[192]" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
        squeeze_47: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
        mul_108: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0001594642002871);  squeeze_47 = None
        mul_109: "f32[192]" = torch.ops.aten.mul.Tensor(mul_108, 0.1);  mul_108 = None
        mul_110: "f32[192]" = torch.ops.aten.mul.Tensor(primals_95, 0.9)
        add_78: "f32[192]" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
        unsqueeze_60: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1)
        unsqueeze_61: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_111: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_61);  mul_105 = unsqueeze_61 = None
        unsqueeze_62: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_97, -1);  primals_97 = None
        unsqueeze_63: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_79: "f32[32, 192, 14, 14]" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_63);  mul_111 = unsqueeze_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_15: "f32[32, 192, 14, 14]" = torch.ops.aten.relu.default(add_79);  add_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_16: "f32[32, 192, 14, 14]" = torch.ops.aten.convolution.default(relu_15, primals_98, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_80: "i64[]" = torch.ops.aten.add.Tensor(primals_99, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_16 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
        getitem_36: "f32[1, 192, 1, 1]" = var_mean_16[0]
        getitem_37: "f32[1, 192, 1, 1]" = var_mean_16[1];  var_mean_16 = None
        add_81: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
        rsqrt_16: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
        sub_16: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_37)
        mul_112: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        squeeze_48: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
        squeeze_49: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_113: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
        mul_114: "f32[192]" = torch.ops.aten.mul.Tensor(primals_100, 0.9)
        add_82: "f32[192]" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
        squeeze_50: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
        mul_115: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0001594642002871);  squeeze_50 = None
        mul_116: "f32[192]" = torch.ops.aten.mul.Tensor(mul_115, 0.1);  mul_115 = None
        mul_117: "f32[192]" = torch.ops.aten.mul.Tensor(primals_101, 0.9)
        add_83: "f32[192]" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
        unsqueeze_64: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_102, -1)
        unsqueeze_65: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        mul_118: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_65);  mul_112 = unsqueeze_65 = None
        unsqueeze_66: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_103, -1);  primals_103 = None
        unsqueeze_67: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        add_84: "f32[32, 192, 14, 14]" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_67);  mul_118 = unsqueeze_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_16: "f32[32, 192, 14, 14]" = torch.ops.aten.relu.default(add_84);  add_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_17: "f32[32, 192, 14, 14]" = torch.ops.aten.convolution.default(relu_16, primals_104, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_85: "i64[]" = torch.ops.aten.add.Tensor(primals_105, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_17 = torch.ops.aten.var_mean.correction(convolution_17, [0, 2, 3], correction = 0, keepdim = True)
        getitem_38: "f32[1, 192, 1, 1]" = var_mean_17[0]
        getitem_39: "f32[1, 192, 1, 1]" = var_mean_17[1];  var_mean_17 = None
        add_86: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05)
        rsqrt_17: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        sub_17: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_17, getitem_39)
        mul_119: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        squeeze_51: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
        squeeze_52: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
        mul_120: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
        mul_121: "f32[192]" = torch.ops.aten.mul.Tensor(primals_106, 0.9)
        add_87: "f32[192]" = torch.ops.aten.add.Tensor(mul_120, mul_121);  mul_120 = mul_121 = None
        squeeze_53: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
        mul_122: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0001594642002871);  squeeze_53 = None
        mul_123: "f32[192]" = torch.ops.aten.mul.Tensor(mul_122, 0.1);  mul_122 = None
        mul_124: "f32[192]" = torch.ops.aten.mul.Tensor(primals_107, 0.9)
        add_88: "f32[192]" = torch.ops.aten.add.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
        unsqueeze_68: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_108, -1)
        unsqueeze_69: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_125: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_69);  mul_119 = unsqueeze_69 = None
        unsqueeze_70: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_109, -1);  primals_109 = None
        unsqueeze_71: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_89: "f32[32, 192, 14, 14]" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_71);  mul_125 = unsqueeze_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_17: "f32[32, 192, 14, 14]" = torch.ops.aten.relu.default(add_89);  add_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_18: "f32[32, 192, 14, 14]" = torch.ops.aten.convolution.default(relu_17, primals_110, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_90: "i64[]" = torch.ops.aten.add.Tensor(primals_111, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_18 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
        getitem_40: "f32[1, 192, 1, 1]" = var_mean_18[0]
        getitem_41: "f32[1, 192, 1, 1]" = var_mean_18[1];  var_mean_18 = None
        add_91: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05)
        rsqrt_18: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
        sub_18: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_41)
        mul_126: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        squeeze_54: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
        squeeze_55: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
        mul_127: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
        mul_128: "f32[192]" = torch.ops.aten.mul.Tensor(primals_112, 0.9)
        add_92: "f32[192]" = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
        squeeze_56: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
        mul_129: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0001594642002871);  squeeze_56 = None
        mul_130: "f32[192]" = torch.ops.aten.mul.Tensor(mul_129, 0.1);  mul_129 = None
        mul_131: "f32[192]" = torch.ops.aten.mul.Tensor(primals_113, 0.9)
        add_93: "f32[192]" = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
        unsqueeze_72: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_114, -1)
        unsqueeze_73: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        mul_132: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(mul_126, unsqueeze_73);  mul_126 = unsqueeze_73 = None
        unsqueeze_74: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_115, -1);  primals_115 = None
        unsqueeze_75: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        add_94: "f32[32, 192, 14, 14]" = torch.ops.aten.add.Tensor(mul_132, unsqueeze_75);  mul_132 = unsqueeze_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_18: "f32[32, 192, 14, 14]" = torch.ops.aten.relu.default(add_94);  add_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_19: "f32[32, 192, 14, 14]" = torch.ops.aten.convolution.default(relu_18, primals_116, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_95: "i64[]" = torch.ops.aten.add.Tensor(primals_117, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_19 = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
        getitem_42: "f32[1, 192, 1, 1]" = var_mean_19[0]
        getitem_43: "f32[1, 192, 1, 1]" = var_mean_19[1];  var_mean_19 = None
        add_96: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05)
        rsqrt_19: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        sub_19: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_43)
        mul_133: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        squeeze_57: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3])
        mul_134: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1);  squeeze_57 = None
        mul_135: "f32[192]" = torch.ops.aten.mul.Tensor(primals_118, 0.9)
        add_97: "f32[192]" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
        squeeze_59: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
        mul_136: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0001594642002871);  squeeze_59 = None
        mul_137: "f32[192]" = torch.ops.aten.mul.Tensor(mul_136, 0.1);  mul_136 = None
        mul_138: "f32[192]" = torch.ops.aten.mul.Tensor(primals_119, 0.9)
        add_98: "f32[192]" = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
        unsqueeze_76: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_120, -1)
        unsqueeze_77: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_139: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_77);  mul_133 = unsqueeze_77 = None
        unsqueeze_78: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_121, -1)
        unsqueeze_79: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_99: "f32[32, 192, 14, 14]" = torch.ops.aten.add.Tensor(mul_139, unsqueeze_79);  mul_139 = unsqueeze_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_19: "f32[32, 192, 14, 14]" = torch.ops.aten.relu.default(add_99);  add_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vovnet.py:40 in forward, code: x = torch.cat(concat_list, dim=1)
        cat_2: "f32[32, 1472, 14, 14]" = torch.ops.aten.cat.default([getitem_32, relu_15, relu_16, relu_17, relu_18, relu_19], 1);  relu_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_20: "f32[32, 768, 14, 14]" = torch.ops.aten.convolution.default(cat_2, primals_122, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_100: "i64[]" = torch.ops.aten.add.Tensor(primals_123, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_20 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
        getitem_44: "f32[1, 768, 1, 1]" = var_mean_20[0]
        getitem_45: "f32[1, 768, 1, 1]" = var_mean_20[1];  var_mean_20 = None
        add_101: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05)
        rsqrt_20: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
        sub_20: "f32[32, 768, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_45)
        mul_140: "f32[32, 768, 14, 14]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
        squeeze_60: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
        squeeze_61: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
        mul_141: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
        mul_142: "f32[768]" = torch.ops.aten.mul.Tensor(primals_124, 0.9)
        add_102: "f32[768]" = torch.ops.aten.add.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None
        squeeze_62: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
        mul_143: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0001594642002871);  squeeze_62 = None
        mul_144: "f32[768]" = torch.ops.aten.mul.Tensor(mul_143, 0.1);  mul_143 = None
        mul_145: "f32[768]" = torch.ops.aten.mul.Tensor(primals_125, 0.9)
        add_103: "f32[768]" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
        unsqueeze_80: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_126, -1)
        unsqueeze_81: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        mul_146: "f32[32, 768, 14, 14]" = torch.ops.aten.mul.Tensor(mul_140, unsqueeze_81);  mul_140 = unsqueeze_81 = None
        unsqueeze_82: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_127, -1);  primals_127 = None
        unsqueeze_83: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        add_104: "f32[32, 768, 14, 14]" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_83);  mul_146 = unsqueeze_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_20: "f32[32, 768, 14, 14]" = torch.ops.aten.relu.default(add_104);  add_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_21: "f32[32, 192, 14, 14]" = torch.ops.aten.convolution.default(relu_20, primals_128, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_105: "i64[]" = torch.ops.aten.add.Tensor(primals_129, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_21 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
        getitem_46: "f32[1, 192, 1, 1]" = var_mean_21[0]
        getitem_47: "f32[1, 192, 1, 1]" = var_mean_21[1];  var_mean_21 = None
        add_106: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
        rsqrt_21: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        sub_21: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_47)
        mul_147: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
        squeeze_63: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
        squeeze_64: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
        mul_148: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
        mul_149: "f32[192]" = torch.ops.aten.mul.Tensor(primals_130, 0.9)
        add_107: "f32[192]" = torch.ops.aten.add.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None
        squeeze_65: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
        mul_150: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0001594642002871);  squeeze_65 = None
        mul_151: "f32[192]" = torch.ops.aten.mul.Tensor(mul_150, 0.1);  mul_150 = None
        mul_152: "f32[192]" = torch.ops.aten.mul.Tensor(primals_131, 0.9)
        add_108: "f32[192]" = torch.ops.aten.add.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
        unsqueeze_84: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_132, -1)
        unsqueeze_85: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_153: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(mul_147, unsqueeze_85);  mul_147 = unsqueeze_85 = None
        unsqueeze_86: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_133, -1);  primals_133 = None
        unsqueeze_87: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_109: "f32[32, 192, 14, 14]" = torch.ops.aten.add.Tensor(mul_153, unsqueeze_87);  mul_153 = unsqueeze_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_21: "f32[32, 192, 14, 14]" = torch.ops.aten.relu.default(add_109);  add_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_22: "f32[32, 192, 14, 14]" = torch.ops.aten.convolution.default(relu_21, primals_134, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_110: "i64[]" = torch.ops.aten.add.Tensor(primals_135, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_22 = torch.ops.aten.var_mean.correction(convolution_22, [0, 2, 3], correction = 0, keepdim = True)
        getitem_48: "f32[1, 192, 1, 1]" = var_mean_22[0]
        getitem_49: "f32[1, 192, 1, 1]" = var_mean_22[1];  var_mean_22 = None
        add_111: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05)
        rsqrt_22: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
        sub_22: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_22, getitem_49)
        mul_154: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        squeeze_66: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
        squeeze_67: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
        mul_155: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
        mul_156: "f32[192]" = torch.ops.aten.mul.Tensor(primals_136, 0.9)
        add_112: "f32[192]" = torch.ops.aten.add.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
        squeeze_68: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
        mul_157: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0001594642002871);  squeeze_68 = None
        mul_158: "f32[192]" = torch.ops.aten.mul.Tensor(mul_157, 0.1);  mul_157 = None
        mul_159: "f32[192]" = torch.ops.aten.mul.Tensor(primals_137, 0.9)
        add_113: "f32[192]" = torch.ops.aten.add.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None
        unsqueeze_88: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_138, -1)
        unsqueeze_89: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        mul_160: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_89);  mul_154 = unsqueeze_89 = None
        unsqueeze_90: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_139, -1);  primals_139 = None
        unsqueeze_91: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        add_114: "f32[32, 192, 14, 14]" = torch.ops.aten.add.Tensor(mul_160, unsqueeze_91);  mul_160 = unsqueeze_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_22: "f32[32, 192, 14, 14]" = torch.ops.aten.relu.default(add_114);  add_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_23: "f32[32, 192, 14, 14]" = torch.ops.aten.convolution.default(relu_22, primals_140, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_115: "i64[]" = torch.ops.aten.add.Tensor(primals_141, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_23 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
        getitem_50: "f32[1, 192, 1, 1]" = var_mean_23[0]
        getitem_51: "f32[1, 192, 1, 1]" = var_mean_23[1];  var_mean_23 = None
        add_116: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05)
        rsqrt_23: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        sub_23: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, getitem_51)
        mul_161: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
        squeeze_69: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
        squeeze_70: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
        mul_162: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
        mul_163: "f32[192]" = torch.ops.aten.mul.Tensor(primals_142, 0.9)
        add_117: "f32[192]" = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
        squeeze_71: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
        mul_164: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0001594642002871);  squeeze_71 = None
        mul_165: "f32[192]" = torch.ops.aten.mul.Tensor(mul_164, 0.1);  mul_164 = None
        mul_166: "f32[192]" = torch.ops.aten.mul.Tensor(primals_143, 0.9)
        add_118: "f32[192]" = torch.ops.aten.add.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
        unsqueeze_92: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_144, -1)
        unsqueeze_93: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        mul_167: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_93);  mul_161 = unsqueeze_93 = None
        unsqueeze_94: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_145, -1);  primals_145 = None
        unsqueeze_95: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        add_119: "f32[32, 192, 14, 14]" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_95);  mul_167 = unsqueeze_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_23: "f32[32, 192, 14, 14]" = torch.ops.aten.relu.default(add_119);  add_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_24: "f32[32, 192, 14, 14]" = torch.ops.aten.convolution.default(relu_23, primals_146, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_120: "i64[]" = torch.ops.aten.add.Tensor(primals_147, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_24 = torch.ops.aten.var_mean.correction(convolution_24, [0, 2, 3], correction = 0, keepdim = True)
        getitem_52: "f32[1, 192, 1, 1]" = var_mean_24[0]
        getitem_53: "f32[1, 192, 1, 1]" = var_mean_24[1];  var_mean_24 = None
        add_121: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05)
        rsqrt_24: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
        sub_24: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_24, getitem_53)
        mul_168: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        squeeze_72: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
        squeeze_73: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
        mul_169: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
        mul_170: "f32[192]" = torch.ops.aten.mul.Tensor(primals_148, 0.9)
        add_122: "f32[192]" = torch.ops.aten.add.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
        squeeze_74: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
        mul_171: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0001594642002871);  squeeze_74 = None
        mul_172: "f32[192]" = torch.ops.aten.mul.Tensor(mul_171, 0.1);  mul_171 = None
        mul_173: "f32[192]" = torch.ops.aten.mul.Tensor(primals_149, 0.9)
        add_123: "f32[192]" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
        unsqueeze_96: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_150, -1)
        unsqueeze_97: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        mul_174: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(mul_168, unsqueeze_97);  mul_168 = unsqueeze_97 = None
        unsqueeze_98: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_151, -1);  primals_151 = None
        unsqueeze_99: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        add_124: "f32[32, 192, 14, 14]" = torch.ops.aten.add.Tensor(mul_174, unsqueeze_99);  mul_174 = unsqueeze_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_24: "f32[32, 192, 14, 14]" = torch.ops.aten.relu.default(add_124);  add_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_25: "f32[32, 192, 14, 14]" = torch.ops.aten.convolution.default(relu_24, primals_152, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_125: "i64[]" = torch.ops.aten.add.Tensor(primals_153, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_25 = torch.ops.aten.var_mean.correction(convolution_25, [0, 2, 3], correction = 0, keepdim = True)
        getitem_54: "f32[1, 192, 1, 1]" = var_mean_25[0]
        getitem_55: "f32[1, 192, 1, 1]" = var_mean_25[1];  var_mean_25 = None
        add_126: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05)
        rsqrt_25: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        sub_25: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_55)
        mul_175: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        squeeze_75: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3])
        mul_176: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1);  squeeze_75 = None
        mul_177: "f32[192]" = torch.ops.aten.mul.Tensor(primals_154, 0.9)
        add_127: "f32[192]" = torch.ops.aten.add.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
        squeeze_77: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
        mul_178: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0001594642002871);  squeeze_77 = None
        mul_179: "f32[192]" = torch.ops.aten.mul.Tensor(mul_178, 0.1);  mul_178 = None
        mul_180: "f32[192]" = torch.ops.aten.mul.Tensor(primals_155, 0.9)
        add_128: "f32[192]" = torch.ops.aten.add.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
        unsqueeze_100: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_156, -1)
        unsqueeze_101: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_181: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_101);  mul_175 = unsqueeze_101 = None
        unsqueeze_102: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_157, -1)
        unsqueeze_103: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_129: "f32[32, 192, 14, 14]" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_103);  mul_181 = unsqueeze_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_25: "f32[32, 192, 14, 14]" = torch.ops.aten.relu.default(add_129);  add_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vovnet.py:40 in forward, code: x = torch.cat(concat_list, dim=1)
        cat_3: "f32[32, 1728, 14, 14]" = torch.ops.aten.cat.default([relu_20, relu_21, relu_22, relu_23, relu_24, relu_25], 1);  relu_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_26: "f32[32, 768, 14, 14]" = torch.ops.aten.convolution.default(cat_3, primals_158, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_130: "i64[]" = torch.ops.aten.add.Tensor(primals_159, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_26 = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
        getitem_56: "f32[1, 768, 1, 1]" = var_mean_26[0]
        getitem_57: "f32[1, 768, 1, 1]" = var_mean_26[1];  var_mean_26 = None
        add_131: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05)
        rsqrt_26: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
        sub_26: "f32[32, 768, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_57)
        mul_182: "f32[32, 768, 14, 14]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
        squeeze_78: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3])
        mul_183: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1);  squeeze_78 = None
        mul_184: "f32[768]" = torch.ops.aten.mul.Tensor(primals_160, 0.9)
        add_132: "f32[768]" = torch.ops.aten.add.Tensor(mul_183, mul_184);  mul_183 = mul_184 = None
        squeeze_80: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
        mul_185: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0001594642002871);  squeeze_80 = None
        mul_186: "f32[768]" = torch.ops.aten.mul.Tensor(mul_185, 0.1);  mul_185 = None
        mul_187: "f32[768]" = torch.ops.aten.mul.Tensor(primals_161, 0.9)
        add_133: "f32[768]" = torch.ops.aten.add.Tensor(mul_186, mul_187);  mul_186 = mul_187 = None
        unsqueeze_104: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_162, -1)
        unsqueeze_105: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        mul_188: "f32[32, 768, 14, 14]" = torch.ops.aten.mul.Tensor(mul_182, unsqueeze_105);  mul_182 = unsqueeze_105 = None
        unsqueeze_106: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_163, -1)
        unsqueeze_107: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
        add_134: "f32[32, 768, 14, 14]" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_107);  mul_188 = unsqueeze_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_26: "f32[32, 768, 14, 14]" = torch.ops.aten.relu.default(add_134);  add_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vovnet.py:161 in forward, code: x = self.pool(x)
        _low_memory_max_pool_with_offsets_2 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_26, [3, 3], [2, 2], [0, 0], [1, 1], True);  relu_26 = None
        getitem_58: "f32[32, 768, 7, 7]" = _low_memory_max_pool_with_offsets_2[0]
        getitem_59: "i8[32, 768, 7, 7]" = _low_memory_max_pool_with_offsets_2[1];  _low_memory_max_pool_with_offsets_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_27: "f32[32, 224, 7, 7]" = torch.ops.aten.convolution.default(getitem_58, primals_164, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_135: "i64[]" = torch.ops.aten.add.Tensor(primals_165, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_27 = torch.ops.aten.var_mean.correction(convolution_27, [0, 2, 3], correction = 0, keepdim = True)
        getitem_60: "f32[1, 224, 1, 1]" = var_mean_27[0]
        getitem_61: "f32[1, 224, 1, 1]" = var_mean_27[1];  var_mean_27 = None
        add_136: "f32[1, 224, 1, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05)
        rsqrt_27: "f32[1, 224, 1, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
        sub_27: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_27, getitem_61)
        mul_189: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
        squeeze_81: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
        squeeze_82: "f32[224]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
        mul_190: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
        mul_191: "f32[224]" = torch.ops.aten.mul.Tensor(primals_166, 0.9)
        add_137: "f32[224]" = torch.ops.aten.add.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
        squeeze_83: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
        mul_192: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0006381620931717);  squeeze_83 = None
        mul_193: "f32[224]" = torch.ops.aten.mul.Tensor(mul_192, 0.1);  mul_192 = None
        mul_194: "f32[224]" = torch.ops.aten.mul.Tensor(primals_167, 0.9)
        add_138: "f32[224]" = torch.ops.aten.add.Tensor(mul_193, mul_194);  mul_193 = mul_194 = None
        unsqueeze_108: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_168, -1)
        unsqueeze_109: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
        mul_195: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(mul_189, unsqueeze_109);  mul_189 = unsqueeze_109 = None
        unsqueeze_110: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_169, -1);  primals_169 = None
        unsqueeze_111: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
        add_139: "f32[32, 224, 7, 7]" = torch.ops.aten.add.Tensor(mul_195, unsqueeze_111);  mul_195 = unsqueeze_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_27: "f32[32, 224, 7, 7]" = torch.ops.aten.relu.default(add_139);  add_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_28: "f32[32, 224, 7, 7]" = torch.ops.aten.convolution.default(relu_27, primals_170, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_140: "i64[]" = torch.ops.aten.add.Tensor(primals_171, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_28 = torch.ops.aten.var_mean.correction(convolution_28, [0, 2, 3], correction = 0, keepdim = True)
        getitem_62: "f32[1, 224, 1, 1]" = var_mean_28[0]
        getitem_63: "f32[1, 224, 1, 1]" = var_mean_28[1];  var_mean_28 = None
        add_141: "f32[1, 224, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
        rsqrt_28: "f32[1, 224, 1, 1]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
        sub_28: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_28, getitem_63)
        mul_196: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
        squeeze_84: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
        squeeze_85: "f32[224]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
        mul_197: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
        mul_198: "f32[224]" = torch.ops.aten.mul.Tensor(primals_172, 0.9)
        add_142: "f32[224]" = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
        squeeze_86: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
        mul_199: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0006381620931717);  squeeze_86 = None
        mul_200: "f32[224]" = torch.ops.aten.mul.Tensor(mul_199, 0.1);  mul_199 = None
        mul_201: "f32[224]" = torch.ops.aten.mul.Tensor(primals_173, 0.9)
        add_143: "f32[224]" = torch.ops.aten.add.Tensor(mul_200, mul_201);  mul_200 = mul_201 = None
        unsqueeze_112: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_174, -1)
        unsqueeze_113: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        mul_202: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_113);  mul_196 = unsqueeze_113 = None
        unsqueeze_114: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_175, -1);  primals_175 = None
        unsqueeze_115: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        add_144: "f32[32, 224, 7, 7]" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_115);  mul_202 = unsqueeze_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_28: "f32[32, 224, 7, 7]" = torch.ops.aten.relu.default(add_144);  add_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_29: "f32[32, 224, 7, 7]" = torch.ops.aten.convolution.default(relu_28, primals_176, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_145: "i64[]" = torch.ops.aten.add.Tensor(primals_177, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_29 = torch.ops.aten.var_mean.correction(convolution_29, [0, 2, 3], correction = 0, keepdim = True)
        getitem_64: "f32[1, 224, 1, 1]" = var_mean_29[0]
        getitem_65: "f32[1, 224, 1, 1]" = var_mean_29[1];  var_mean_29 = None
        add_146: "f32[1, 224, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
        rsqrt_29: "f32[1, 224, 1, 1]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
        sub_29: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_29, getitem_65)
        mul_203: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
        squeeze_87: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
        squeeze_88: "f32[224]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
        mul_204: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
        mul_205: "f32[224]" = torch.ops.aten.mul.Tensor(primals_178, 0.9)
        add_147: "f32[224]" = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None
        squeeze_89: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
        mul_206: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0006381620931717);  squeeze_89 = None
        mul_207: "f32[224]" = torch.ops.aten.mul.Tensor(mul_206, 0.1);  mul_206 = None
        mul_208: "f32[224]" = torch.ops.aten.mul.Tensor(primals_179, 0.9)
        add_148: "f32[224]" = torch.ops.aten.add.Tensor(mul_207, mul_208);  mul_207 = mul_208 = None
        unsqueeze_116: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_180, -1)
        unsqueeze_117: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        mul_209: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_117);  mul_203 = unsqueeze_117 = None
        unsqueeze_118: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_181, -1);  primals_181 = None
        unsqueeze_119: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
        add_149: "f32[32, 224, 7, 7]" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_119);  mul_209 = unsqueeze_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_29: "f32[32, 224, 7, 7]" = torch.ops.aten.relu.default(add_149);  add_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_30: "f32[32, 224, 7, 7]" = torch.ops.aten.convolution.default(relu_29, primals_182, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_150: "i64[]" = torch.ops.aten.add.Tensor(primals_183, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_30 = torch.ops.aten.var_mean.correction(convolution_30, [0, 2, 3], correction = 0, keepdim = True)
        getitem_66: "f32[1, 224, 1, 1]" = var_mean_30[0]
        getitem_67: "f32[1, 224, 1, 1]" = var_mean_30[1];  var_mean_30 = None
        add_151: "f32[1, 224, 1, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05)
        rsqrt_30: "f32[1, 224, 1, 1]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
        sub_30: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_30, getitem_67)
        mul_210: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
        squeeze_90: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
        squeeze_91: "f32[224]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
        mul_211: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
        mul_212: "f32[224]" = torch.ops.aten.mul.Tensor(primals_184, 0.9)
        add_152: "f32[224]" = torch.ops.aten.add.Tensor(mul_211, mul_212);  mul_211 = mul_212 = None
        squeeze_92: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
        mul_213: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0006381620931717);  squeeze_92 = None
        mul_214: "f32[224]" = torch.ops.aten.mul.Tensor(mul_213, 0.1);  mul_213 = None
        mul_215: "f32[224]" = torch.ops.aten.mul.Tensor(primals_185, 0.9)
        add_153: "f32[224]" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
        unsqueeze_120: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_186, -1)
        unsqueeze_121: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        mul_216: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_121);  mul_210 = unsqueeze_121 = None
        unsqueeze_122: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_187, -1);  primals_187 = None
        unsqueeze_123: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        add_154: "f32[32, 224, 7, 7]" = torch.ops.aten.add.Tensor(mul_216, unsqueeze_123);  mul_216 = unsqueeze_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_30: "f32[32, 224, 7, 7]" = torch.ops.aten.relu.default(add_154);  add_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_31: "f32[32, 224, 7, 7]" = torch.ops.aten.convolution.default(relu_30, primals_188, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_155: "i64[]" = torch.ops.aten.add.Tensor(primals_189, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_31 = torch.ops.aten.var_mean.correction(convolution_31, [0, 2, 3], correction = 0, keepdim = True)
        getitem_68: "f32[1, 224, 1, 1]" = var_mean_31[0]
        getitem_69: "f32[1, 224, 1, 1]" = var_mean_31[1];  var_mean_31 = None
        add_156: "f32[1, 224, 1, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05)
        rsqrt_31: "f32[1, 224, 1, 1]" = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
        sub_31: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_31, getitem_69)
        mul_217: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
        squeeze_93: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3])
        mul_218: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1);  squeeze_93 = None
        mul_219: "f32[224]" = torch.ops.aten.mul.Tensor(primals_190, 0.9)
        add_157: "f32[224]" = torch.ops.aten.add.Tensor(mul_218, mul_219);  mul_218 = mul_219 = None
        squeeze_95: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
        mul_220: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0006381620931717);  squeeze_95 = None
        mul_221: "f32[224]" = torch.ops.aten.mul.Tensor(mul_220, 0.1);  mul_220 = None
        mul_222: "f32[224]" = torch.ops.aten.mul.Tensor(primals_191, 0.9)
        add_158: "f32[224]" = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
        unsqueeze_124: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_192, -1)
        unsqueeze_125: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        mul_223: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_125);  mul_217 = unsqueeze_125 = None
        unsqueeze_126: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_193, -1)
        unsqueeze_127: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        add_159: "f32[32, 224, 7, 7]" = torch.ops.aten.add.Tensor(mul_223, unsqueeze_127);  mul_223 = unsqueeze_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_31: "f32[32, 224, 7, 7]" = torch.ops.aten.relu.default(add_159);  add_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vovnet.py:40 in forward, code: x = torch.cat(concat_list, dim=1)
        cat_4: "f32[32, 1888, 7, 7]" = torch.ops.aten.cat.default([getitem_58, relu_27, relu_28, relu_29, relu_30, relu_31], 1);  relu_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_32: "f32[32, 1024, 7, 7]" = torch.ops.aten.convolution.default(cat_4, primals_194, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_160: "i64[]" = torch.ops.aten.add.Tensor(primals_195, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_32 = torch.ops.aten.var_mean.correction(convolution_32, [0, 2, 3], correction = 0, keepdim = True)
        getitem_70: "f32[1, 1024, 1, 1]" = var_mean_32[0]
        getitem_71: "f32[1, 1024, 1, 1]" = var_mean_32[1];  var_mean_32 = None
        add_161: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05)
        rsqrt_32: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
        sub_32: "f32[32, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_32, getitem_71)
        mul_224: "f32[32, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
        squeeze_96: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
        squeeze_97: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
        mul_225: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
        mul_226: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_196, 0.9)
        add_162: "f32[1024]" = torch.ops.aten.add.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
        squeeze_98: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
        mul_227: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0006381620931717);  squeeze_98 = None
        mul_228: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_227, 0.1);  mul_227 = None
        mul_229: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_197, 0.9)
        add_163: "f32[1024]" = torch.ops.aten.add.Tensor(mul_228, mul_229);  mul_228 = mul_229 = None
        unsqueeze_128: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_198, -1)
        unsqueeze_129: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
        mul_230: "f32[32, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(mul_224, unsqueeze_129);  mul_224 = unsqueeze_129 = None
        unsqueeze_130: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_199, -1);  primals_199 = None
        unsqueeze_131: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
        add_164: "f32[32, 1024, 7, 7]" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_131);  mul_230 = unsqueeze_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_32: "f32[32, 1024, 7, 7]" = torch.ops.aten.relu.default(add_164);  add_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_33: "f32[32, 224, 7, 7]" = torch.ops.aten.convolution.default(relu_32, primals_200, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_165: "i64[]" = torch.ops.aten.add.Tensor(primals_201, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_33 = torch.ops.aten.var_mean.correction(convolution_33, [0, 2, 3], correction = 0, keepdim = True)
        getitem_72: "f32[1, 224, 1, 1]" = var_mean_33[0]
        getitem_73: "f32[1, 224, 1, 1]" = var_mean_33[1];  var_mean_33 = None
        add_166: "f32[1, 224, 1, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05)
        rsqrt_33: "f32[1, 224, 1, 1]" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
        sub_33: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_33, getitem_73)
        mul_231: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
        squeeze_99: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
        squeeze_100: "f32[224]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
        mul_232: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
        mul_233: "f32[224]" = torch.ops.aten.mul.Tensor(primals_202, 0.9)
        add_167: "f32[224]" = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
        squeeze_101: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
        mul_234: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0006381620931717);  squeeze_101 = None
        mul_235: "f32[224]" = torch.ops.aten.mul.Tensor(mul_234, 0.1);  mul_234 = None
        mul_236: "f32[224]" = torch.ops.aten.mul.Tensor(primals_203, 0.9)
        add_168: "f32[224]" = torch.ops.aten.add.Tensor(mul_235, mul_236);  mul_235 = mul_236 = None
        unsqueeze_132: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_204, -1)
        unsqueeze_133: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
        mul_237: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(mul_231, unsqueeze_133);  mul_231 = unsqueeze_133 = None
        unsqueeze_134: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_205, -1);  primals_205 = None
        unsqueeze_135: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
        add_169: "f32[32, 224, 7, 7]" = torch.ops.aten.add.Tensor(mul_237, unsqueeze_135);  mul_237 = unsqueeze_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_33: "f32[32, 224, 7, 7]" = torch.ops.aten.relu.default(add_169);  add_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_34: "f32[32, 224, 7, 7]" = torch.ops.aten.convolution.default(relu_33, primals_206, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_170: "i64[]" = torch.ops.aten.add.Tensor(primals_207, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_34 = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
        getitem_74: "f32[1, 224, 1, 1]" = var_mean_34[0]
        getitem_75: "f32[1, 224, 1, 1]" = var_mean_34[1];  var_mean_34 = None
        add_171: "f32[1, 224, 1, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05)
        rsqrt_34: "f32[1, 224, 1, 1]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
        sub_34: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_75)
        mul_238: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
        squeeze_102: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
        squeeze_103: "f32[224]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
        mul_239: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
        mul_240: "f32[224]" = torch.ops.aten.mul.Tensor(primals_208, 0.9)
        add_172: "f32[224]" = torch.ops.aten.add.Tensor(mul_239, mul_240);  mul_239 = mul_240 = None
        squeeze_104: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
        mul_241: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0006381620931717);  squeeze_104 = None
        mul_242: "f32[224]" = torch.ops.aten.mul.Tensor(mul_241, 0.1);  mul_241 = None
        mul_243: "f32[224]" = torch.ops.aten.mul.Tensor(primals_209, 0.9)
        add_173: "f32[224]" = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
        unsqueeze_136: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_210, -1)
        unsqueeze_137: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
        mul_244: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_137);  mul_238 = unsqueeze_137 = None
        unsqueeze_138: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_211, -1);  primals_211 = None
        unsqueeze_139: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
        add_174: "f32[32, 224, 7, 7]" = torch.ops.aten.add.Tensor(mul_244, unsqueeze_139);  mul_244 = unsqueeze_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_34: "f32[32, 224, 7, 7]" = torch.ops.aten.relu.default(add_174);  add_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_35: "f32[32, 224, 7, 7]" = torch.ops.aten.convolution.default(relu_34, primals_212, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_175: "i64[]" = torch.ops.aten.add.Tensor(primals_213, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_35 = torch.ops.aten.var_mean.correction(convolution_35, [0, 2, 3], correction = 0, keepdim = True)
        getitem_76: "f32[1, 224, 1, 1]" = var_mean_35[0]
        getitem_77: "f32[1, 224, 1, 1]" = var_mean_35[1];  var_mean_35 = None
        add_176: "f32[1, 224, 1, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05)
        rsqrt_35: "f32[1, 224, 1, 1]" = torch.ops.aten.rsqrt.default(add_176);  add_176 = None
        sub_35: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_35, getitem_77)
        mul_245: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
        squeeze_105: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
        squeeze_106: "f32[224]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
        mul_246: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
        mul_247: "f32[224]" = torch.ops.aten.mul.Tensor(primals_214, 0.9)
        add_177: "f32[224]" = torch.ops.aten.add.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
        squeeze_107: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
        mul_248: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_107, 1.0006381620931717);  squeeze_107 = None
        mul_249: "f32[224]" = torch.ops.aten.mul.Tensor(mul_248, 0.1);  mul_248 = None
        mul_250: "f32[224]" = torch.ops.aten.mul.Tensor(primals_215, 0.9)
        add_178: "f32[224]" = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
        unsqueeze_140: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_216, -1)
        unsqueeze_141: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
        mul_251: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_141);  mul_245 = unsqueeze_141 = None
        unsqueeze_142: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_217, -1);  primals_217 = None
        unsqueeze_143: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
        add_179: "f32[32, 224, 7, 7]" = torch.ops.aten.add.Tensor(mul_251, unsqueeze_143);  mul_251 = unsqueeze_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_35: "f32[32, 224, 7, 7]" = torch.ops.aten.relu.default(add_179);  add_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_36: "f32[32, 224, 7, 7]" = torch.ops.aten.convolution.default(relu_35, primals_218, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_180: "i64[]" = torch.ops.aten.add.Tensor(primals_219, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_36 = torch.ops.aten.var_mean.correction(convolution_36, [0, 2, 3], correction = 0, keepdim = True)
        getitem_78: "f32[1, 224, 1, 1]" = var_mean_36[0]
        getitem_79: "f32[1, 224, 1, 1]" = var_mean_36[1];  var_mean_36 = None
        add_181: "f32[1, 224, 1, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05)
        rsqrt_36: "f32[1, 224, 1, 1]" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
        sub_36: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_36, getitem_79)
        mul_252: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
        squeeze_108: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
        squeeze_109: "f32[224]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
        mul_253: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
        mul_254: "f32[224]" = torch.ops.aten.mul.Tensor(primals_220, 0.9)
        add_182: "f32[224]" = torch.ops.aten.add.Tensor(mul_253, mul_254);  mul_253 = mul_254 = None
        squeeze_110: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
        mul_255: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0006381620931717);  squeeze_110 = None
        mul_256: "f32[224]" = torch.ops.aten.mul.Tensor(mul_255, 0.1);  mul_255 = None
        mul_257: "f32[224]" = torch.ops.aten.mul.Tensor(primals_221, 0.9)
        add_183: "f32[224]" = torch.ops.aten.add.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
        unsqueeze_144: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_222, -1)
        unsqueeze_145: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
        mul_258: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(mul_252, unsqueeze_145);  mul_252 = unsqueeze_145 = None
        unsqueeze_146: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_223, -1);  primals_223 = None
        unsqueeze_147: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
        add_184: "f32[32, 224, 7, 7]" = torch.ops.aten.add.Tensor(mul_258, unsqueeze_147);  mul_258 = unsqueeze_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_36: "f32[32, 224, 7, 7]" = torch.ops.aten.relu.default(add_184);  add_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_37: "f32[32, 224, 7, 7]" = torch.ops.aten.convolution.default(relu_36, primals_224, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_185: "i64[]" = torch.ops.aten.add.Tensor(primals_225, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_37 = torch.ops.aten.var_mean.correction(convolution_37, [0, 2, 3], correction = 0, keepdim = True)
        getitem_80: "f32[1, 224, 1, 1]" = var_mean_37[0]
        getitem_81: "f32[1, 224, 1, 1]" = var_mean_37[1];  var_mean_37 = None
        add_186: "f32[1, 224, 1, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05)
        rsqrt_37: "f32[1, 224, 1, 1]" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
        sub_37: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_37, getitem_81)
        mul_259: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
        squeeze_111: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3])
        mul_260: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1);  squeeze_111 = None
        mul_261: "f32[224]" = torch.ops.aten.mul.Tensor(primals_226, 0.9)
        add_187: "f32[224]" = torch.ops.aten.add.Tensor(mul_260, mul_261);  mul_260 = mul_261 = None
        squeeze_113: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
        mul_262: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0006381620931717);  squeeze_113 = None
        mul_263: "f32[224]" = torch.ops.aten.mul.Tensor(mul_262, 0.1);  mul_262 = None
        mul_264: "f32[224]" = torch.ops.aten.mul.Tensor(primals_227, 0.9)
        add_188: "f32[224]" = torch.ops.aten.add.Tensor(mul_263, mul_264);  mul_263 = mul_264 = None
        unsqueeze_148: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_228, -1)
        unsqueeze_149: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
        mul_265: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_149);  mul_259 = unsqueeze_149 = None
        unsqueeze_150: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_229, -1)
        unsqueeze_151: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
        add_189: "f32[32, 224, 7, 7]" = torch.ops.aten.add.Tensor(mul_265, unsqueeze_151);  mul_265 = unsqueeze_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_37: "f32[32, 224, 7, 7]" = torch.ops.aten.relu.default(add_189);  add_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vovnet.py:40 in forward, code: x = torch.cat(concat_list, dim=1)
        cat_5: "f32[32, 2144, 7, 7]" = torch.ops.aten.cat.default([relu_32, relu_33, relu_34, relu_35, relu_36, relu_37], 1);  relu_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_38: "f32[32, 1024, 7, 7]" = torch.ops.aten.convolution.default(cat_5, primals_230, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_190: "i64[]" = torch.ops.aten.add.Tensor(primals_231, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_38 = torch.ops.aten.var_mean.correction(convolution_38, [0, 2, 3], correction = 0, keepdim = True)
        getitem_82: "f32[1, 1024, 1, 1]" = var_mean_38[0]
        getitem_83: "f32[1, 1024, 1, 1]" = var_mean_38[1];  var_mean_38 = None
        add_191: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05)
        rsqrt_38: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
        sub_38: "f32[32, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_38, getitem_83)
        mul_266: "f32[32, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
        squeeze_114: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3])
        mul_267: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1);  squeeze_114 = None
        mul_268: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_232, 0.9)
        add_192: "f32[1024]" = torch.ops.aten.add.Tensor(mul_267, mul_268);  mul_267 = mul_268 = None
        squeeze_116: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
        mul_269: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0006381620931717);  squeeze_116 = None
        mul_270: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_269, 0.1);  mul_269 = None
        mul_271: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_233, 0.9)
        add_193: "f32[1024]" = torch.ops.aten.add.Tensor(mul_270, mul_271);  mul_270 = mul_271 = None
        unsqueeze_152: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_234, -1)
        unsqueeze_153: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
        mul_272: "f32[32, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(mul_266, unsqueeze_153);  mul_266 = unsqueeze_153 = None
        unsqueeze_154: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_235, -1)
        unsqueeze_155: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
        add_194: "f32[32, 1024, 7, 7]" = torch.ops.aten.add.Tensor(mul_272, unsqueeze_155);  mul_272 = unsqueeze_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_38: "f32[32, 1024, 7, 7]" = torch.ops.aten.relu.default(add_194);  add_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean: "f32[32, 1024, 1, 1]" = torch.ops.aten.mean.dim(relu_38, [-1, -2], True);  relu_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view: "f32[32, 1024]" = torch.ops.aten.reshape.default(mean, [32, 1024]);  mean = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:141 in forward, code: x = self.fc(x)
        permute: "f32[1024, 1000]" = torch.ops.aten.permute.default(primals_236, [1, 0])
        addmm: "f32[32, 1000]" = torch.ops.aten.addmm.default(primals_237, view, permute);  primals_237 = permute = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_180: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
        unsqueeze_181: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, 2);  unsqueeze_180 = None
        unsqueeze_182: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_181, 3);  unsqueeze_181 = None
        unsqueeze_192: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
        unsqueeze_193: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, 2);  unsqueeze_192 = None
        unsqueeze_194: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_193, 3);  unsqueeze_193 = None
        unsqueeze_204: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
        unsqueeze_205: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, 2);  unsqueeze_204 = None
        unsqueeze_206: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 3);  unsqueeze_205 = None
        unsqueeze_216: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
        unsqueeze_217: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, 2);  unsqueeze_216 = None
        unsqueeze_218: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 3);  unsqueeze_217 = None
        unsqueeze_228: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
        unsqueeze_229: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, 2);  unsqueeze_228 = None
        unsqueeze_230: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 3);  unsqueeze_229 = None
        unsqueeze_252: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
        unsqueeze_253: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, 2);  unsqueeze_252 = None
        unsqueeze_254: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 3);  unsqueeze_253 = None
        unsqueeze_264: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
        unsqueeze_265: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 2);  unsqueeze_264 = None
        unsqueeze_266: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 3);  unsqueeze_265 = None
        unsqueeze_276: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
        unsqueeze_277: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 2);  unsqueeze_276 = None
        unsqueeze_278: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 3);  unsqueeze_277 = None
        unsqueeze_288: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
        unsqueeze_289: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 2);  unsqueeze_288 = None
        unsqueeze_290: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 3);  unsqueeze_289 = None
        unsqueeze_324: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
        unsqueeze_325: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, 2);  unsqueeze_324 = None
        unsqueeze_326: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 3);  unsqueeze_325 = None
        unsqueeze_336: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
        unsqueeze_337: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 2);  unsqueeze_336 = None
        unsqueeze_338: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 3);  unsqueeze_337 = None
        unsqueeze_348: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
        unsqueeze_349: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 2);  unsqueeze_348 = None
        unsqueeze_350: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 3);  unsqueeze_349 = None
        unsqueeze_360: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
        unsqueeze_361: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, 2);  unsqueeze_360 = None
        unsqueeze_362: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 3);  unsqueeze_361 = None
        unsqueeze_372: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
        unsqueeze_373: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, 2);  unsqueeze_372 = None
        unsqueeze_374: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 3);  unsqueeze_373 = None
        unsqueeze_396: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
        unsqueeze_397: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, 2);  unsqueeze_396 = None
        unsqueeze_398: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 3);  unsqueeze_397 = None
        unsqueeze_408: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
        unsqueeze_409: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, 2);  unsqueeze_408 = None
        unsqueeze_410: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 3);  unsqueeze_409 = None
        unsqueeze_420: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
        unsqueeze_421: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 2);  unsqueeze_420 = None
        unsqueeze_422: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 3);  unsqueeze_421 = None
        unsqueeze_432: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
        unsqueeze_433: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, 2);  unsqueeze_432 = None
        unsqueeze_434: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 3);  unsqueeze_433 = None
        unsqueeze_468: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
        unsqueeze_469: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, 2);  unsqueeze_468 = None
        unsqueeze_470: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 3);  unsqueeze_469 = None
        unsqueeze_480: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
        unsqueeze_481: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, 2);  unsqueeze_480 = None
        unsqueeze_482: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 3);  unsqueeze_481 = None
        unsqueeze_492: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
        unsqueeze_493: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, 2);  unsqueeze_492 = None
        unsqueeze_494: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 3);  unsqueeze_493 = None
        unsqueeze_504: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
        unsqueeze_505: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, 2);  unsqueeze_504 = None
        unsqueeze_506: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 3);  unsqueeze_505 = None
        unsqueeze_540: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
        unsqueeze_541: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, 2);  unsqueeze_540 = None
        unsqueeze_542: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 3);  unsqueeze_541 = None
        unsqueeze_552: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
        unsqueeze_553: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, 2);  unsqueeze_552 = None
        unsqueeze_554: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 3);  unsqueeze_553 = None
        unsqueeze_564: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
        unsqueeze_565: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, 2);  unsqueeze_564 = None
        unsqueeze_566: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 3);  unsqueeze_565 = None
        unsqueeze_576: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
        unsqueeze_577: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, 2);  unsqueeze_576 = None
        unsqueeze_578: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_577, 3);  unsqueeze_577 = None
        unsqueeze_588: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
        unsqueeze_589: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, 2);  unsqueeze_588 = None
        unsqueeze_590: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 3);  unsqueeze_589 = None
        unsqueeze_600: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
        unsqueeze_601: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, 2);  unsqueeze_600 = None
        unsqueeze_602: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 3);  unsqueeze_601 = None
        unsqueeze_612: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_613: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, 2);  unsqueeze_612 = None
        unsqueeze_614: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 3);  unsqueeze_613 = None

        # No stacktrace found for following nodes
        copy_: "i64[]" = torch.ops.aten.copy_.default(primals_3, add);  primals_3 = add = copy_ = None
        copy__1: "f32[64]" = torch.ops.aten.copy_.default(primals_4, add_2);  primals_4 = add_2 = copy__1 = None
        copy__2: "f32[64]" = torch.ops.aten.copy_.default(primals_5, add_3);  primals_5 = add_3 = copy__2 = None
        copy__3: "i64[]" = torch.ops.aten.copy_.default(primals_9, add_5);  primals_9 = add_5 = copy__3 = None
        copy__4: "f32[64]" = torch.ops.aten.copy_.default(primals_10, add_7);  primals_10 = add_7 = copy__4 = None
        copy__5: "f32[64]" = torch.ops.aten.copy_.default(primals_11, add_8);  primals_11 = add_8 = copy__5 = None
        copy__6: "i64[]" = torch.ops.aten.copy_.default(primals_15, add_10);  primals_15 = add_10 = copy__6 = None
        copy__7: "f32[128]" = torch.ops.aten.copy_.default(primals_16, add_12);  primals_16 = add_12 = copy__7 = None
        copy__8: "f32[128]" = torch.ops.aten.copy_.default(primals_17, add_13);  primals_17 = add_13 = copy__8 = None
        copy__9: "i64[]" = torch.ops.aten.copy_.default(primals_21, add_15);  primals_21 = add_15 = copy__9 = None
        copy__10: "f32[128]" = torch.ops.aten.copy_.default(primals_22, add_17);  primals_22 = add_17 = copy__10 = None
        copy__11: "f32[128]" = torch.ops.aten.copy_.default(primals_23, add_18);  primals_23 = add_18 = copy__11 = None
        copy__12: "i64[]" = torch.ops.aten.copy_.default(primals_27, add_20);  primals_27 = add_20 = copy__12 = None
        copy__13: "f32[128]" = torch.ops.aten.copy_.default(primals_28, add_22);  primals_28 = add_22 = copy__13 = None
        copy__14: "f32[128]" = torch.ops.aten.copy_.default(primals_29, add_23);  primals_29 = add_23 = copy__14 = None
        copy__15: "i64[]" = torch.ops.aten.copy_.default(primals_33, add_25);  primals_33 = add_25 = copy__15 = None
        copy__16: "f32[128]" = torch.ops.aten.copy_.default(primals_34, add_27);  primals_34 = add_27 = copy__16 = None
        copy__17: "f32[128]" = torch.ops.aten.copy_.default(primals_35, add_28);  primals_35 = add_28 = copy__17 = None
        copy__18: "i64[]" = torch.ops.aten.copy_.default(primals_39, add_30);  primals_39 = add_30 = copy__18 = None
        copy__19: "f32[128]" = torch.ops.aten.copy_.default(primals_40, add_32);  primals_40 = add_32 = copy__19 = None
        copy__20: "f32[128]" = torch.ops.aten.copy_.default(primals_41, add_33);  primals_41 = add_33 = copy__20 = None
        copy__21: "i64[]" = torch.ops.aten.copy_.default(primals_45, add_35);  primals_45 = add_35 = copy__21 = None
        copy__22: "f32[128]" = torch.ops.aten.copy_.default(primals_46, add_37);  primals_46 = add_37 = copy__22 = None
        copy__23: "f32[128]" = torch.ops.aten.copy_.default(primals_47, add_38);  primals_47 = add_38 = copy__23 = None
        copy__24: "i64[]" = torch.ops.aten.copy_.default(primals_51, add_40);  primals_51 = add_40 = copy__24 = None
        copy__25: "f32[256]" = torch.ops.aten.copy_.default(primals_52, add_42);  primals_52 = add_42 = copy__25 = None
        copy__26: "f32[256]" = torch.ops.aten.copy_.default(primals_53, add_43);  primals_53 = add_43 = copy__26 = None
        copy__27: "i64[]" = torch.ops.aten.copy_.default(primals_57, add_45);  primals_57 = add_45 = copy__27 = None
        copy__28: "f32[160]" = torch.ops.aten.copy_.default(primals_58, add_47);  primals_58 = add_47 = copy__28 = None
        copy__29: "f32[160]" = torch.ops.aten.copy_.default(primals_59, add_48);  primals_59 = add_48 = copy__29 = None
        copy__30: "i64[]" = torch.ops.aten.copy_.default(primals_63, add_50);  primals_63 = add_50 = copy__30 = None
        copy__31: "f32[160]" = torch.ops.aten.copy_.default(primals_64, add_52);  primals_64 = add_52 = copy__31 = None
        copy__32: "f32[160]" = torch.ops.aten.copy_.default(primals_65, add_53);  primals_65 = add_53 = copy__32 = None
        copy__33: "i64[]" = torch.ops.aten.copy_.default(primals_69, add_55);  primals_69 = add_55 = copy__33 = None
        copy__34: "f32[160]" = torch.ops.aten.copy_.default(primals_70, add_57);  primals_70 = add_57 = copy__34 = None
        copy__35: "f32[160]" = torch.ops.aten.copy_.default(primals_71, add_58);  primals_71 = add_58 = copy__35 = None
        copy__36: "i64[]" = torch.ops.aten.copy_.default(primals_75, add_60);  primals_75 = add_60 = copy__36 = None
        copy__37: "f32[160]" = torch.ops.aten.copy_.default(primals_76, add_62);  primals_76 = add_62 = copy__37 = None
        copy__38: "f32[160]" = torch.ops.aten.copy_.default(primals_77, add_63);  primals_77 = add_63 = copy__38 = None
        copy__39: "i64[]" = torch.ops.aten.copy_.default(primals_81, add_65);  primals_81 = add_65 = copy__39 = None
        copy__40: "f32[160]" = torch.ops.aten.copy_.default(primals_82, add_67);  primals_82 = add_67 = copy__40 = None
        copy__41: "f32[160]" = torch.ops.aten.copy_.default(primals_83, add_68);  primals_83 = add_68 = copy__41 = None
        copy__42: "i64[]" = torch.ops.aten.copy_.default(primals_87, add_70);  primals_87 = add_70 = copy__42 = None
        copy__43: "f32[512]" = torch.ops.aten.copy_.default(primals_88, add_72);  primals_88 = add_72 = copy__43 = None
        copy__44: "f32[512]" = torch.ops.aten.copy_.default(primals_89, add_73);  primals_89 = add_73 = copy__44 = None
        copy__45: "i64[]" = torch.ops.aten.copy_.default(primals_93, add_75);  primals_93 = add_75 = copy__45 = None
        copy__46: "f32[192]" = torch.ops.aten.copy_.default(primals_94, add_77);  primals_94 = add_77 = copy__46 = None
        copy__47: "f32[192]" = torch.ops.aten.copy_.default(primals_95, add_78);  primals_95 = add_78 = copy__47 = None
        copy__48: "i64[]" = torch.ops.aten.copy_.default(primals_99, add_80);  primals_99 = add_80 = copy__48 = None
        copy__49: "f32[192]" = torch.ops.aten.copy_.default(primals_100, add_82);  primals_100 = add_82 = copy__49 = None
        copy__50: "f32[192]" = torch.ops.aten.copy_.default(primals_101, add_83);  primals_101 = add_83 = copy__50 = None
        copy__51: "i64[]" = torch.ops.aten.copy_.default(primals_105, add_85);  primals_105 = add_85 = copy__51 = None
        copy__52: "f32[192]" = torch.ops.aten.copy_.default(primals_106, add_87);  primals_106 = add_87 = copy__52 = None
        copy__53: "f32[192]" = torch.ops.aten.copy_.default(primals_107, add_88);  primals_107 = add_88 = copy__53 = None
        copy__54: "i64[]" = torch.ops.aten.copy_.default(primals_111, add_90);  primals_111 = add_90 = copy__54 = None
        copy__55: "f32[192]" = torch.ops.aten.copy_.default(primals_112, add_92);  primals_112 = add_92 = copy__55 = None
        copy__56: "f32[192]" = torch.ops.aten.copy_.default(primals_113, add_93);  primals_113 = add_93 = copy__56 = None
        copy__57: "i64[]" = torch.ops.aten.copy_.default(primals_117, add_95);  primals_117 = add_95 = copy__57 = None
        copy__58: "f32[192]" = torch.ops.aten.copy_.default(primals_118, add_97);  primals_118 = add_97 = copy__58 = None
        copy__59: "f32[192]" = torch.ops.aten.copy_.default(primals_119, add_98);  primals_119 = add_98 = copy__59 = None
        copy__60: "i64[]" = torch.ops.aten.copy_.default(primals_123, add_100);  primals_123 = add_100 = copy__60 = None
        copy__61: "f32[768]" = torch.ops.aten.copy_.default(primals_124, add_102);  primals_124 = add_102 = copy__61 = None
        copy__62: "f32[768]" = torch.ops.aten.copy_.default(primals_125, add_103);  primals_125 = add_103 = copy__62 = None
        copy__63: "i64[]" = torch.ops.aten.copy_.default(primals_129, add_105);  primals_129 = add_105 = copy__63 = None
        copy__64: "f32[192]" = torch.ops.aten.copy_.default(primals_130, add_107);  primals_130 = add_107 = copy__64 = None
        copy__65: "f32[192]" = torch.ops.aten.copy_.default(primals_131, add_108);  primals_131 = add_108 = copy__65 = None
        copy__66: "i64[]" = torch.ops.aten.copy_.default(primals_135, add_110);  primals_135 = add_110 = copy__66 = None
        copy__67: "f32[192]" = torch.ops.aten.copy_.default(primals_136, add_112);  primals_136 = add_112 = copy__67 = None
        copy__68: "f32[192]" = torch.ops.aten.copy_.default(primals_137, add_113);  primals_137 = add_113 = copy__68 = None
        copy__69: "i64[]" = torch.ops.aten.copy_.default(primals_141, add_115);  primals_141 = add_115 = copy__69 = None
        copy__70: "f32[192]" = torch.ops.aten.copy_.default(primals_142, add_117);  primals_142 = add_117 = copy__70 = None
        copy__71: "f32[192]" = torch.ops.aten.copy_.default(primals_143, add_118);  primals_143 = add_118 = copy__71 = None
        copy__72: "i64[]" = torch.ops.aten.copy_.default(primals_147, add_120);  primals_147 = add_120 = copy__72 = None
        copy__73: "f32[192]" = torch.ops.aten.copy_.default(primals_148, add_122);  primals_148 = add_122 = copy__73 = None
        copy__74: "f32[192]" = torch.ops.aten.copy_.default(primals_149, add_123);  primals_149 = add_123 = copy__74 = None
        copy__75: "i64[]" = torch.ops.aten.copy_.default(primals_153, add_125);  primals_153 = add_125 = copy__75 = None
        copy__76: "f32[192]" = torch.ops.aten.copy_.default(primals_154, add_127);  primals_154 = add_127 = copy__76 = None
        copy__77: "f32[192]" = torch.ops.aten.copy_.default(primals_155, add_128);  primals_155 = add_128 = copy__77 = None
        copy__78: "i64[]" = torch.ops.aten.copy_.default(primals_159, add_130);  primals_159 = add_130 = copy__78 = None
        copy__79: "f32[768]" = torch.ops.aten.copy_.default(primals_160, add_132);  primals_160 = add_132 = copy__79 = None
        copy__80: "f32[768]" = torch.ops.aten.copy_.default(primals_161, add_133);  primals_161 = add_133 = copy__80 = None
        copy__81: "i64[]" = torch.ops.aten.copy_.default(primals_165, add_135);  primals_165 = add_135 = copy__81 = None
        copy__82: "f32[224]" = torch.ops.aten.copy_.default(primals_166, add_137);  primals_166 = add_137 = copy__82 = None
        copy__83: "f32[224]" = torch.ops.aten.copy_.default(primals_167, add_138);  primals_167 = add_138 = copy__83 = None
        copy__84: "i64[]" = torch.ops.aten.copy_.default(primals_171, add_140);  primals_171 = add_140 = copy__84 = None
        copy__85: "f32[224]" = torch.ops.aten.copy_.default(primals_172, add_142);  primals_172 = add_142 = copy__85 = None
        copy__86: "f32[224]" = torch.ops.aten.copy_.default(primals_173, add_143);  primals_173 = add_143 = copy__86 = None
        copy__87: "i64[]" = torch.ops.aten.copy_.default(primals_177, add_145);  primals_177 = add_145 = copy__87 = None
        copy__88: "f32[224]" = torch.ops.aten.copy_.default(primals_178, add_147);  primals_178 = add_147 = copy__88 = None
        copy__89: "f32[224]" = torch.ops.aten.copy_.default(primals_179, add_148);  primals_179 = add_148 = copy__89 = None
        copy__90: "i64[]" = torch.ops.aten.copy_.default(primals_183, add_150);  primals_183 = add_150 = copy__90 = None
        copy__91: "f32[224]" = torch.ops.aten.copy_.default(primals_184, add_152);  primals_184 = add_152 = copy__91 = None
        copy__92: "f32[224]" = torch.ops.aten.copy_.default(primals_185, add_153);  primals_185 = add_153 = copy__92 = None
        copy__93: "i64[]" = torch.ops.aten.copy_.default(primals_189, add_155);  primals_189 = add_155 = copy__93 = None
        copy__94: "f32[224]" = torch.ops.aten.copy_.default(primals_190, add_157);  primals_190 = add_157 = copy__94 = None
        copy__95: "f32[224]" = torch.ops.aten.copy_.default(primals_191, add_158);  primals_191 = add_158 = copy__95 = None
        copy__96: "i64[]" = torch.ops.aten.copy_.default(primals_195, add_160);  primals_195 = add_160 = copy__96 = None
        copy__97: "f32[1024]" = torch.ops.aten.copy_.default(primals_196, add_162);  primals_196 = add_162 = copy__97 = None
        copy__98: "f32[1024]" = torch.ops.aten.copy_.default(primals_197, add_163);  primals_197 = add_163 = copy__98 = None
        copy__99: "i64[]" = torch.ops.aten.copy_.default(primals_201, add_165);  primals_201 = add_165 = copy__99 = None
        copy__100: "f32[224]" = torch.ops.aten.copy_.default(primals_202, add_167);  primals_202 = add_167 = copy__100 = None
        copy__101: "f32[224]" = torch.ops.aten.copy_.default(primals_203, add_168);  primals_203 = add_168 = copy__101 = None
        copy__102: "i64[]" = torch.ops.aten.copy_.default(primals_207, add_170);  primals_207 = add_170 = copy__102 = None
        copy__103: "f32[224]" = torch.ops.aten.copy_.default(primals_208, add_172);  primals_208 = add_172 = copy__103 = None
        copy__104: "f32[224]" = torch.ops.aten.copy_.default(primals_209, add_173);  primals_209 = add_173 = copy__104 = None
        copy__105: "i64[]" = torch.ops.aten.copy_.default(primals_213, add_175);  primals_213 = add_175 = copy__105 = None
        copy__106: "f32[224]" = torch.ops.aten.copy_.default(primals_214, add_177);  primals_214 = add_177 = copy__106 = None
        copy__107: "f32[224]" = torch.ops.aten.copy_.default(primals_215, add_178);  primals_215 = add_178 = copy__107 = None
        copy__108: "i64[]" = torch.ops.aten.copy_.default(primals_219, add_180);  primals_219 = add_180 = copy__108 = None
        copy__109: "f32[224]" = torch.ops.aten.copy_.default(primals_220, add_182);  primals_220 = add_182 = copy__109 = None
        copy__110: "f32[224]" = torch.ops.aten.copy_.default(primals_221, add_183);  primals_221 = add_183 = copy__110 = None
        copy__111: "i64[]" = torch.ops.aten.copy_.default(primals_225, add_185);  primals_225 = add_185 = copy__111 = None
        copy__112: "f32[224]" = torch.ops.aten.copy_.default(primals_226, add_187);  primals_226 = add_187 = copy__112 = None
        copy__113: "f32[224]" = torch.ops.aten.copy_.default(primals_227, add_188);  primals_227 = add_188 = copy__113 = None
        copy__114: "i64[]" = torch.ops.aten.copy_.default(primals_231, add_190);  primals_231 = add_190 = copy__114 = None
        copy__115: "f32[1024]" = torch.ops.aten.copy_.default(primals_232, add_192);  primals_232 = add_192 = copy__115 = None
        copy__116: "f32[1024]" = torch.ops.aten.copy_.default(primals_233, add_193);  primals_233 = add_193 = copy__116 = None
        return (addmm, primals_1, primals_2, primals_6, primals_8, primals_12, primals_14, primals_18, primals_20, primals_24, primals_26, primals_30, primals_32, primals_36, primals_38, primals_42, primals_44, primals_48, primals_49, primals_50, primals_54, primals_55, primals_56, primals_60, primals_62, primals_66, primals_68, primals_72, primals_74, primals_78, primals_80, primals_84, primals_85, primals_86, primals_90, primals_91, primals_92, primals_96, primals_98, primals_102, primals_104, primals_108, primals_110, primals_114, primals_116, primals_120, primals_121, primals_122, primals_126, primals_128, primals_132, primals_134, primals_138, primals_140, primals_144, primals_146, primals_150, primals_152, primals_156, primals_157, primals_158, primals_162, primals_163, primals_164, primals_168, primals_170, primals_174, primals_176, primals_180, primals_182, primals_186, primals_188, primals_192, primals_193, primals_194, primals_198, primals_200, primals_204, primals_206, primals_210, primals_212, primals_216, primals_218, primals_222, primals_224, primals_228, primals_229, primals_230, primals_234, primals_235, primals_236, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, convolution_3, squeeze_10, relu_3, convolution_4, squeeze_13, relu_4, convolution_5, squeeze_16, relu_5, convolution_6, squeeze_19, relu_6, convolution_7, getitem_15, rsqrt_7, cat, convolution_8, getitem_17, rsqrt_8, getitem_18, getitem_19, convolution_9, squeeze_28, relu_9, convolution_10, squeeze_31, relu_10, convolution_11, squeeze_34, relu_11, convolution_12, squeeze_37, relu_12, convolution_13, getitem_29, rsqrt_13, cat_1, convolution_14, getitem_31, rsqrt_14, getitem_32, getitem_33, convolution_15, squeeze_46, relu_15, convolution_16, squeeze_49, relu_16, convolution_17, squeeze_52, relu_17, convolution_18, squeeze_55, relu_18, convolution_19, getitem_43, rsqrt_19, cat_2, convolution_20, squeeze_61, relu_20, convolution_21, squeeze_64, relu_21, convolution_22, squeeze_67, relu_22, convolution_23, squeeze_70, relu_23, convolution_24, squeeze_73, relu_24, convolution_25, getitem_55, rsqrt_25, cat_3, convolution_26, getitem_57, rsqrt_26, getitem_58, getitem_59, convolution_27, squeeze_82, relu_27, convolution_28, squeeze_85, relu_28, convolution_29, squeeze_88, relu_29, convolution_30, squeeze_91, relu_30, convolution_31, getitem_69, rsqrt_31, cat_4, convolution_32, squeeze_97, relu_32, convolution_33, squeeze_100, relu_33, convolution_34, squeeze_103, relu_34, convolution_35, squeeze_106, relu_35, convolution_36, squeeze_109, relu_36, convolution_37, getitem_81, rsqrt_37, cat_5, convolution_38, getitem_83, rsqrt_38, view, unsqueeze_182, unsqueeze_194, unsqueeze_206, unsqueeze_218, unsqueeze_230, unsqueeze_254, unsqueeze_266, unsqueeze_278, unsqueeze_290, unsqueeze_326, unsqueeze_338, unsqueeze_350, unsqueeze_362, unsqueeze_374, unsqueeze_398, unsqueeze_410, unsqueeze_422, unsqueeze_434, unsqueeze_470, unsqueeze_482, unsqueeze_494, unsqueeze_506, unsqueeze_542, unsqueeze_554, unsqueeze_566, unsqueeze_578, unsqueeze_590, unsqueeze_602, unsqueeze_614)
