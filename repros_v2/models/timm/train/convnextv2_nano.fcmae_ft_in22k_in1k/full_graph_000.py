class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[80, 3, 4, 4][48, 1, 12, 3]cuda:0", primals_2: "f32[80][1]cuda:0", primals_3: "f32[128, 3, 224, 224][150528, 1, 672, 3]cuda:0", primals_4: "f32[80][1]cuda:0", primals_5: "f32[80][1]cuda:0", primals_6: "f32[80, 1, 7, 7][49, 1, 7, 1]cuda:0", primals_7: "f32[80][1]cuda:0", primals_8: "f32[80][1]cuda:0", primals_9: "f32[80][1]cuda:0", primals_10: "f32[320, 80, 1, 1][80, 1, 80, 80]cuda:0", primals_11: "f32[320][1]cuda:0", primals_12: "f32[320][1]cuda:0", primals_13: "f32[320][1]cuda:0", primals_14: "f32[80, 320, 1, 1][320, 1, 320, 320]cuda:0", primals_15: "f32[80][1]cuda:0", primals_16: "f32[80, 1, 7, 7][49, 1, 7, 1]cuda:0", primals_17: "f32[80][1]cuda:0", primals_18: "f32[80][1]cuda:0", primals_19: "f32[80][1]cuda:0", primals_20: "f32[320, 80, 1, 1][80, 1, 80, 80]cuda:0", primals_21: "f32[320][1]cuda:0", primals_22: "f32[320][1]cuda:0", primals_23: "f32[320][1]cuda:0", primals_24: "f32[80, 320, 1, 1][320, 1, 320, 320]cuda:0", primals_25: "f32[80][1]cuda:0", primals_26: "f32[80][1]cuda:0", primals_27: "f32[80][1]cuda:0", primals_28: "f32[160, 80, 2, 2][320, 1, 160, 80]cuda:0", primals_29: "f32[160][1]cuda:0", primals_30: "f32[160, 1, 7, 7][49, 1, 7, 1]cuda:0", primals_31: "f32[160][1]cuda:0", primals_32: "f32[160][1]cuda:0", primals_33: "f32[160][1]cuda:0", primals_34: "f32[640, 160, 1, 1][160, 1, 160, 160]cuda:0", primals_35: "f32[640][1]cuda:0", primals_36: "f32[640][1]cuda:0", primals_37: "f32[640][1]cuda:0", primals_38: "f32[160, 640, 1, 1][640, 1, 640, 640]cuda:0", primals_39: "f32[160][1]cuda:0", primals_40: "f32[160, 1, 7, 7][49, 1, 7, 1]cuda:0", primals_41: "f32[160][1]cuda:0", primals_42: "f32[160][1]cuda:0", primals_43: "f32[160][1]cuda:0", primals_44: "f32[640, 160, 1, 1][160, 1, 160, 160]cuda:0", primals_45: "f32[640][1]cuda:0", primals_46: "f32[640][1]cuda:0", primals_47: "f32[640][1]cuda:0", primals_48: "f32[160, 640, 1, 1][640, 1, 640, 640]cuda:0", primals_49: "f32[160][1]cuda:0", primals_50: "f32[160][1]cuda:0", primals_51: "f32[160][1]cuda:0", primals_52: "f32[320, 160, 2, 2][640, 1, 320, 160]cuda:0", primals_53: "f32[320][1]cuda:0", primals_54: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0", primals_55: "f32[320][1]cuda:0", primals_56: "f32[320][1]cuda:0", primals_57: "f32[320][1]cuda:0", primals_58: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", primals_59: "f32[1280][1]cuda:0", primals_60: "f32[1280][1]cuda:0", primals_61: "f32[1280][1]cuda:0", primals_62: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", primals_63: "f32[320][1]cuda:0", primals_64: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0", primals_65: "f32[320][1]cuda:0", primals_66: "f32[320][1]cuda:0", primals_67: "f32[320][1]cuda:0", primals_68: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", primals_69: "f32[1280][1]cuda:0", primals_70: "f32[1280][1]cuda:0", primals_71: "f32[1280][1]cuda:0", primals_72: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", primals_73: "f32[320][1]cuda:0", primals_74: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0", primals_75: "f32[320][1]cuda:0", primals_76: "f32[320][1]cuda:0", primals_77: "f32[320][1]cuda:0", primals_78: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", primals_79: "f32[1280][1]cuda:0", primals_80: "f32[1280][1]cuda:0", primals_81: "f32[1280][1]cuda:0", primals_82: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", primals_83: "f32[320][1]cuda:0", primals_84: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0", primals_85: "f32[320][1]cuda:0", primals_86: "f32[320][1]cuda:0", primals_87: "f32[320][1]cuda:0", primals_88: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", primals_89: "f32[1280][1]cuda:0", primals_90: "f32[1280][1]cuda:0", primals_91: "f32[1280][1]cuda:0", primals_92: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", primals_93: "f32[320][1]cuda:0", primals_94: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0", primals_95: "f32[320][1]cuda:0", primals_96: "f32[320][1]cuda:0", primals_97: "f32[320][1]cuda:0", primals_98: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", primals_99: "f32[1280][1]cuda:0", primals_100: "f32[1280][1]cuda:0", primals_101: "f32[1280][1]cuda:0", primals_102: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", primals_103: "f32[320][1]cuda:0", primals_104: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0", primals_105: "f32[320][1]cuda:0", primals_106: "f32[320][1]cuda:0", primals_107: "f32[320][1]cuda:0", primals_108: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", primals_109: "f32[1280][1]cuda:0", primals_110: "f32[1280][1]cuda:0", primals_111: "f32[1280][1]cuda:0", primals_112: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", primals_113: "f32[320][1]cuda:0", primals_114: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0", primals_115: "f32[320][1]cuda:0", primals_116: "f32[320][1]cuda:0", primals_117: "f32[320][1]cuda:0", primals_118: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", primals_119: "f32[1280][1]cuda:0", primals_120: "f32[1280][1]cuda:0", primals_121: "f32[1280][1]cuda:0", primals_122: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", primals_123: "f32[320][1]cuda:0", primals_124: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0", primals_125: "f32[320][1]cuda:0", primals_126: "f32[320][1]cuda:0", primals_127: "f32[320][1]cuda:0", primals_128: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", primals_129: "f32[1280][1]cuda:0", primals_130: "f32[1280][1]cuda:0", primals_131: "f32[1280][1]cuda:0", primals_132: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", primals_133: "f32[320][1]cuda:0", primals_134: "f32[320][1]cuda:0", primals_135: "f32[320][1]cuda:0", primals_136: "f32[640, 320, 2, 2][1280, 1, 640, 320]cuda:0", primals_137: "f32[640][1]cuda:0", primals_138: "f32[640, 1, 7, 7][49, 1, 7, 1]cuda:0", primals_139: "f32[640][1]cuda:0", primals_140: "f32[640][1]cuda:0", primals_141: "f32[640][1]cuda:0", primals_142: "f32[2560, 640, 1, 1][640, 1, 640, 640]cuda:0", primals_143: "f32[2560][1]cuda:0", primals_144: "f32[2560][1]cuda:0", primals_145: "f32[2560][1]cuda:0", primals_146: "f32[640, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0", primals_147: "f32[640][1]cuda:0", primals_148: "f32[640, 1, 7, 7][49, 1, 7, 1]cuda:0", primals_149: "f32[640][1]cuda:0", primals_150: "f32[640][1]cuda:0", primals_151: "f32[640][1]cuda:0", primals_152: "f32[2560, 640, 1, 1][640, 1, 640, 640]cuda:0", primals_153: "f32[2560][1]cuda:0", primals_154: "f32[2560][1]cuda:0", primals_155: "f32[2560][1]cuda:0", primals_156: "f32[640, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0", primals_157: "f32[640][1]cuda:0", primals_158: "f32[640][1]cuda:0", primals_159: "f32[640][1]cuda:0", primals_160: "f32[1000, 640][640, 1]cuda:0", primals_161: "f32[1000][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:608 in forward_features, code: x = self.stem(x)
        convert_element_type: "bf16[80][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16);  primals_2 = None
        convert_element_type_1: "bf16[80, 3, 4, 4][48, 1, 12, 3]cuda:0" = torch.ops.prims.convert_element_type.default(primals_1, torch.bfloat16);  primals_1 = None
        convert_element_type_2: "bf16[128, 3, 224, 224][150528, 1, 672, 3]cuda:0" = torch.ops.prims.convert_element_type.default(primals_3, torch.bfloat16);  primals_3 = None
        convolution: "bf16[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_2, convert_element_type_1, convert_element_type, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute: "bf16[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.permute.default(convolution, [0, 2, 3, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_3: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute, torch.float32);  permute = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_3, [3], correction = 0, keepdim = True)
        getitem: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        add: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_3, getitem_1);  convert_element_type_3 = None
        mul: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, primals_4);  mul = None
        add_1: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, primals_5);  mul_1 = primals_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_1: "f32[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = torch.ops.aten.permute.default(add_1, [0, 3, 1, 2]);  add_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convert_element_type_4: "bf16[80][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_7, torch.bfloat16);  primals_7 = None
        convert_element_type_5: "bf16[80, 1, 7, 7][49, 1, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_6, torch.bfloat16);  primals_6 = None
        convert_element_type_6: "bf16[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = torch.ops.prims.convert_element_type.default(permute_1, torch.bfloat16)
        convolution_1: "bf16[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_6, convert_element_type_5, convert_element_type_4, [1, 1], [3, 3], [1, 1], False, [0, 0], 80);  convert_element_type_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_2: "bf16[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.permute.default(convolution_1, [0, 2, 3, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_7: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_2, torch.float32);  permute_2 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_7, [3], correction = 0, keepdim = True)
        getitem_2: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = var_mean_1[0]
        getitem_3: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        add_2: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_2, 1e-06);  getitem_2 = None
        rsqrt_1: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub_1: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_7, getitem_3);  convert_element_type_7 = None
        mul_2: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_3: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2, primals_8);  mul_2 = None
        add_3: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_3, primals_9);  mul_3 = primals_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_3: "f32[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = torch.ops.aten.permute.default(add_3, [0, 3, 1, 2]);  add_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convert_element_type_8: "bf16[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_11, torch.bfloat16);  primals_11 = None
        convert_element_type_9: "bf16[320, 80, 1, 1][80, 1, 80, 80]cuda:0" = torch.ops.prims.convert_element_type.default(primals_10, torch.bfloat16);  primals_10 = None
        convert_element_type_10: "bf16[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = torch.ops.prims.convert_element_type.default(permute_3, torch.bfloat16);  permute_3 = None
        convolution_2: "bf16[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_10, convert_element_type_9, convert_element_type_8, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_11: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_2, torch.float32)
        mul_4: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_11, 0.5)
        mul_5: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_11, 0.7071067811865476);  convert_element_type_11 = None
        erf: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_4: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(mul_4, add_4);  mul_4 = add_4 = None
        convert_element_type_12: "bf16[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        convert_element_type_13: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_12, torch.float32)
        pow_1: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_13, 2);  convert_element_type_13 = None
        sum_1: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_1, [2, 3], True);  pow_1 = None
        pow_2: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_1, 0.5);  sum_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_2, [1], True)
        add_5: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        div: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(pow_2, add_5)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_12, [1, -1, 1, 1]);  primals_12 = None
        view_1: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_13, [1, -1, 1, 1])
        mul_7: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_12, div);  div = None
        addcmul: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.addcmul.default(view, view_1, mul_7);  view = view_1 = mul_7 = None
        add_6: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_12, addcmul);  convert_element_type_12 = addcmul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convert_element_type_14: "bf16[80][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16);  primals_15 = None
        convert_element_type_15: "bf16[80, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
        convert_element_type_16: "bf16[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16);  add_6 = None
        convolution_3: "bf16[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_16, convert_element_type_15, convert_element_type_14, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_7: "f32[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = torch.ops.aten.add.Tensor(convolution_3, permute_1);  convolution_3 = permute_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convert_element_type_17: "bf16[80][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_17, torch.bfloat16);  primals_17 = None
        convert_element_type_18: "bf16[80, 1, 7, 7][49, 1, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_16, torch.bfloat16);  primals_16 = None
        convert_element_type_19: "bf16[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = torch.ops.prims.convert_element_type.default(add_7, torch.bfloat16)
        convolution_4: "bf16[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_19, convert_element_type_18, convert_element_type_17, [1, 1], [3, 3], [1, 1], False, [0, 0], 80);  convert_element_type_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_4: "bf16[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.permute.default(convolution_4, [0, 2, 3, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_20: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_4, torch.float32);  permute_4 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_20, [3], correction = 0, keepdim = True)
        getitem_4: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = var_mean_2[0]
        getitem_5: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        add_8: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 1e-06);  getitem_4 = None
        rsqrt_2: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_2: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_20, getitem_5);  convert_element_type_20 = None
        mul_8: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        mul_9: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_8, primals_18);  mul_8 = None
        add_9: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_9, primals_19);  mul_9 = primals_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_5: "f32[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = torch.ops.aten.permute.default(add_9, [0, 3, 1, 2]);  add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convert_element_type_21: "bf16[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_21, torch.bfloat16);  primals_21 = None
        convert_element_type_22: "bf16[320, 80, 1, 1][80, 1, 80, 80]cuda:0" = torch.ops.prims.convert_element_type.default(primals_20, torch.bfloat16);  primals_20 = None
        convert_element_type_23: "bf16[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = torch.ops.prims.convert_element_type.default(permute_5, torch.bfloat16);  permute_5 = None
        convolution_5: "bf16[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_23, convert_element_type_22, convert_element_type_21, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_24: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_5, torch.float32)
        mul_10: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_24, 0.5)
        mul_11: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_24, 0.7071067811865476);  convert_element_type_24 = None
        erf_1: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.erf.default(mul_11);  mul_11 = None
        add_10: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_12: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, add_10);  mul_10 = add_10 = None
        convert_element_type_25: "bf16[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.prims.convert_element_type.default(mul_12, torch.bfloat16);  mul_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        convert_element_type_26: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_25, torch.float32)
        pow_3: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_26, 2);  convert_element_type_26 = None
        sum_2: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_3, [2, 3], True);  pow_3 = None
        pow_4: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_2, 0.5);  sum_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_1: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_4, [1], True)
        add_11: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        div_1: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(pow_4, add_11)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_2: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_22, [1, -1, 1, 1]);  primals_22 = None
        view_3: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_23, [1, -1, 1, 1])
        mul_13: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_25, div_1);  div_1 = None
        addcmul_1: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.addcmul.default(view_2, view_3, mul_13);  view_2 = view_3 = mul_13 = None
        add_12: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_25, addcmul_1);  convert_element_type_25 = addcmul_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convert_element_type_27: "bf16[80][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_25, torch.bfloat16);  primals_25 = None
        convert_element_type_28: "bf16[80, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.prims.convert_element_type.default(primals_24, torch.bfloat16);  primals_24 = None
        convert_element_type_29: "bf16[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.prims.convert_element_type.default(add_12, torch.bfloat16);  add_12 = None
        convolution_6: "bf16[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_29, convert_element_type_28, convert_element_type_27, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_13: "f32[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = torch.ops.aten.add.Tensor(convolution_6, add_7);  convolution_6 = add_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_6: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.permute.default(add_13, [0, 2, 3, 1]);  add_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_3 = torch.ops.aten.var_mean.correction(permute_6, [3], correction = 0, keepdim = True)
        getitem_6: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = var_mean_3[0]
        getitem_7: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        add_14: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_6, 1e-06);  getitem_6 = None
        rsqrt_3: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        sub_3: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute_6, getitem_7);  permute_6 = getitem_7 = None
        mul_14: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        mul_15: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, primals_26)
        add_15: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_15, primals_27);  mul_15 = primals_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_7: "f32[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = torch.ops.aten.permute.default(add_15, [0, 3, 1, 2]);  add_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:302 in forward, code: x = self.downsample(x)
        convert_element_type_30: "bf16[160][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_29, torch.bfloat16);  primals_29 = None
        convert_element_type_31: "bf16[160, 80, 2, 2][320, 1, 160, 80]cuda:0" = torch.ops.prims.convert_element_type.default(primals_28, torch.bfloat16);  primals_28 = None
        convert_element_type_32: "bf16[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = torch.ops.prims.convert_element_type.default(permute_7, torch.bfloat16);  permute_7 = None
        convolution_7: "bf16[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_32, convert_element_type_31, convert_element_type_30, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convert_element_type_33: "bf16[160][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_31, torch.bfloat16);  primals_31 = None
        convert_element_type_34: "bf16[160, 1, 7, 7][49, 1, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_30, torch.bfloat16);  primals_30 = None
        convolution_8: "bf16[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = torch.ops.aten.convolution.default(convolution_7, convert_element_type_34, convert_element_type_33, [1, 1], [3, 3], [1, 1], False, [0, 0], 160);  convert_element_type_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_8: "bf16[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.permute.default(convolution_8, [0, 2, 3, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_35: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_8, torch.float32);  permute_8 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_35, [3], correction = 0, keepdim = True)
        getitem_8: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = var_mean_4[0]
        getitem_9: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        add_16: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 1e-06);  getitem_8 = None
        rsqrt_4: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_4: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_35, getitem_9);  convert_element_type_35 = None
        mul_16: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        mul_17: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_16, primals_32);  mul_16 = None
        add_17: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_17, primals_33);  mul_17 = primals_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_9: "f32[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = torch.ops.aten.permute.default(add_17, [0, 3, 1, 2]);  add_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convert_element_type_36: "bf16[640][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_35, torch.bfloat16);  primals_35 = None
        convert_element_type_37: "bf16[640, 160, 1, 1][160, 1, 160, 160]cuda:0" = torch.ops.prims.convert_element_type.default(primals_34, torch.bfloat16);  primals_34 = None
        convert_element_type_38: "bf16[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = torch.ops.prims.convert_element_type.default(permute_9, torch.bfloat16);  permute_9 = None
        convolution_9: "bf16[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_38, convert_element_type_37, convert_element_type_36, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_39: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_9, torch.float32)
        mul_18: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_39, 0.5)
        mul_19: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_39, 0.7071067811865476);  convert_element_type_39 = None
        erf_2: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
        add_18: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_20: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(mul_18, add_18);  mul_18 = add_18 = None
        convert_element_type_40: "bf16[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.prims.convert_element_type.default(mul_20, torch.bfloat16);  mul_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        convert_element_type_41: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_40, torch.float32)
        pow_5: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_41, 2);  convert_element_type_41 = None
        sum_3: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_5, [2, 3], True);  pow_5 = None
        pow_6: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_3, 0.5);  sum_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_2: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_6, [1], True)
        add_19: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
        div_2: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(pow_6, add_19)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_4: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_36, [1, -1, 1, 1]);  primals_36 = None
        view_5: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_37, [1, -1, 1, 1])
        mul_21: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_40, div_2);  div_2 = None
        addcmul_2: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.addcmul.default(view_4, view_5, mul_21);  view_4 = view_5 = mul_21 = None
        add_20: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_40, addcmul_2);  convert_element_type_40 = addcmul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convert_element_type_42: "bf16[160][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_39, torch.bfloat16);  primals_39 = None
        convert_element_type_43: "bf16[160, 640, 1, 1][640, 1, 640, 640]cuda:0" = torch.ops.prims.convert_element_type.default(primals_38, torch.bfloat16);  primals_38 = None
        convert_element_type_44: "bf16[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.prims.convert_element_type.default(add_20, torch.bfloat16);  add_20 = None
        convolution_10: "bf16[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_44, convert_element_type_43, convert_element_type_42, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_21: "bf16[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = torch.ops.aten.add.Tensor(convolution_10, convolution_7);  convolution_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convert_element_type_45: "bf16[160][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_41, torch.bfloat16);  primals_41 = None
        convert_element_type_46: "bf16[160, 1, 7, 7][49, 1, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_40, torch.bfloat16);  primals_40 = None
        convolution_11: "bf16[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = torch.ops.aten.convolution.default(add_21, convert_element_type_46, convert_element_type_45, [1, 1], [3, 3], [1, 1], False, [0, 0], 160);  convert_element_type_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_10: "bf16[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.permute.default(convolution_11, [0, 2, 3, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_47: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_10, torch.float32);  permute_10 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_47, [3], correction = 0, keepdim = True)
        getitem_10: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = var_mean_5[0]
        getitem_11: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        add_22: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_10, 1e-06);  getitem_10 = None
        rsqrt_5: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        sub_5: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_47, getitem_11);  convert_element_type_47 = None
        mul_22: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        mul_23: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_22, primals_42);  mul_22 = None
        add_23: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_23, primals_43);  mul_23 = primals_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_11: "f32[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = torch.ops.aten.permute.default(add_23, [0, 3, 1, 2]);  add_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convert_element_type_48: "bf16[640][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_45, torch.bfloat16);  primals_45 = None
        convert_element_type_49: "bf16[640, 160, 1, 1][160, 1, 160, 160]cuda:0" = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16);  primals_44 = None
        convert_element_type_50: "bf16[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = torch.ops.prims.convert_element_type.default(permute_11, torch.bfloat16);  permute_11 = None
        convolution_12: "bf16[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_50, convert_element_type_49, convert_element_type_48, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_51: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_12, torch.float32)
        mul_24: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_51, 0.5)
        mul_25: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_51, 0.7071067811865476);  convert_element_type_51 = None
        erf_3: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.erf.default(mul_25);  mul_25 = None
        add_24: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_26: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, add_24);  mul_24 = add_24 = None
        convert_element_type_52: "bf16[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.prims.convert_element_type.default(mul_26, torch.bfloat16);  mul_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        convert_element_type_53: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_52, torch.float32)
        pow_7: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_53, 2);  convert_element_type_53 = None
        sum_4: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_7, [2, 3], True);  pow_7 = None
        pow_8: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_4, 0.5);  sum_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_3: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_8, [1], True)
        add_25: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None
        div_3: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(pow_8, add_25)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_6: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_46, [1, -1, 1, 1]);  primals_46 = None
        view_7: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_47, [1, -1, 1, 1])
        mul_27: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_52, div_3);  div_3 = None
        addcmul_3: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.addcmul.default(view_6, view_7, mul_27);  view_6 = view_7 = mul_27 = None
        add_26: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_52, addcmul_3);  convert_element_type_52 = addcmul_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convert_element_type_54: "bf16[160][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_49, torch.bfloat16);  primals_49 = None
        convert_element_type_55: "bf16[160, 640, 1, 1][640, 1, 640, 640]cuda:0" = torch.ops.prims.convert_element_type.default(primals_48, torch.bfloat16);  primals_48 = None
        convert_element_type_56: "bf16[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.prims.convert_element_type.default(add_26, torch.bfloat16);  add_26 = None
        convolution_13: "bf16[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_56, convert_element_type_55, convert_element_type_54, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_27: "bf16[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = torch.ops.aten.add.Tensor(convolution_13, add_21);  convolution_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_12: "bf16[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.permute.default(add_27, [0, 2, 3, 1]);  add_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_57: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_12, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_57, [3], correction = 0, keepdim = True)
        getitem_12: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = var_mean_6[0]
        getitem_13: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        add_28: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
        rsqrt_6: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_6: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_57, getitem_13);  convert_element_type_57 = None
        mul_28: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        mul_29: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, primals_50);  mul_28 = None
        add_29: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_29, primals_51);  mul_29 = primals_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_13: "f32[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = torch.ops.aten.permute.default(add_29, [0, 3, 1, 2]);  add_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:302 in forward, code: x = self.downsample(x)
        convert_element_type_58: "bf16[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_53, torch.bfloat16);  primals_53 = None
        convert_element_type_59: "bf16[320, 160, 2, 2][640, 1, 320, 160]cuda:0" = torch.ops.prims.convert_element_type.default(primals_52, torch.bfloat16);  primals_52 = None
        convert_element_type_60: "bf16[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = torch.ops.prims.convert_element_type.default(permute_13, torch.bfloat16);  permute_13 = None
        convolution_14: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_60, convert_element_type_59, convert_element_type_58, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convert_element_type_61: "bf16[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_55, torch.bfloat16);  primals_55 = None
        convert_element_type_62: "bf16[320, 1, 7, 7][49, 1, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_54, torch.bfloat16);  primals_54 = None
        convolution_15: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.convolution.default(convolution_14, convert_element_type_62, convert_element_type_61, [1, 1], [3, 3], [1, 1], False, [0, 0], 320);  convert_element_type_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_14: "bf16[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_15, [0, 2, 3, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_63: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_14, torch.float32);  permute_14 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_63, [3], correction = 0, keepdim = True)
        getitem_14: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_7[0]
        getitem_15: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        add_30: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
        rsqrt_7: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        sub_7: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_63, getitem_15);  convert_element_type_63 = None
        mul_30: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        mul_31: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_30, primals_56);  mul_30 = None
        add_31: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_31, primals_57);  mul_31 = primals_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_15: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.permute.default(add_31, [0, 3, 1, 2]);  add_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convert_element_type_64: "bf16[1280][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_59, torch.bfloat16);  primals_59 = None
        convert_element_type_65: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.prims.convert_element_type.default(primals_58, torch.bfloat16);  primals_58 = None
        convert_element_type_66: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.prims.convert_element_type.default(permute_15, torch.bfloat16);  permute_15 = None
        convolution_16: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_66, convert_element_type_65, convert_element_type_64, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_67: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_16, torch.float32)
        mul_32: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_67, 0.5)
        mul_33: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_67, 0.7071067811865476);  convert_element_type_67 = None
        erf_4: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
        add_32: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_34: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_32, add_32);  mul_32 = add_32 = None
        convert_element_type_68: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_34, torch.bfloat16);  mul_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        convert_element_type_69: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_68, torch.float32)
        pow_9: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_69, 2);  convert_element_type_69 = None
        sum_5: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_9, [2, 3], True);  pow_9 = None
        pow_10: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_5, 0.5);  sum_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_4: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_10, [1], True)
        add_33: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_4, 1e-06);  mean_4 = None
        div_4: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(pow_10, add_33)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_8: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_60, [1, -1, 1, 1]);  primals_60 = None
        view_9: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_61, [1, -1, 1, 1])
        mul_35: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_68, div_4);  div_4 = None
        addcmul_4: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.addcmul.default(view_8, view_9, mul_35);  view_8 = view_9 = mul_35 = None
        add_34: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_68, addcmul_4);  convert_element_type_68 = addcmul_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convert_element_type_70: "bf16[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_63, torch.bfloat16);  primals_63 = None
        convert_element_type_71: "bf16[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(primals_62, torch.bfloat16);  primals_62 = None
        convert_element_type_72: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(add_34, torch.bfloat16);  add_34 = None
        convolution_17: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_72, convert_element_type_71, convert_element_type_70, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_35: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.add.Tensor(convolution_17, convolution_14);  convolution_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convert_element_type_73: "bf16[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_65, torch.bfloat16);  primals_65 = None
        convert_element_type_74: "bf16[320, 1, 7, 7][49, 1, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_64, torch.bfloat16);  primals_64 = None
        convolution_18: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.convolution.default(add_35, convert_element_type_74, convert_element_type_73, [1, 1], [3, 3], [1, 1], False, [0, 0], 320);  convert_element_type_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_16: "bf16[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_18, [0, 2, 3, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_75: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_16, torch.float32);  permute_16 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_75, [3], correction = 0, keepdim = True)
        getitem_16: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_8[0]
        getitem_17: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        add_36: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-06);  getitem_16 = None
        rsqrt_8: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_8: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_75, getitem_17);  convert_element_type_75 = None
        mul_36: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        mul_37: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_36, primals_66);  mul_36 = None
        add_37: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_37, primals_67);  mul_37 = primals_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_17: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.permute.default(add_37, [0, 3, 1, 2]);  add_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convert_element_type_76: "bf16[1280][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_69, torch.bfloat16);  primals_69 = None
        convert_element_type_77: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.prims.convert_element_type.default(primals_68, torch.bfloat16);  primals_68 = None
        convert_element_type_78: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.prims.convert_element_type.default(permute_17, torch.bfloat16);  permute_17 = None
        convolution_19: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_78, convert_element_type_77, convert_element_type_76, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_79: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_19, torch.float32)
        mul_38: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_79, 0.5)
        mul_39: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_79, 0.7071067811865476);  convert_element_type_79 = None
        erf_5: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
        add_38: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_40: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, add_38);  mul_38 = add_38 = None
        convert_element_type_80: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_40, torch.bfloat16);  mul_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        convert_element_type_81: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_80, torch.float32)
        pow_11: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_81, 2);  convert_element_type_81 = None
        sum_6: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_11, [2, 3], True);  pow_11 = None
        pow_12: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_6, 0.5);  sum_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_5: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_12, [1], True)
        add_39: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_5, 1e-06);  mean_5 = None
        div_5: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(pow_12, add_39)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_10: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_70, [1, -1, 1, 1]);  primals_70 = None
        view_11: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_71, [1, -1, 1, 1])
        mul_41: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_80, div_5);  div_5 = None
        addcmul_5: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.addcmul.default(view_10, view_11, mul_41);  view_10 = view_11 = mul_41 = None
        add_40: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_80, addcmul_5);  convert_element_type_80 = addcmul_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convert_element_type_82: "bf16[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_73, torch.bfloat16);  primals_73 = None
        convert_element_type_83: "bf16[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(primals_72, torch.bfloat16);  primals_72 = None
        convert_element_type_84: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(add_40, torch.bfloat16);  add_40 = None
        convolution_20: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_84, convert_element_type_83, convert_element_type_82, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_41: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.add.Tensor(convolution_20, add_35);  convolution_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convert_element_type_85: "bf16[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_75, torch.bfloat16);  primals_75 = None
        convert_element_type_86: "bf16[320, 1, 7, 7][49, 1, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_74, torch.bfloat16);  primals_74 = None
        convolution_21: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.convolution.default(add_41, convert_element_type_86, convert_element_type_85, [1, 1], [3, 3], [1, 1], False, [0, 0], 320);  convert_element_type_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_18: "bf16[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_21, [0, 2, 3, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_87: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_18, torch.float32);  permute_18 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_87, [3], correction = 0, keepdim = True)
        getitem_18: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_9[0]
        getitem_19: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        add_42: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 1e-06);  getitem_18 = None
        rsqrt_9: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_9: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_87, getitem_19);  convert_element_type_87 = None
        mul_42: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        mul_43: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, primals_76);  mul_42 = None
        add_43: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_43, primals_77);  mul_43 = primals_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_19: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.permute.default(add_43, [0, 3, 1, 2]);  add_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convert_element_type_88: "bf16[1280][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_79, torch.bfloat16);  primals_79 = None
        convert_element_type_89: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.prims.convert_element_type.default(primals_78, torch.bfloat16);  primals_78 = None
        convert_element_type_90: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.prims.convert_element_type.default(permute_19, torch.bfloat16);  permute_19 = None
        convolution_22: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_90, convert_element_type_89, convert_element_type_88, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_91: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_22, torch.float32)
        mul_44: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_91, 0.5)
        mul_45: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_91, 0.7071067811865476);  convert_element_type_91 = None
        erf_6: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.erf.default(mul_45);  mul_45 = None
        add_44: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_46: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_44, add_44);  mul_44 = add_44 = None
        convert_element_type_92: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_46, torch.bfloat16);  mul_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        convert_element_type_93: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_92, torch.float32)
        pow_13: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_93, 2);  convert_element_type_93 = None
        sum_7: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_13, [2, 3], True);  pow_13 = None
        pow_14: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_7, 0.5);  sum_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_6: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_14, [1], True)
        add_45: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_6, 1e-06);  mean_6 = None
        div_6: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(pow_14, add_45)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_12: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_80, [1, -1, 1, 1]);  primals_80 = None
        view_13: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_81, [1, -1, 1, 1])
        mul_47: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_92, div_6);  div_6 = None
        addcmul_6: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.addcmul.default(view_12, view_13, mul_47);  view_12 = view_13 = mul_47 = None
        add_46: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_92, addcmul_6);  convert_element_type_92 = addcmul_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convert_element_type_94: "bf16[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_83, torch.bfloat16);  primals_83 = None
        convert_element_type_95: "bf16[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(primals_82, torch.bfloat16);  primals_82 = None
        convert_element_type_96: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(add_46, torch.bfloat16);  add_46 = None
        convolution_23: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_96, convert_element_type_95, convert_element_type_94, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_47: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.add.Tensor(convolution_23, add_41);  convolution_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convert_element_type_97: "bf16[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_85, torch.bfloat16);  primals_85 = None
        convert_element_type_98: "bf16[320, 1, 7, 7][49, 1, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_84, torch.bfloat16);  primals_84 = None
        convolution_24: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.convolution.default(add_47, convert_element_type_98, convert_element_type_97, [1, 1], [3, 3], [1, 1], False, [0, 0], 320);  convert_element_type_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_20: "bf16[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_24, [0, 2, 3, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_99: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_20, torch.float32);  permute_20 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_99, [3], correction = 0, keepdim = True)
        getitem_20: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_10[0]
        getitem_21: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        add_48: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 1e-06);  getitem_20 = None
        rsqrt_10: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        sub_10: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_99, getitem_21);  convert_element_type_99 = None
        mul_48: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        mul_49: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_48, primals_86);  mul_48 = None
        add_49: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_49, primals_87);  mul_49 = primals_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_21: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.permute.default(add_49, [0, 3, 1, 2]);  add_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convert_element_type_100: "bf16[1280][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_89, torch.bfloat16);  primals_89 = None
        convert_element_type_101: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.prims.convert_element_type.default(primals_88, torch.bfloat16);  primals_88 = None
        convert_element_type_102: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.prims.convert_element_type.default(permute_21, torch.bfloat16);  permute_21 = None
        convolution_25: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_102, convert_element_type_101, convert_element_type_100, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_103: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_25, torch.float32)
        mul_50: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_103, 0.5)
        mul_51: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_103, 0.7071067811865476);  convert_element_type_103 = None
        erf_7: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
        add_50: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_52: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_50, add_50);  mul_50 = add_50 = None
        convert_element_type_104: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_52, torch.bfloat16);  mul_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        convert_element_type_105: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_104, torch.float32)
        pow_15: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_105, 2);  convert_element_type_105 = None
        sum_8: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_15, [2, 3], True);  pow_15 = None
        pow_16: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_8, 0.5);  sum_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_7: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_16, [1], True)
        add_51: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_7, 1e-06);  mean_7 = None
        div_7: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(pow_16, add_51)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_14: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_90, [1, -1, 1, 1]);  primals_90 = None
        view_15: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_91, [1, -1, 1, 1])
        mul_53: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_104, div_7);  div_7 = None
        addcmul_7: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.addcmul.default(view_14, view_15, mul_53);  view_14 = view_15 = mul_53 = None
        add_52: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_104, addcmul_7);  convert_element_type_104 = addcmul_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convert_element_type_106: "bf16[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_93, torch.bfloat16);  primals_93 = None
        convert_element_type_107: "bf16[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(primals_92, torch.bfloat16);  primals_92 = None
        convert_element_type_108: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(add_52, torch.bfloat16);  add_52 = None
        convolution_26: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_108, convert_element_type_107, convert_element_type_106, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_53: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.add.Tensor(convolution_26, add_47);  convolution_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convert_element_type_109: "bf16[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_95, torch.bfloat16);  primals_95 = None
        convert_element_type_110: "bf16[320, 1, 7, 7][49, 1, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_94, torch.bfloat16);  primals_94 = None
        convolution_27: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.convolution.default(add_53, convert_element_type_110, convert_element_type_109, [1, 1], [3, 3], [1, 1], False, [0, 0], 320);  convert_element_type_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_22: "bf16[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_27, [0, 2, 3, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_111: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_22, torch.float32);  permute_22 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_111, [3], correction = 0, keepdim = True)
        getitem_22: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_11[0]
        getitem_23: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        add_54: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_22, 1e-06);  getitem_22 = None
        rsqrt_11: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        sub_11: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_111, getitem_23);  convert_element_type_111 = None
        mul_54: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        mul_55: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_54, primals_96);  mul_54 = None
        add_55: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_55, primals_97);  mul_55 = primals_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_23: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.permute.default(add_55, [0, 3, 1, 2]);  add_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convert_element_type_112: "bf16[1280][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_99, torch.bfloat16);  primals_99 = None
        convert_element_type_113: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.prims.convert_element_type.default(primals_98, torch.bfloat16);  primals_98 = None
        convert_element_type_114: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.prims.convert_element_type.default(permute_23, torch.bfloat16);  permute_23 = None
        convolution_28: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_114, convert_element_type_113, convert_element_type_112, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_115: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_28, torch.float32)
        mul_56: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_115, 0.5)
        mul_57: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_115, 0.7071067811865476);  convert_element_type_115 = None
        erf_8: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.erf.default(mul_57);  mul_57 = None
        add_56: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_58: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_56, add_56);  mul_56 = add_56 = None
        convert_element_type_116: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_58, torch.bfloat16);  mul_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        convert_element_type_117: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_116, torch.float32)
        pow_17: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_117, 2);  convert_element_type_117 = None
        sum_9: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_17, [2, 3], True);  pow_17 = None
        pow_18: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_9, 0.5);  sum_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_8: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_18, [1], True)
        add_57: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_8, 1e-06);  mean_8 = None
        div_8: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(pow_18, add_57)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_16: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_100, [1, -1, 1, 1]);  primals_100 = None
        view_17: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_101, [1, -1, 1, 1])
        mul_59: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_116, div_8);  div_8 = None
        addcmul_8: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.addcmul.default(view_16, view_17, mul_59);  view_16 = view_17 = mul_59 = None
        add_58: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_116, addcmul_8);  convert_element_type_116 = addcmul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convert_element_type_118: "bf16[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_103, torch.bfloat16);  primals_103 = None
        convert_element_type_119: "bf16[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(primals_102, torch.bfloat16);  primals_102 = None
        convert_element_type_120: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(add_58, torch.bfloat16);  add_58 = None
        convolution_29: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_120, convert_element_type_119, convert_element_type_118, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_59: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.add.Tensor(convolution_29, add_53);  convolution_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convert_element_type_121: "bf16[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_105, torch.bfloat16);  primals_105 = None
        convert_element_type_122: "bf16[320, 1, 7, 7][49, 1, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_104, torch.bfloat16);  primals_104 = None
        convolution_30: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.convolution.default(add_59, convert_element_type_122, convert_element_type_121, [1, 1], [3, 3], [1, 1], False, [0, 0], 320);  convert_element_type_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_24: "bf16[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_30, [0, 2, 3, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_123: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_24, torch.float32);  permute_24 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_123, [3], correction = 0, keepdim = True)
        getitem_24: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_12[0]
        getitem_25: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        add_60: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-06);  getitem_24 = None
        rsqrt_12: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_12: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_123, getitem_25);  convert_element_type_123 = None
        mul_60: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        mul_61: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_60, primals_106);  mul_60 = None
        add_61: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_61, primals_107);  mul_61 = primals_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_25: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.permute.default(add_61, [0, 3, 1, 2]);  add_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convert_element_type_124: "bf16[1280][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_109, torch.bfloat16);  primals_109 = None
        convert_element_type_125: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.prims.convert_element_type.default(primals_108, torch.bfloat16);  primals_108 = None
        convert_element_type_126: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.prims.convert_element_type.default(permute_25, torch.bfloat16);  permute_25 = None
        convolution_31: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_126, convert_element_type_125, convert_element_type_124, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_127: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_31, torch.float32)
        mul_62: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_127, 0.5)
        mul_63: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_127, 0.7071067811865476);  convert_element_type_127 = None
        erf_9: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
        add_62: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_64: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_62, add_62);  mul_62 = add_62 = None
        convert_element_type_128: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_64, torch.bfloat16);  mul_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        convert_element_type_129: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_128, torch.float32)
        pow_19: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_129, 2);  convert_element_type_129 = None
        sum_10: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_19, [2, 3], True);  pow_19 = None
        pow_20: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_10, 0.5);  sum_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_9: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_20, [1], True)
        add_63: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_9, 1e-06);  mean_9 = None
        div_9: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(pow_20, add_63)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_18: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_110, [1, -1, 1, 1]);  primals_110 = None
        view_19: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_111, [1, -1, 1, 1])
        mul_65: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_128, div_9);  div_9 = None
        addcmul_9: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.addcmul.default(view_18, view_19, mul_65);  view_18 = view_19 = mul_65 = None
        add_64: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_128, addcmul_9);  convert_element_type_128 = addcmul_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convert_element_type_130: "bf16[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_113, torch.bfloat16);  primals_113 = None
        convert_element_type_131: "bf16[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(primals_112, torch.bfloat16);  primals_112 = None
        convert_element_type_132: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(add_64, torch.bfloat16);  add_64 = None
        convolution_32: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_132, convert_element_type_131, convert_element_type_130, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_65: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.add.Tensor(convolution_32, add_59);  convolution_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convert_element_type_133: "bf16[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_115, torch.bfloat16);  primals_115 = None
        convert_element_type_134: "bf16[320, 1, 7, 7][49, 1, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_114, torch.bfloat16);  primals_114 = None
        convolution_33: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.convolution.default(add_65, convert_element_type_134, convert_element_type_133, [1, 1], [3, 3], [1, 1], False, [0, 0], 320);  convert_element_type_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_26: "bf16[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_33, [0, 2, 3, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_135: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_26, torch.float32);  permute_26 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_135, [3], correction = 0, keepdim = True)
        getitem_26: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_13[0]
        getitem_27: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        add_66: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-06);  getitem_26 = None
        rsqrt_13: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_13: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_135, getitem_27);  convert_element_type_135 = None
        mul_66: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        mul_67: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, primals_116);  mul_66 = None
        add_67: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_67, primals_117);  mul_67 = primals_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_27: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.permute.default(add_67, [0, 3, 1, 2]);  add_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convert_element_type_136: "bf16[1280][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_119, torch.bfloat16);  primals_119 = None
        convert_element_type_137: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.prims.convert_element_type.default(primals_118, torch.bfloat16);  primals_118 = None
        convert_element_type_138: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.prims.convert_element_type.default(permute_27, torch.bfloat16);  permute_27 = None
        convolution_34: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_138, convert_element_type_137, convert_element_type_136, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_139: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_34, torch.float32)
        mul_68: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_139, 0.5)
        mul_69: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_139, 0.7071067811865476);  convert_element_type_139 = None
        erf_10: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
        add_68: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_70: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, add_68);  mul_68 = add_68 = None
        convert_element_type_140: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_70, torch.bfloat16);  mul_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        convert_element_type_141: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_140, torch.float32)
        pow_21: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_141, 2);  convert_element_type_141 = None
        sum_11: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_21, [2, 3], True);  pow_21 = None
        pow_22: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_11, 0.5);  sum_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_10: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_22, [1], True)
        add_69: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_10, 1e-06);  mean_10 = None
        div_10: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(pow_22, add_69)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_20: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_120, [1, -1, 1, 1]);  primals_120 = None
        view_21: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_121, [1, -1, 1, 1])
        mul_71: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_140, div_10);  div_10 = None
        addcmul_10: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.addcmul.default(view_20, view_21, mul_71);  view_20 = view_21 = mul_71 = None
        add_70: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_140, addcmul_10);  convert_element_type_140 = addcmul_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convert_element_type_142: "bf16[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_123, torch.bfloat16);  primals_123 = None
        convert_element_type_143: "bf16[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(primals_122, torch.bfloat16);  primals_122 = None
        convert_element_type_144: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(add_70, torch.bfloat16);  add_70 = None
        convolution_35: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_144, convert_element_type_143, convert_element_type_142, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_71: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.add.Tensor(convolution_35, add_65);  convolution_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convert_element_type_145: "bf16[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_125, torch.bfloat16);  primals_125 = None
        convert_element_type_146: "bf16[320, 1, 7, 7][49, 1, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_124, torch.bfloat16);  primals_124 = None
        convolution_36: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.convolution.default(add_71, convert_element_type_146, convert_element_type_145, [1, 1], [3, 3], [1, 1], False, [0, 0], 320);  convert_element_type_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_28: "bf16[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_36, [0, 2, 3, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_147: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_28, torch.float32);  permute_28 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_147, [3], correction = 0, keepdim = True)
        getitem_28: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_14[0]
        getitem_29: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        add_72: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_28, 1e-06);  getitem_28 = None
        rsqrt_14: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        sub_14: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_147, getitem_29);  convert_element_type_147 = None
        mul_72: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        mul_73: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_72, primals_126);  mul_72 = None
        add_73: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_73, primals_127);  mul_73 = primals_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_29: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.permute.default(add_73, [0, 3, 1, 2]);  add_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convert_element_type_148: "bf16[1280][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_129, torch.bfloat16);  primals_129 = None
        convert_element_type_149: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.prims.convert_element_type.default(primals_128, torch.bfloat16);  primals_128 = None
        convert_element_type_150: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.prims.convert_element_type.default(permute_29, torch.bfloat16);  permute_29 = None
        convolution_37: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_150, convert_element_type_149, convert_element_type_148, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_151: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_37, torch.float32)
        mul_74: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_151, 0.5)
        mul_75: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_151, 0.7071067811865476);  convert_element_type_151 = None
        erf_11: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
        add_74: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_76: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_74, add_74);  mul_74 = add_74 = None
        convert_element_type_152: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_76, torch.bfloat16);  mul_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        convert_element_type_153: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_152, torch.float32)
        pow_23: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_153, 2);  convert_element_type_153 = None
        sum_12: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_23, [2, 3], True);  pow_23 = None
        pow_24: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_12, 0.5);  sum_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_11: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_24, [1], True)
        add_75: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_11, 1e-06);  mean_11 = None
        div_11: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(pow_24, add_75)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_22: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_130, [1, -1, 1, 1]);  primals_130 = None
        view_23: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_131, [1, -1, 1, 1])
        mul_77: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_152, div_11);  div_11 = None
        addcmul_11: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.addcmul.default(view_22, view_23, mul_77);  view_22 = view_23 = mul_77 = None
        add_76: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_152, addcmul_11);  convert_element_type_152 = addcmul_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convert_element_type_154: "bf16[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_133, torch.bfloat16);  primals_133 = None
        convert_element_type_155: "bf16[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(primals_132, torch.bfloat16);  primals_132 = None
        convert_element_type_156: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(add_76, torch.bfloat16);  add_76 = None
        convolution_38: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_156, convert_element_type_155, convert_element_type_154, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_77: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.add.Tensor(convolution_38, add_71);  convolution_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_30: "bf16[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(add_77, [0, 2, 3, 1]);  add_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_157: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_30, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_157, [3], correction = 0, keepdim = True)
        getitem_30: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_15[0]
        getitem_31: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        add_78: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
        rsqrt_15: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        sub_15: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_157, getitem_31);  convert_element_type_157 = None
        mul_78: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        mul_79: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_78, primals_134);  mul_78 = None
        add_79: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_79, primals_135);  mul_79 = primals_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_31: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.aten.permute.default(add_79, [0, 3, 1, 2]);  add_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:302 in forward, code: x = self.downsample(x)
        convert_element_type_158: "bf16[640][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_137, torch.bfloat16);  primals_137 = None
        convert_element_type_159: "bf16[640, 320, 2, 2][1280, 1, 640, 320]cuda:0" = torch.ops.prims.convert_element_type.default(primals_136, torch.bfloat16);  primals_136 = None
        convert_element_type_160: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.prims.convert_element_type.default(permute_31, torch.bfloat16);  permute_31 = None
        convolution_39: "bf16[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_160, convert_element_type_159, convert_element_type_158, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convert_element_type_161: "bf16[640][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_139, torch.bfloat16);  primals_139 = None
        convert_element_type_162: "bf16[640, 1, 7, 7][49, 1, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_138, torch.bfloat16);  primals_138 = None
        convolution_40: "bf16[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0" = torch.ops.aten.convolution.default(convolution_39, convert_element_type_162, convert_element_type_161, [1, 1], [3, 3], [1, 1], False, [0, 0], 640);  convert_element_type_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_32: "bf16[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.permute.default(convolution_40, [0, 2, 3, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_163: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_32, torch.float32);  permute_32 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_163, [3], correction = 0, keepdim = True)
        getitem_32: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = var_mean_16[0]
        getitem_33: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        add_80: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
        rsqrt_16: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        sub_16: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_163, getitem_33);  convert_element_type_163 = None
        mul_80: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        mul_81: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_80, primals_140);  mul_80 = None
        add_81: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_81, primals_141);  mul_81 = primals_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_33: "f32[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0" = torch.ops.aten.permute.default(add_81, [0, 3, 1, 2]);  add_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convert_element_type_164: "bf16[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_143, torch.bfloat16);  primals_143 = None
        convert_element_type_165: "bf16[2560, 640, 1, 1][640, 1, 640, 640]cuda:0" = torch.ops.prims.convert_element_type.default(primals_142, torch.bfloat16);  primals_142 = None
        convert_element_type_166: "bf16[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0" = torch.ops.prims.convert_element_type.default(permute_33, torch.bfloat16);  permute_33 = None
        convolution_41: "bf16[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_166, convert_element_type_165, convert_element_type_164, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_167: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_41, torch.float32)
        mul_82: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_167, 0.5)
        mul_83: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_167, 0.7071067811865476);  convert_element_type_167 = None
        erf_12: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
        add_82: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_84: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(mul_82, add_82);  mul_82 = add_82 = None
        convert_element_type_168: "bf16[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.prims.convert_element_type.default(mul_84, torch.bfloat16);  mul_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        convert_element_type_169: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_168, torch.float32)
        pow_25: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_169, 2);  convert_element_type_169 = None
        sum_13: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_25, [2, 3], True);  pow_25 = None
        pow_26: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_13, 0.5);  sum_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_12: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_26, [1], True)
        add_83: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_12, 1e-06);  mean_12 = None
        div_12: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(pow_26, add_83)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_24: "f32[1, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_144, [1, -1, 1, 1]);  primals_144 = None
        view_25: "f32[1, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_145, [1, -1, 1, 1])
        mul_85: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_168, div_12);  div_12 = None
        addcmul_12: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.addcmul.default(view_24, view_25, mul_85);  view_24 = view_25 = mul_85 = None
        add_84: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_168, addcmul_12);  convert_element_type_168 = addcmul_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convert_element_type_170: "bf16[640][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_147, torch.bfloat16);  primals_147 = None
        convert_element_type_171: "bf16[640, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0" = torch.ops.prims.convert_element_type.default(primals_146, torch.bfloat16);  primals_146 = None
        convert_element_type_172: "bf16[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.prims.convert_element_type.default(add_84, torch.bfloat16);  add_84 = None
        convolution_42: "bf16[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_172, convert_element_type_171, convert_element_type_170, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_85: "bf16[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0" = torch.ops.aten.add.Tensor(convolution_42, convolution_39);  convolution_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        convert_element_type_173: "bf16[640][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_149, torch.bfloat16);  primals_149 = None
        convert_element_type_174: "bf16[640, 1, 7, 7][49, 1, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_148, torch.bfloat16);  primals_148 = None
        convolution_43: "bf16[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0" = torch.ops.aten.convolution.default(add_85, convert_element_type_174, convert_element_type_173, [1, 1], [3, 3], [1, 1], False, [0, 0], 640);  convert_element_type_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_34: "bf16[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.permute.default(convolution_43, [0, 2, 3, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_175: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_34, torch.float32);  permute_34 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_175, [3], correction = 0, keepdim = True)
        getitem_34: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = var_mean_17[0]
        getitem_35: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        add_86: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 1e-06);  getitem_34 = None
        rsqrt_17: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        sub_17: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_175, getitem_35);  convert_element_type_175 = None
        mul_86: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        mul_87: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_86, primals_150);  mul_86 = None
        add_87: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_87, primals_151);  mul_87 = primals_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_35: "f32[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0" = torch.ops.aten.permute.default(add_87, [0, 3, 1, 2]);  add_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convert_element_type_176: "bf16[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_153, torch.bfloat16);  primals_153 = None
        convert_element_type_177: "bf16[2560, 640, 1, 1][640, 1, 640, 640]cuda:0" = torch.ops.prims.convert_element_type.default(primals_152, torch.bfloat16);  primals_152 = None
        convert_element_type_178: "bf16[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0" = torch.ops.prims.convert_element_type.default(permute_35, torch.bfloat16);  permute_35 = None
        convolution_44: "bf16[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_178, convert_element_type_177, convert_element_type_176, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_179: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_44, torch.float32)
        mul_88: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_179, 0.5)
        mul_89: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_179, 0.7071067811865476);  convert_element_type_179 = None
        erf_13: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.erf.default(mul_89);  mul_89 = None
        add_88: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_90: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(mul_88, add_88);  mul_88 = add_88 = None
        convert_element_type_180: "bf16[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.prims.convert_element_type.default(mul_90, torch.bfloat16);  mul_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        convert_element_type_181: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_180, torch.float32)
        pow_27: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_181, 2);  convert_element_type_181 = None
        sum_14: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_27, [2, 3], True);  pow_27 = None
        pow_28: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_14, 0.5);  sum_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_13: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_28, [1], True)
        add_89: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_13, 1e-06);  mean_13 = None
        div_13: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(pow_28, add_89)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_26: "f32[1, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_154, [1, -1, 1, 1]);  primals_154 = None
        view_27: "f32[1, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_155, [1, -1, 1, 1])
        mul_91: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_180, div_13);  div_13 = None
        addcmul_13: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.addcmul.default(view_26, view_27, mul_91);  view_26 = view_27 = mul_91 = None
        add_90: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_180, addcmul_13);  convert_element_type_180 = addcmul_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convert_element_type_182: "bf16[640][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_157, torch.bfloat16);  primals_157 = None
        convert_element_type_183: "bf16[640, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0" = torch.ops.prims.convert_element_type.default(primals_156, torch.bfloat16);  primals_156 = None
        convert_element_type_184: "bf16[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.prims.convert_element_type.default(add_90, torch.bfloat16);  add_90 = None
        convolution_45: "bf16[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_184, convert_element_type_183, convert_element_type_182, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_91: "bf16[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0" = torch.ops.aten.add.Tensor(convolution_45, add_85);  convolution_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_14: "bf16[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(add_91, [-1, -2], True);  add_91 = None
        as_strided: "bf16[128, 640, 1, 1][640, 1, 640, 640]cuda:0" = torch.ops.aten.as_strided.default(mean_14, [128, 640, 1, 1], [640, 1, 640, 640]);  mean_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_36: "bf16[128, 1, 1, 640][640, 640, 640, 1]cuda:0" = torch.ops.aten.permute.default(as_strided, [0, 2, 3, 1]);  as_strided = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_185: "f32[128, 1, 1, 640][640, 640, 640, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_36, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_185, [3], correction = 0, keepdim = True)
        getitem_36: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = var_mean_18[0]
        getitem_37: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        add_92: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_36, 1e-06);  getitem_36 = None
        rsqrt_18: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        sub_18: "f32[128, 1, 1, 640][640, 640, 640, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_185, getitem_37);  convert_element_type_185 = None
        mul_92: "f32[128, 1, 1, 640][640, 640, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        mul_93: "f32[128, 1, 1, 640][640, 640, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_92, primals_158);  mul_92 = None
        add_93: "f32[128, 1, 1, 640][640, 640, 640, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_93, primals_159);  mul_93 = primals_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_37: "f32[128, 640, 1, 1][640, 1, 640, 640]cuda:0" = torch.ops.aten.permute.default(add_93, [0, 3, 1, 2]);  add_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:214 in forward, code: x = self.flatten(x)
        view_28: "f32[128, 640][640, 1]cuda:0" = torch.ops.aten.reshape.default(permute_37, [128, 640]);  permute_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:219 in forward, code: x = self.fc(x)
        convert_element_type_186: "bf16[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_161, torch.bfloat16);  primals_161 = None
        convert_element_type_187: "bf16[1000, 640][640, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_160, torch.bfloat16);  primals_160 = None
        convert_element_type_188: "bf16[128, 640][640, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_28, torch.bfloat16);  view_28 = None
        permute_38: "bf16[640, 1000][1, 640]cuda:0" = torch.ops.aten.permute.default(convert_element_type_187, [1, 0]);  convert_element_type_187 = None
        addmm: "bf16[128, 1000][1000, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_186, convert_element_type_188, permute_38);  convert_element_type_186 = None
        permute_39: "bf16[1000, 640][640, 1]cuda:0" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        div_90: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_3, 80);  rsqrt_3 = None
        return (addmm, primals_4, primals_8, primals_13, primals_18, primals_23, primals_26, primals_32, primals_37, primals_42, primals_47, primals_50, primals_56, primals_61, primals_66, primals_71, primals_76, primals_81, primals_86, primals_91, primals_96, primals_101, primals_106, primals_111, primals_116, primals_121, primals_126, primals_131, primals_134, primals_140, primals_145, primals_150, primals_155, primals_158, convert_element_type_1, convert_element_type_2, convolution, getitem_1, rsqrt, convert_element_type_5, convert_element_type_6, convolution_1, getitem_3, rsqrt_1, convert_element_type_9, convert_element_type_10, convolution_2, pow_2, add_5, convert_element_type_15, convert_element_type_16, convert_element_type_18, convert_element_type_19, convolution_4, getitem_5, rsqrt_2, convert_element_type_22, convert_element_type_23, convolution_5, pow_4, add_11, convert_element_type_28, convert_element_type_29, mul_14, convert_element_type_31, convert_element_type_32, convolution_7, convert_element_type_34, convolution_8, getitem_9, rsqrt_4, convert_element_type_37, convert_element_type_38, convolution_9, pow_6, add_19, convert_element_type_43, convert_element_type_44, add_21, convert_element_type_46, convolution_11, getitem_11, rsqrt_5, convert_element_type_49, convert_element_type_50, convolution_12, pow_8, add_25, convert_element_type_55, convert_element_type_56, permute_12, getitem_13, rsqrt_6, convert_element_type_59, convert_element_type_60, convolution_14, convert_element_type_62, convolution_15, getitem_15, rsqrt_7, convert_element_type_65, convert_element_type_66, convolution_16, pow_10, add_33, convert_element_type_71, convert_element_type_72, add_35, convert_element_type_74, convolution_18, getitem_17, rsqrt_8, convert_element_type_77, convert_element_type_78, convolution_19, pow_12, add_39, convert_element_type_83, convert_element_type_84, add_41, convert_element_type_86, convolution_21, getitem_19, rsqrt_9, convert_element_type_89, convert_element_type_90, convolution_22, pow_14, add_45, convert_element_type_95, convert_element_type_96, add_47, convert_element_type_98, convolution_24, getitem_21, rsqrt_10, convert_element_type_101, convert_element_type_102, convolution_25, pow_16, add_51, convert_element_type_107, convert_element_type_108, add_53, convert_element_type_110, convolution_27, getitem_23, rsqrt_11, convert_element_type_113, convert_element_type_114, convolution_28, pow_18, add_57, convert_element_type_119, convert_element_type_120, add_59, convert_element_type_122, convolution_30, getitem_25, rsqrt_12, convert_element_type_125, convert_element_type_126, convolution_31, pow_20, add_63, convert_element_type_131, convert_element_type_132, add_65, convert_element_type_134, convolution_33, getitem_27, rsqrt_13, convert_element_type_137, convert_element_type_138, convolution_34, pow_22, add_69, convert_element_type_143, convert_element_type_144, add_71, convert_element_type_146, convolution_36, getitem_29, rsqrt_14, convert_element_type_149, convert_element_type_150, convolution_37, pow_24, add_75, convert_element_type_155, convert_element_type_156, permute_30, getitem_31, rsqrt_15, convert_element_type_159, convert_element_type_160, convolution_39, convert_element_type_162, convolution_40, getitem_33, rsqrt_16, convert_element_type_165, convert_element_type_166, convolution_41, pow_26, add_83, convert_element_type_171, convert_element_type_172, add_85, convert_element_type_174, convolution_43, getitem_35, rsqrt_17, convert_element_type_177, convert_element_type_178, convolution_44, pow_28, add_89, convert_element_type_183, convert_element_type_184, permute_36, getitem_37, rsqrt_18, convert_element_type_188, permute_39, div_90)
