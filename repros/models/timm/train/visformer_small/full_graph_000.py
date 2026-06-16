class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[32, 3, 7, 7][147, 1, 21, 3]cuda:0", primals_2: "f32[128, 3, 224, 224][150528, 1, 672, 3]cuda:0", primals_3: "i64[][]cuda:0", primals_4: "f32[32][1]cuda:0", primals_5: "f32[32][1]cuda:0", primals_6: "f32[32][1]cuda:0", primals_7: "f32[32][1]cuda:0", primals_8: "f32[192, 32, 4, 4][512, 1, 128, 32]cuda:0", primals_9: "f32[192][1]cuda:0", primals_10: "i64[][]cuda:0", primals_11: "f32[192][1]cuda:0", primals_12: "f32[192][1]cuda:0", primals_13: "f32[192][1]cuda:0", primals_14: "f32[192][1]cuda:0", primals_15: "f32[1, 192, 28, 28][150528, 1, 5376, 192]cuda:0", primals_16: "i64[][]cuda:0", primals_17: "f32[192][1]cuda:0", primals_18: "f32[192][1]cuda:0", primals_19: "f32[192][1]cuda:0", primals_20: "f32[192][1]cuda:0", primals_21: "f32[384, 192, 1, 1][192, 1, 192, 192]cuda:0", primals_22: "f32[384, 48, 3, 3][432, 1, 144, 48]cuda:0", primals_23: "f32[192, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_24: "i64[][]cuda:0", primals_25: "f32[192][1]cuda:0", primals_26: "f32[192][1]cuda:0", primals_27: "f32[192][1]cuda:0", primals_28: "f32[192][1]cuda:0", primals_29: "f32[384, 192, 1, 1][192, 1, 192, 192]cuda:0", primals_30: "f32[384, 48, 3, 3][432, 1, 144, 48]cuda:0", primals_31: "f32[192, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_32: "i64[][]cuda:0", primals_33: "f32[192][1]cuda:0", primals_34: "f32[192][1]cuda:0", primals_35: "f32[192][1]cuda:0", primals_36: "f32[192][1]cuda:0", primals_37: "f32[384, 192, 1, 1][192, 1, 192, 192]cuda:0", primals_38: "f32[384, 48, 3, 3][432, 1, 144, 48]cuda:0", primals_39: "f32[192, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_40: "i64[][]cuda:0", primals_41: "f32[192][1]cuda:0", primals_42: "f32[192][1]cuda:0", primals_43: "f32[192][1]cuda:0", primals_44: "f32[192][1]cuda:0", primals_45: "f32[384, 192, 1, 1][192, 1, 192, 192]cuda:0", primals_46: "f32[384, 48, 3, 3][432, 1, 144, 48]cuda:0", primals_47: "f32[192, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_48: "i64[][]cuda:0", primals_49: "f32[192][1]cuda:0", primals_50: "f32[192][1]cuda:0", primals_51: "f32[192][1]cuda:0", primals_52: "f32[192][1]cuda:0", primals_53: "f32[384, 192, 1, 1][192, 1, 192, 192]cuda:0", primals_54: "f32[384, 48, 3, 3][432, 1, 144, 48]cuda:0", primals_55: "f32[192, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_56: "i64[][]cuda:0", primals_57: "f32[192][1]cuda:0", primals_58: "f32[192][1]cuda:0", primals_59: "f32[192][1]cuda:0", primals_60: "f32[192][1]cuda:0", primals_61: "f32[384, 192, 1, 1][192, 1, 192, 192]cuda:0", primals_62: "f32[384, 48, 3, 3][432, 1, 144, 48]cuda:0", primals_63: "f32[192, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_64: "i64[][]cuda:0", primals_65: "f32[192][1]cuda:0", primals_66: "f32[192][1]cuda:0", primals_67: "f32[192][1]cuda:0", primals_68: "f32[192][1]cuda:0", primals_69: "f32[384, 192, 1, 1][192, 1, 192, 192]cuda:0", primals_70: "f32[384, 48, 3, 3][432, 1, 144, 48]cuda:0", primals_71: "f32[192, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_72: "f32[384, 192, 2, 2][768, 1, 384, 192]cuda:0", primals_73: "f32[384][1]cuda:0", primals_74: "i64[][]cuda:0", primals_75: "f32[384][1]cuda:0", primals_76: "f32[384][1]cuda:0", primals_77: "f32[384][1]cuda:0", primals_78: "f32[384][1]cuda:0", primals_79: "f32[1, 384, 14, 14][75264, 1, 5376, 384]cuda:0", primals_80: "i64[][]cuda:0", primals_81: "f32[384][1]cuda:0", primals_82: "f32[384][1]cuda:0", primals_83: "f32[384][1]cuda:0", primals_84: "f32[384][1]cuda:0", primals_85: "f32[1152, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_86: "f32[384, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_87: "i64[][]cuda:0", primals_88: "f32[384][1]cuda:0", primals_89: "f32[384][1]cuda:0", primals_90: "f32[384][1]cuda:0", primals_91: "f32[384][1]cuda:0", primals_92: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_93: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_94: "i64[][]cuda:0", primals_95: "f32[384][1]cuda:0", primals_96: "f32[384][1]cuda:0", primals_97: "f32[384][1]cuda:0", primals_98: "f32[384][1]cuda:0", primals_99: "f32[1152, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_100: "f32[384, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_101: "i64[][]cuda:0", primals_102: "f32[384][1]cuda:0", primals_103: "f32[384][1]cuda:0", primals_104: "f32[384][1]cuda:0", primals_105: "f32[384][1]cuda:0", primals_106: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_107: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_108: "i64[][]cuda:0", primals_109: "f32[384][1]cuda:0", primals_110: "f32[384][1]cuda:0", primals_111: "f32[384][1]cuda:0", primals_112: "f32[384][1]cuda:0", primals_113: "f32[1152, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_114: "f32[384, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_115: "i64[][]cuda:0", primals_116: "f32[384][1]cuda:0", primals_117: "f32[384][1]cuda:0", primals_118: "f32[384][1]cuda:0", primals_119: "f32[384][1]cuda:0", primals_120: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_121: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_122: "i64[][]cuda:0", primals_123: "f32[384][1]cuda:0", primals_124: "f32[384][1]cuda:0", primals_125: "f32[384][1]cuda:0", primals_126: "f32[384][1]cuda:0", primals_127: "f32[1152, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_128: "f32[384, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_129: "i64[][]cuda:0", primals_130: "f32[384][1]cuda:0", primals_131: "f32[384][1]cuda:0", primals_132: "f32[384][1]cuda:0", primals_133: "f32[384][1]cuda:0", primals_134: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_135: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_136: "f32[768, 384, 2, 2][1536, 1, 768, 384]cuda:0", primals_137: "f32[768][1]cuda:0", primals_138: "i64[][]cuda:0", primals_139: "f32[768][1]cuda:0", primals_140: "f32[768][1]cuda:0", primals_141: "f32[768][1]cuda:0", primals_142: "f32[768][1]cuda:0", primals_143: "f32[1, 768, 7, 7][37632, 1, 5376, 768]cuda:0", primals_144: "i64[][]cuda:0", primals_145: "f32[768][1]cuda:0", primals_146: "f32[768][1]cuda:0", primals_147: "f32[768][1]cuda:0", primals_148: "f32[768][1]cuda:0", primals_149: "f32[2304, 768, 1, 1][768, 1, 768, 768]cuda:0", primals_150: "f32[768, 768, 1, 1][768, 1, 768, 768]cuda:0", primals_151: "i64[][]cuda:0", primals_152: "f32[768][1]cuda:0", primals_153: "f32[768][1]cuda:0", primals_154: "f32[768][1]cuda:0", primals_155: "f32[768][1]cuda:0", primals_156: "f32[3072, 768, 1, 1][768, 1, 768, 768]cuda:0", primals_157: "f32[768, 3072, 1, 1][3072, 1, 3072, 3072]cuda:0", primals_158: "i64[][]cuda:0", primals_159: "f32[768][1]cuda:0", primals_160: "f32[768][1]cuda:0", primals_161: "f32[768][1]cuda:0", primals_162: "f32[768][1]cuda:0", primals_163: "f32[2304, 768, 1, 1][768, 1, 768, 768]cuda:0", primals_164: "f32[768, 768, 1, 1][768, 1, 768, 768]cuda:0", primals_165: "i64[][]cuda:0", primals_166: "f32[768][1]cuda:0", primals_167: "f32[768][1]cuda:0", primals_168: "f32[768][1]cuda:0", primals_169: "f32[768][1]cuda:0", primals_170: "f32[3072, 768, 1, 1][768, 1, 768, 768]cuda:0", primals_171: "f32[768, 3072, 1, 1][3072, 1, 3072, 3072]cuda:0", primals_172: "i64[][]cuda:0", primals_173: "f32[768][1]cuda:0", primals_174: "f32[768][1]cuda:0", primals_175: "f32[768][1]cuda:0", primals_176: "f32[768][1]cuda:0", primals_177: "f32[2304, 768, 1, 1][768, 1, 768, 768]cuda:0", primals_178: "f32[768, 768, 1, 1][768, 1, 768, 768]cuda:0", primals_179: "i64[][]cuda:0", primals_180: "f32[768][1]cuda:0", primals_181: "f32[768][1]cuda:0", primals_182: "f32[768][1]cuda:0", primals_183: "f32[768][1]cuda:0", primals_184: "f32[3072, 768, 1, 1][768, 1, 768, 768]cuda:0", primals_185: "f32[768, 3072, 1, 1][3072, 1, 3072, 3072]cuda:0", primals_186: "i64[][]cuda:0", primals_187: "f32[768][1]cuda:0", primals_188: "f32[768][1]cuda:0", primals_189: "f32[768][1]cuda:0", primals_190: "f32[768][1]cuda:0", primals_191: "f32[2304, 768, 1, 1][768, 1, 768, 768]cuda:0", primals_192: "f32[768, 768, 1, 1][768, 1, 768, 768]cuda:0", primals_193: "i64[][]cuda:0", primals_194: "f32[768][1]cuda:0", primals_195: "f32[768][1]cuda:0", primals_196: "f32[768][1]cuda:0", primals_197: "f32[768][1]cuda:0", primals_198: "f32[3072, 768, 1, 1][768, 1, 768, 768]cuda:0", primals_199: "f32[768, 3072, 1, 1][3072, 1, 3072, 3072]cuda:0", primals_200: "i64[][]cuda:0", primals_201: "f32[768][1]cuda:0", primals_202: "f32[768][1]cuda:0", primals_203: "f32[768][1]cuda:0", primals_204: "f32[768][1]cuda:0", primals_205: "f32[1000, 768][768, 1]cuda:0", primals_206: "f32[1000][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:437 in forward_features, code: x = self.stem(x)
        convert_element_type: "bf16[32, 3, 7, 7][147, 1, 21, 3]cuda:0" = torch.ops.prims.convert_element_type.default(primals_1, torch.bfloat16);  primals_1 = None
        convert_element_type_1: "bf16[128, 3, 224, 224][150528, 1, 672, 3]cuda:0" = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16);  primals_2 = None
        convolution: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_1, convert_element_type, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1)
        add: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_3, 1)
        convert_element_type_2: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(convolution, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_2, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_2 = None
        getitem: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        add_1: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze: "f32[32][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_1: "f32[32][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_1: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
        mul_2: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_4, 0.9)
        add_2: "f32[32][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_2: "f32[32][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_3: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_2, 1.0000006228081046);  squeeze_2 = None
        mul_4: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
        mul_5: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_5, 0.9)
        add_3: "f32[32][1]cuda:0" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_6, -1)
        unsqueeze_1: "f32[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_7, -1);  primals_7 = None
        unsqueeze_3: "f32[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_4: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        convert_element_type_3: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None
        relu: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.relu.default(convert_element_type_3);  convert_element_type_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.proj(x)
        convert_element_type_4: "bf16[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_9, torch.bfloat16);  primals_9 = None
        convert_element_type_5: "bf16[192, 32, 4, 4][512, 1, 128, 32]cuda:0" = torch.ops.prims.convert_element_type.default(primals_8, torch.bfloat16);  primals_8 = None
        convolution_1: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.convolution.default(relu, convert_element_type_5, convert_element_type_4, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:141 in forward, code: x = self.norm(x)
        add_5: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_10, 1)
        convert_element_type_6: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_6, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_6 = None
        getitem_2: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_1[0]
        getitem_3: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        add_6: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
        rsqrt_1: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_1: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
        mul_7: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        squeeze_3: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3])
        mul_8: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1);  squeeze_3 = None
        mul_9: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_11, 0.9)
        add_7: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        squeeze_5: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_10: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_5, 1.00000996502277);  squeeze_5 = None
        mul_11: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
        mul_12: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_12, 0.9)
        add_8: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        unsqueeze_4: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_13, -1)
        unsqueeze_5: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_13: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
        unsqueeze_6: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_14, -1)
        unsqueeze_7: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_9: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
        convert_element_type_7: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_9, torch.bfloat16);  add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:442 in forward_features, code: x = self.pos_drop(x + self.pos_embed1)
        add_10: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_7, primals_15);  convert_element_type_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_11: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_16, 1)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_10, [0, 2, 3], correction = 0, keepdim = True)
        getitem_4: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_2[0]
        getitem_5: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        add_12: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
        rsqrt_2: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        sub_2: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(add_10, getitem_5)
        mul_14: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        squeeze_6: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        squeeze_7: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_15: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
        mul_16: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_17, 0.9)
        add_13: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
        squeeze_8: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_17: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_8, 1.00000996502277);  squeeze_8 = None
        mul_18: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
        mul_19: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_18, 0.9)
        add_14: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
        unsqueeze_8: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_19, -1)
        unsqueeze_9: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        mul_20: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
        unsqueeze_10: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_20, -1);  primals_20 = None
        unsqueeze_11: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        add_15: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.conv1(x)
        convert_element_type_8: "bf16[384, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(primals_21, torch.bfloat16);  primals_21 = None
        convert_element_type_9: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_15, torch.bfloat16);  add_15 = None
        convolution_2: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_9, convert_element_type_8, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.act1(x)
        convert_element_type_10: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_2, torch.float32)
        mul_21: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_10, 0.5)
        mul_22: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_10, 0.7071067811865476);  convert_element_type_10 = None
        erf: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.erf.default(mul_22);  mul_22 = None
        add_16: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_23: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_21, add_16);  mul_21 = add_16 = None
        convert_element_type_11: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_23, torch.bfloat16);  mul_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:71 in forward, code: x = self.conv2(x)
        convert_element_type_12: "bf16[384, 48, 3, 3][432, 1, 144, 48]cuda:0" = torch.ops.prims.convert_element_type.default(primals_22, torch.bfloat16);  primals_22 = None
        convolution_3: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_11, convert_element_type_12, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:72 in forward, code: x = self.act2(x)
        convert_element_type_13: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_3, torch.float32)
        mul_24: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_13, 0.5)
        mul_25: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_13, 0.7071067811865476);  convert_element_type_13 = None
        erf_1: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.erf.default(mul_25);  mul_25 = None
        add_17: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_26: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, add_17);  mul_24 = add_17 = None
        convert_element_type_14: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_26, torch.bfloat16);  mul_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:73 in forward, code: x = self.conv3(x)
        convert_element_type_15: "bf16[192, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_23, torch.bfloat16);  primals_23 = None
        convolution_4: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_14, convert_element_type_15, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_18: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(add_10, convolution_4);  add_10 = convolution_4 = None
        add_19: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_24, 1)
        var_mean_3 = torch.ops.aten.var_mean.correction(add_18, [0, 2, 3], correction = 0, keepdim = True)
        getitem_6: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_3[0]
        getitem_7: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        add_20: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
        rsqrt_3: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        sub_3: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(add_18, getitem_7)
        mul_27: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        squeeze_9: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        squeeze_10: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_28: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
        mul_29: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_25, 0.9)
        add_21: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_28, mul_29);  mul_28 = mul_29 = None
        squeeze_11: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_30: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_11, 1.00000996502277);  squeeze_11 = None
        mul_31: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_30, 0.1);  mul_30 = None
        mul_32: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_26, 0.9)
        add_22: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_31, mul_32);  mul_31 = mul_32 = None
        unsqueeze_12: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_27, -1)
        unsqueeze_13: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_33: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_27, unsqueeze_13);  mul_27 = unsqueeze_13 = None
        unsqueeze_14: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_28, -1);  primals_28 = None
        unsqueeze_15: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_23: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_33, unsqueeze_15);  mul_33 = unsqueeze_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.conv1(x)
        convert_element_type_16: "bf16[384, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(primals_29, torch.bfloat16);  primals_29 = None
        convert_element_type_17: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_23, torch.bfloat16);  add_23 = None
        convolution_5: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_17, convert_element_type_16, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.act1(x)
        convert_element_type_18: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_5, torch.float32)
        mul_34: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_18, 0.5)
        mul_35: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_18, 0.7071067811865476);  convert_element_type_18 = None
        erf_2: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.erf.default(mul_35);  mul_35 = None
        add_24: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_36: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_34, add_24);  mul_34 = add_24 = None
        convert_element_type_19: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_36, torch.bfloat16);  mul_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:71 in forward, code: x = self.conv2(x)
        convert_element_type_20: "bf16[384, 48, 3, 3][432, 1, 144, 48]cuda:0" = torch.ops.prims.convert_element_type.default(primals_30, torch.bfloat16);  primals_30 = None
        convolution_6: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_19, convert_element_type_20, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:72 in forward, code: x = self.act2(x)
        convert_element_type_21: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_6, torch.float32)
        mul_37: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_21, 0.5)
        mul_38: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_21, 0.7071067811865476);  convert_element_type_21 = None
        erf_3: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.erf.default(mul_38);  mul_38 = None
        add_25: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_39: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_37, add_25);  mul_37 = add_25 = None
        convert_element_type_22: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_39, torch.bfloat16);  mul_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:73 in forward, code: x = self.conv3(x)
        convert_element_type_23: "bf16[192, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_31, torch.bfloat16);  primals_31 = None
        convolution_7: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_22, convert_element_type_23, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_26: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(add_18, convolution_7);  convolution_7 = None
        add_27: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_32, 1)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_26, [0, 2, 3], correction = 0, keepdim = True)
        getitem_8: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_4[0]
        getitem_9: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        add_28: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
        rsqrt_4: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_4: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(add_26, getitem_9)
        mul_40: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        squeeze_12: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        squeeze_13: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_41: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
        mul_42: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_33, 0.9)
        add_29: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_41, mul_42);  mul_41 = mul_42 = None
        squeeze_14: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_43: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_14, 1.00000996502277);  squeeze_14 = None
        mul_44: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_43, 0.1);  mul_43 = None
        mul_45: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_34, 0.9)
        add_30: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_44, mul_45);  mul_44 = mul_45 = None
        unsqueeze_16: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_35, -1)
        unsqueeze_17: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        mul_46: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_40, unsqueeze_17);  mul_40 = unsqueeze_17 = None
        unsqueeze_18: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
        unsqueeze_19: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        add_31: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_46, unsqueeze_19);  mul_46 = unsqueeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.conv1(x)
        convert_element_type_24: "bf16[384, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(primals_37, torch.bfloat16);  primals_37 = None
        convert_element_type_25: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_31, torch.bfloat16);  add_31 = None
        convolution_8: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_25, convert_element_type_24, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.act1(x)
        convert_element_type_26: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_8, torch.float32)
        mul_47: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_26, 0.5)
        mul_48: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_26, 0.7071067811865476);  convert_element_type_26 = None
        erf_4: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
        add_32: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_49: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_47, add_32);  mul_47 = add_32 = None
        convert_element_type_27: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_49, torch.bfloat16);  mul_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:71 in forward, code: x = self.conv2(x)
        convert_element_type_28: "bf16[384, 48, 3, 3][432, 1, 144, 48]cuda:0" = torch.ops.prims.convert_element_type.default(primals_38, torch.bfloat16);  primals_38 = None
        convolution_9: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_27, convert_element_type_28, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:72 in forward, code: x = self.act2(x)
        convert_element_type_29: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_9, torch.float32)
        mul_50: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_29, 0.5)
        mul_51: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_29, 0.7071067811865476);  convert_element_type_29 = None
        erf_5: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
        add_33: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_52: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_50, add_33);  mul_50 = add_33 = None
        convert_element_type_30: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_52, torch.bfloat16);  mul_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:73 in forward, code: x = self.conv3(x)
        convert_element_type_31: "bf16[192, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_39, torch.bfloat16);  primals_39 = None
        convolution_10: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_30, convert_element_type_31, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_34: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(add_26, convolution_10);  convolution_10 = None
        add_35: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_40, 1)
        var_mean_5 = torch.ops.aten.var_mean.correction(add_34, [0, 2, 3], correction = 0, keepdim = True)
        getitem_10: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_5[0]
        getitem_11: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        add_36: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
        rsqrt_5: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_5: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(add_34, getitem_11)
        mul_53: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        squeeze_15: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        squeeze_16: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_54: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
        mul_55: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_41, 0.9)
        add_37: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_54, mul_55);  mul_54 = mul_55 = None
        squeeze_17: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_56: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_17, 1.00000996502277);  squeeze_17 = None
        mul_57: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_56, 0.1);  mul_56 = None
        mul_58: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_42, 0.9)
        add_38: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
        unsqueeze_20: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_43, -1)
        unsqueeze_21: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_59: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_53, unsqueeze_21);  mul_53 = unsqueeze_21 = None
        unsqueeze_22: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_44, -1);  primals_44 = None
        unsqueeze_23: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_39: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_59, unsqueeze_23);  mul_59 = unsqueeze_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.conv1(x)
        convert_element_type_32: "bf16[384, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(primals_45, torch.bfloat16);  primals_45 = None
        convert_element_type_33: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_39, torch.bfloat16);  add_39 = None
        convolution_11: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_33, convert_element_type_32, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.act1(x)
        convert_element_type_34: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_11, torch.float32)
        mul_60: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_34, 0.5)
        mul_61: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_34, 0.7071067811865476);  convert_element_type_34 = None
        erf_6: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
        add_40: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_62: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_60, add_40);  mul_60 = add_40 = None
        convert_element_type_35: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_62, torch.bfloat16);  mul_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:71 in forward, code: x = self.conv2(x)
        convert_element_type_36: "bf16[384, 48, 3, 3][432, 1, 144, 48]cuda:0" = torch.ops.prims.convert_element_type.default(primals_46, torch.bfloat16);  primals_46 = None
        convolution_12: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_35, convert_element_type_36, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:72 in forward, code: x = self.act2(x)
        convert_element_type_37: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_12, torch.float32)
        mul_63: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_37, 0.5)
        mul_64: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_37, 0.7071067811865476);  convert_element_type_37 = None
        erf_7: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.erf.default(mul_64);  mul_64 = None
        add_41: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_65: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_63, add_41);  mul_63 = add_41 = None
        convert_element_type_38: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_65, torch.bfloat16);  mul_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:73 in forward, code: x = self.conv3(x)
        convert_element_type_39: "bf16[192, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_47, torch.bfloat16);  primals_47 = None
        convolution_13: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_38, convert_element_type_39, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_42: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(add_34, convolution_13);  convolution_13 = None
        add_43: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_48, 1)
        var_mean_6 = torch.ops.aten.var_mean.correction(add_42, [0, 2, 3], correction = 0, keepdim = True)
        getitem_12: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_6[0]
        getitem_13: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        add_44: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
        rsqrt_6: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_6: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(add_42, getitem_13)
        mul_66: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        squeeze_18: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        squeeze_19: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_67: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
        mul_68: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_49, 0.9)
        add_45: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
        squeeze_20: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
        mul_69: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_20, 1.00000996502277);  squeeze_20 = None
        mul_70: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_69, 0.1);  mul_69 = None
        mul_71: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_50, 0.9)
        add_46: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_70, mul_71);  mul_70 = mul_71 = None
        unsqueeze_24: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_51, -1)
        unsqueeze_25: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        mul_72: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, unsqueeze_25);  mul_66 = unsqueeze_25 = None
        unsqueeze_26: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_52, -1);  primals_52 = None
        unsqueeze_27: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        add_47: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_72, unsqueeze_27);  mul_72 = unsqueeze_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.conv1(x)
        convert_element_type_40: "bf16[384, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(primals_53, torch.bfloat16);  primals_53 = None
        convert_element_type_41: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_47, torch.bfloat16);  add_47 = None
        convolution_14: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_41, convert_element_type_40, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.act1(x)
        convert_element_type_42: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_14, torch.float32)
        mul_73: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_42, 0.5)
        mul_74: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_42, 0.7071067811865476);  convert_element_type_42 = None
        erf_8: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.erf.default(mul_74);  mul_74 = None
        add_48: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_75: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_73, add_48);  mul_73 = add_48 = None
        convert_element_type_43: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_75, torch.bfloat16);  mul_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:71 in forward, code: x = self.conv2(x)
        convert_element_type_44: "bf16[384, 48, 3, 3][432, 1, 144, 48]cuda:0" = torch.ops.prims.convert_element_type.default(primals_54, torch.bfloat16);  primals_54 = None
        convolution_15: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_43, convert_element_type_44, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:72 in forward, code: x = self.act2(x)
        convert_element_type_45: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_15, torch.float32)
        mul_76: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_45, 0.5)
        mul_77: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_45, 0.7071067811865476);  convert_element_type_45 = None
        erf_9: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.erf.default(mul_77);  mul_77 = None
        add_49: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_78: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_76, add_49);  mul_76 = add_49 = None
        convert_element_type_46: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_78, torch.bfloat16);  mul_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:73 in forward, code: x = self.conv3(x)
        convert_element_type_47: "bf16[192, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_55, torch.bfloat16);  primals_55 = None
        convolution_16: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_46, convert_element_type_47, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_50: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(add_42, convolution_16);  convolution_16 = None
        add_51: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_56, 1)
        var_mean_7 = torch.ops.aten.var_mean.correction(add_50, [0, 2, 3], correction = 0, keepdim = True)
        getitem_14: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_7[0]
        getitem_15: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        add_52: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
        rsqrt_7: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        sub_7: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(add_50, getitem_15)
        mul_79: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        squeeze_21: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        squeeze_22: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
        mul_80: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
        mul_81: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_57, 0.9)
        add_53: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_80, mul_81);  mul_80 = mul_81 = None
        squeeze_23: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
        mul_82: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_23, 1.00000996502277);  squeeze_23 = None
        mul_83: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_82, 0.1);  mul_82 = None
        mul_84: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_58, 0.9)
        add_54: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_83, mul_84);  mul_83 = mul_84 = None
        unsqueeze_28: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_59, -1)
        unsqueeze_29: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_85: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_79, unsqueeze_29);  mul_79 = unsqueeze_29 = None
        unsqueeze_30: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
        unsqueeze_31: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_55: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_85, unsqueeze_31);  mul_85 = unsqueeze_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.conv1(x)
        convert_element_type_48: "bf16[384, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(primals_61, torch.bfloat16);  primals_61 = None
        convert_element_type_49: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_55, torch.bfloat16);  add_55 = None
        convolution_17: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_49, convert_element_type_48, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.act1(x)
        convert_element_type_50: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_17, torch.float32)
        mul_86: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_50, 0.5)
        mul_87: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_50, 0.7071067811865476);  convert_element_type_50 = None
        erf_10: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
        add_56: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_88: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_86, add_56);  mul_86 = add_56 = None
        convert_element_type_51: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_88, torch.bfloat16);  mul_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:71 in forward, code: x = self.conv2(x)
        convert_element_type_52: "bf16[384, 48, 3, 3][432, 1, 144, 48]cuda:0" = torch.ops.prims.convert_element_type.default(primals_62, torch.bfloat16);  primals_62 = None
        convolution_18: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_51, convert_element_type_52, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:72 in forward, code: x = self.act2(x)
        convert_element_type_53: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_18, torch.float32)
        mul_89: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_53, 0.5)
        mul_90: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_53, 0.7071067811865476);  convert_element_type_53 = None
        erf_11: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.erf.default(mul_90);  mul_90 = None
        add_57: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_91: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_89, add_57);  mul_89 = add_57 = None
        convert_element_type_54: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_91, torch.bfloat16);  mul_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:73 in forward, code: x = self.conv3(x)
        convert_element_type_55: "bf16[192, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_63, torch.bfloat16);  primals_63 = None
        convolution_19: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_54, convert_element_type_55, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_58: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(add_50, convolution_19);  convolution_19 = None
        add_59: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_64, 1)
        var_mean_8 = torch.ops.aten.var_mean.correction(add_58, [0, 2, 3], correction = 0, keepdim = True)
        getitem_16: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_8[0]
        getitem_17: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        add_60: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
        rsqrt_8: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_8: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(add_58, getitem_17)
        mul_92: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        squeeze_24: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
        squeeze_25: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_93: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
        mul_94: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_65, 0.9)
        add_61: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_93, mul_94);  mul_93 = mul_94 = None
        squeeze_26: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
        mul_95: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_26, 1.00000996502277);  squeeze_26 = None
        mul_96: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_95, 0.1);  mul_95 = None
        mul_97: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_66, 0.9)
        add_62: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_96, mul_97);  mul_96 = mul_97 = None
        unsqueeze_32: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_67, -1)
        unsqueeze_33: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        mul_98: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_92, unsqueeze_33);  mul_92 = unsqueeze_33 = None
        unsqueeze_34: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_68, -1);  primals_68 = None
        unsqueeze_35: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        add_63: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_98, unsqueeze_35);  mul_98 = unsqueeze_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.conv1(x)
        convert_element_type_56: "bf16[384, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(primals_69, torch.bfloat16);  primals_69 = None
        convert_element_type_57: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_63, torch.bfloat16);  add_63 = None
        convolution_20: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_57, convert_element_type_56, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.act1(x)
        convert_element_type_58: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_20, torch.float32)
        mul_99: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_58, 0.5)
        mul_100: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_58, 0.7071067811865476);  convert_element_type_58 = None
        erf_12: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.erf.default(mul_100);  mul_100 = None
        add_64: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_101: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_99, add_64);  mul_99 = add_64 = None
        convert_element_type_59: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_101, torch.bfloat16);  mul_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:71 in forward, code: x = self.conv2(x)
        convert_element_type_60: "bf16[384, 48, 3, 3][432, 1, 144, 48]cuda:0" = torch.ops.prims.convert_element_type.default(primals_70, torch.bfloat16);  primals_70 = None
        convolution_21: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_59, convert_element_type_60, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:72 in forward, code: x = self.act2(x)
        convert_element_type_61: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_21, torch.float32)
        mul_102: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_61, 0.5)
        mul_103: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_61, 0.7071067811865476);  convert_element_type_61 = None
        erf_13: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.erf.default(mul_103);  mul_103 = None
        add_65: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_104: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_102, add_65);  mul_102 = add_65 = None
        convert_element_type_62: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_104, torch.bfloat16);  mul_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:73 in forward, code: x = self.conv3(x)
        convert_element_type_63: "bf16[192, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_71, torch.bfloat16);  primals_71 = None
        convolution_22: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_62, convert_element_type_63, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_66: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(add_58, convolution_22);  convolution_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.proj(x)
        convert_element_type_64: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_73, torch.bfloat16);  primals_73 = None
        convert_element_type_65: "bf16[384, 192, 2, 2][768, 1, 384, 192]cuda:0" = torch.ops.prims.convert_element_type.default(primals_72, torch.bfloat16);  primals_72 = None
        convert_element_type_66: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_66, torch.bfloat16);  add_66 = None
        convolution_23: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_66, convert_element_type_65, convert_element_type_64, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:141 in forward, code: x = self.norm(x)
        add_67: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_74, 1)
        convert_element_type_67: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_23, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_67, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_67 = None
        getitem_18: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_9[0]
        getitem_19: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        add_68: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
        rsqrt_9: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        sub_9: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_23, getitem_19)
        mul_105: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        squeeze_27: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3])
        mul_106: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1);  squeeze_27 = None
        mul_107: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_75, 0.9)
        add_69: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
        squeeze_29: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
        mul_108: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0000398612827361);  squeeze_29 = None
        mul_109: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_108, 0.1);  mul_108 = None
        mul_110: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_76, 0.9)
        add_70: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
        unsqueeze_36: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_77, -1)
        unsqueeze_37: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_111: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_37);  mul_105 = unsqueeze_37 = None
        unsqueeze_38: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_78, -1)
        unsqueeze_39: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_71: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_39);  mul_111 = unsqueeze_39 = None
        convert_element_type_68: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_71, torch.bfloat16);  add_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:452 in forward_features, code: x = self.pos_drop(x + self.pos_embed2)
        add_72: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_68, primals_79);  convert_element_type_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        add_73: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_80, 1)
        var_mean_10 = torch.ops.aten.var_mean.correction(add_72, [0, 2, 3], correction = 0, keepdim = True)
        getitem_20: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_10[0]
        getitem_21: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        add_74: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
        rsqrt_10: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_10: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(add_72, getitem_21)
        mul_112: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        squeeze_30: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        squeeze_31: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_113: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
        mul_114: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_81, 0.9)
        add_75: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
        squeeze_32: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        mul_115: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0000398612827361);  squeeze_32 = None
        mul_116: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_115, 0.1);  mul_115 = None
        mul_117: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_82, 0.9)
        add_76: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
        unsqueeze_40: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_83, -1)
        unsqueeze_41: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        mul_118: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_41);  mul_112 = unsqueeze_41 = None
        unsqueeze_42: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_84, -1);  primals_84 = None
        unsqueeze_43: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        add_77: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_43);  mul_118 = unsqueeze_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:107 in forward, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
        convert_element_type_69: "bf16[1152, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_85, torch.bfloat16);  primals_85 = None
        convert_element_type_70: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_77, torch.bfloat16);  add_77 = None
        convolution_24: "bf16[128, 1152, 14, 14][225792, 1, 16128, 1152]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_70, convert_element_type_69, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        view: "bf16[128, 3, 6, 64, 196][225792, 384, 64, 1, 1152]cuda:0" = torch.ops.aten.reshape.default(convolution_24, [128, 3, 6, 64, 196]);  convolution_24 = None
        permute: "bf16[3, 128, 6, 196, 64][384, 225792, 64, 1152, 1]cuda:0" = torch.ops.aten.permute.default(view, [1, 0, 2, 4, 3]);  view = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:108 in forward, code: q, k, v = x.unbind(0)
        unbind = torch.ops.aten.unbind.int(permute);  permute = None
        getitem_22: "bf16[128, 6, 196, 64][225792, 64, 1152, 1]cuda:0" = unbind[0]
        getitem_23: "bf16[128, 6, 196, 64][225792, 64, 1152, 1]cuda:0" = unbind[1]
        getitem_24: "bf16[128, 6, 196, 64][225792, 64, 1152, 1]cuda:0" = unbind[2];  unbind = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_1: "bf16[128, 6, 64, 196][225792, 64, 1, 1152]cuda:0" = torch.ops.aten.permute.default(getitem_23, [0, 1, 3, 2]);  getitem_23 = None
        expand: "bf16[128, 6, 196, 64][225792, 64, 1152, 1]cuda:0" = torch.ops.aten.expand.default(getitem_22, [128, 6, 196, 64]);  getitem_22 = None
        clone_16: "bf16[128, 6, 196, 64][75264, 12544, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_1: "bf16[768, 196, 64][12544, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_16, [768, 196, 64]);  clone_16 = None
        expand_1: "bf16[128, 6, 64, 196][225792, 64, 1, 1152]cuda:0" = torch.ops.aten.expand.default(permute_1, [128, 6, 64, 196]);  permute_1 = None
        clone_17: "bf16[128, 6, 64, 196][75264, 12544, 196, 1]cuda:0" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_2: "bf16[768, 64, 196][12544, 196, 1]cuda:0" = torch.ops.aten.reshape.default(clone_17, [768, 64, 196]);  clone_17 = None
        constant_pad_nd_default_22: "bf16[768, 200, 64][12800, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_1, [0, 0, 0, 4, 0, 0])
        constant_pad_nd_default_23: "bf16[768, 64, 200][12800, 200, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_2, [0, 4, 0, 0, 0, 0])
        bmm_default_11: "bf16[768, 200, 200][40000, 200, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_22, constant_pad_nd_default_23);  constant_pad_nd_default_22 = constant_pad_nd_default_23 = None
        slice_tensor_18: "bf16[768, 196, 200][40000, 200, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_11, 1, 0, -4)
        slice_tensor_19: "bf16[768, 196, 196][40000, 200, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_18, 2, 0, -4);  slice_tensor_18 = None
        view_3: "bf16[128, 6, 196, 196][240000, 40000, 200, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_19, [128, 6, 196, 196]);  slice_tensor_19 = None

        # No stacktrace found for following nodes
        mul_tensor_28: "bf16[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_3, 0.125)
        convert_element_type_default_14: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_28, torch.float32);  mul_tensor_28 = None
        convert_element_type_default_15: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        mul_tensor_29: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_15, 1);  convert_element_type_default_15 = None
        amax_default_14: "f32[128, 6, 196, 1][1176, 196, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_29, [-1], True)
        sub_tensor_14: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_29, amax_default_14);  mul_tensor_29 = None
        mul_tensor_30: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_14, 0.125);  sub_tensor_14 = None
        amax_default_15: "f32[128, 6, 196, 1][1176, 196, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_14, [-1], True)
        sub_tensor_15: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_14, amax_default_15)
        abs_default_7: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_14)
        ne_scalar_7: "b8[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_7, inf);  abs_default_7 = None
        eq_tensor_7: "b8[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_14, convert_element_type_default_14);  convert_element_type_default_14 = None
        mul_tensor_31: "b8[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_7, ne_scalar_7);  eq_tensor_7 = ne_scalar_7 = None
        logical_not_default_14: "b8[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_31);  mul_tensor_31 = None
        any_dims_7: "b8[128, 6, 196, 1][1176, 196, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_14, [-1], True);  logical_not_default_14 = None
        logical_not_default_15: "b8[128, 6, 196, 1][1176, 196, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_7);  any_dims_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:117 in forward, code: attn = attn.softmax(dim=-1)
        where_self_7: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_15, mul_tensor_30, sub_tensor_15);  mul_tensor_30 = sub_tensor_15 = None
        exp: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.exp.default(where_self_7);  where_self_7 = None
        sum_1: "f32[128, 6, 196, 1][1176, 196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        convert_element_type_74: "bf16[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        expand_2: "bf16[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_74, [128, 6, 196, 196]);  convert_element_type_74 = None
        view_4: "bf16[768, 196, 196][38416, 196, 1]cuda:0" = torch.ops.aten.reshape.default(expand_2, [768, 196, 196]);  expand_2 = None
        expand_3: "bf16[128, 6, 196, 64][225792, 64, 1152, 1]cuda:0" = torch.ops.aten.expand.default(getitem_24, [128, 6, 196, 64]);  getitem_24 = None
        clone_19: "bf16[128, 6, 196, 64][75264, 12544, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_5: "bf16[768, 196, 64][12544, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_19, [768, 196, 64]);  clone_19 = None
        constant_pad_nd_default_20: "bf16[768, 200, 200][40000, 200, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_4, [0, 4, 0, 4, 0, 0])
        constant_pad_nd_default_21: "bf16[768, 200, 64][12800, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_5, [0, 0, 0, 4, 0, 0])
        bmm_default_10: "bf16[768, 200, 64][12800, 64, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_20, constant_pad_nd_default_21);  constant_pad_nd_default_20 = constant_pad_nd_default_21 = None
        slice_tensor_17: "bf16[768, 196, 64][12800, 64, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_10, 1, 0, -4);  bmm_default_10 = None
        view_6: "bf16[128, 6, 196, 64][76800, 12800, 64, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_17, [128, 6, 196, 64]);  slice_tensor_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:121 in forward, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
        permute_2: "bf16[128, 6, 64, 196][76800, 12800, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_6, [0, 1, 3, 2]);  view_6 = None
        clone_20: "bf16[128, 6, 64, 196][75264, 12544, 196, 1]cuda:0" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_7: "bf16[128, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(clone_20, [128, 384, 14, 14]);  clone_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:122 in forward, code: x = self.proj(x)
        convert_element_type_77: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_86, torch.bfloat16);  primals_86 = None
        convolution_25: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(view_7, convert_element_type_77, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        add_78: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_72, convolution_25);  add_72 = convolution_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_79: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_87, 1)
        var_mean_11 = torch.ops.aten.var_mean.correction(add_78, [0, 2, 3], correction = 0, keepdim = True)
        getitem_25: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_11[0]
        getitem_26: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        add_80: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_25, 1e-05)
        rsqrt_11: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        sub_12: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(add_78, getitem_26)
        mul_120: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_11);  sub_12 = None
        squeeze_33: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        squeeze_34: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
        mul_121: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
        mul_122: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_88, 0.9)
        add_81: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_121, mul_122);  mul_121 = mul_122 = None
        squeeze_35: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        mul_123: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0000398612827361);  squeeze_35 = None
        mul_124: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_123, 0.1);  mul_123 = None
        mul_125: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_89, 0.9)
        add_82: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_124, mul_125);  mul_124 = mul_125 = None
        unsqueeze_44: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_90, -1)
        unsqueeze_45: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_126: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_120, unsqueeze_45);  mul_120 = unsqueeze_45 = None
        unsqueeze_46: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_91, -1);  primals_91 = None
        unsqueeze_47: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_83: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_126, unsqueeze_47);  mul_126 = unsqueeze_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.conv1(x)
        convert_element_type_78: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_92, torch.bfloat16);  primals_92 = None
        convert_element_type_79: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_83, torch.bfloat16);  add_83 = None
        convolution_26: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_79, convert_element_type_78, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.act1(x)
        convert_element_type_80: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_26, torch.float32)
        mul_127: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_80, 0.5)
        mul_128: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_80, 0.7071067811865476);  convert_element_type_80 = None
        erf_14: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.erf.default(mul_128);  mul_128 = None
        add_84: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_129: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_127, add_84);  mul_127 = add_84 = None
        convert_element_type_81: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(mul_129, torch.bfloat16);  mul_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:73 in forward, code: x = self.conv3(x)
        convert_element_type_82: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(primals_93, torch.bfloat16);  primals_93 = None
        convolution_27: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_81, convert_element_type_82, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_85: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_78, convolution_27);  convolution_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        add_86: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_94, 1)
        var_mean_12 = torch.ops.aten.var_mean.correction(add_85, [0, 2, 3], correction = 0, keepdim = True)
        getitem_27: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_12[0]
        getitem_28: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        add_87: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_27, 1e-05)
        rsqrt_12: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
        sub_13: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(add_85, getitem_28)
        mul_130: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_12);  sub_13 = None
        squeeze_36: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        squeeze_37: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_131: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
        mul_132: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_95, 0.9)
        add_88: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_131, mul_132);  mul_131 = mul_132 = None
        squeeze_38: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        mul_133: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0000398612827361);  squeeze_38 = None
        mul_134: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_133, 0.1);  mul_133 = None
        mul_135: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_96, 0.9)
        add_89: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
        unsqueeze_48: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_97, -1)
        unsqueeze_49: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        mul_136: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_130, unsqueeze_49);  mul_130 = unsqueeze_49 = None
        unsqueeze_50: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_98, -1);  primals_98 = None
        unsqueeze_51: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        add_90: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_136, unsqueeze_51);  mul_136 = unsqueeze_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:107 in forward, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
        convert_element_type_83: "bf16[1152, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_99, torch.bfloat16);  primals_99 = None
        convert_element_type_84: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_90, torch.bfloat16);  add_90 = None
        convolution_28: "bf16[128, 1152, 14, 14][225792, 1, 16128, 1152]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_84, convert_element_type_83, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        view_8: "bf16[128, 3, 6, 64, 196][225792, 384, 64, 1, 1152]cuda:0" = torch.ops.aten.reshape.default(convolution_28, [128, 3, 6, 64, 196]);  convolution_28 = None
        permute_3: "bf16[3, 128, 6, 196, 64][384, 225792, 64, 1152, 1]cuda:0" = torch.ops.aten.permute.default(view_8, [1, 0, 2, 4, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:108 in forward, code: q, k, v = x.unbind(0)
        unbind_1 = torch.ops.aten.unbind.int(permute_3);  permute_3 = None
        getitem_29: "bf16[128, 6, 196, 64][225792, 64, 1152, 1]cuda:0" = unbind_1[0]
        getitem_30: "bf16[128, 6, 196, 64][225792, 64, 1152, 1]cuda:0" = unbind_1[1]
        getitem_31: "bf16[128, 6, 196, 64][225792, 64, 1152, 1]cuda:0" = unbind_1[2];  unbind_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_4: "bf16[128, 6, 64, 196][225792, 64, 1, 1152]cuda:0" = torch.ops.aten.permute.default(getitem_30, [0, 1, 3, 2]);  getitem_30 = None
        expand_4: "bf16[128, 6, 196, 64][225792, 64, 1152, 1]cuda:0" = torch.ops.aten.expand.default(getitem_29, [128, 6, 196, 64]);  getitem_29 = None
        clone_24: "bf16[128, 6, 196, 64][75264, 12544, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_9: "bf16[768, 196, 64][12544, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_24, [768, 196, 64]);  clone_24 = None
        expand_5: "bf16[128, 6, 64, 196][225792, 64, 1, 1152]cuda:0" = torch.ops.aten.expand.default(permute_4, [128, 6, 64, 196]);  permute_4 = None
        clone_25: "bf16[128, 6, 64, 196][75264, 12544, 196, 1]cuda:0" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_10: "bf16[768, 64, 196][12544, 196, 1]cuda:0" = torch.ops.aten.reshape.default(clone_25, [768, 64, 196]);  clone_25 = None
        constant_pad_nd_default_18: "bf16[768, 200, 64][12800, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_9, [0, 0, 0, 4, 0, 0])
        constant_pad_nd_default_19: "bf16[768, 64, 200][12800, 200, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_10, [0, 4, 0, 0, 0, 0])
        bmm_default_9: "bf16[768, 200, 200][40000, 200, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_18, constant_pad_nd_default_19);  constant_pad_nd_default_18 = constant_pad_nd_default_19 = None
        slice_tensor_15: "bf16[768, 196, 200][40000, 200, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_9, 1, 0, -4)
        slice_tensor_16: "bf16[768, 196, 196][40000, 200, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_15, 2, 0, -4);  slice_tensor_15 = None
        view_11: "bf16[128, 6, 196, 196][240000, 40000, 200, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_16, [128, 6, 196, 196]);  slice_tensor_16 = None

        # No stacktrace found for following nodes
        mul_tensor_24: "bf16[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_11, 0.125)
        convert_element_type_default_12: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_24, torch.float32);  mul_tensor_24 = None
        convert_element_type_default_13: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_11, torch.float32);  view_11 = None
        mul_tensor_25: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_13, 1);  convert_element_type_default_13 = None
        amax_default_12: "f32[128, 6, 196, 1][1176, 196, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_25, [-1], True)
        sub_tensor_12: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_25, amax_default_12);  mul_tensor_25 = None
        mul_tensor_26: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_12, 0.125);  sub_tensor_12 = None
        amax_default_13: "f32[128, 6, 196, 1][1176, 196, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_12, [-1], True)
        sub_tensor_13: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_12, amax_default_13)
        abs_default_6: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_12)
        ne_scalar_6: "b8[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_6, inf);  abs_default_6 = None
        eq_tensor_6: "b8[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_12, convert_element_type_default_12);  convert_element_type_default_12 = None
        mul_tensor_27: "b8[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_6, ne_scalar_6);  eq_tensor_6 = ne_scalar_6 = None
        logical_not_default_12: "b8[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_27);  mul_tensor_27 = None
        any_dims_6: "b8[128, 6, 196, 1][1176, 196, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_12, [-1], True);  logical_not_default_12 = None
        logical_not_default_13: "b8[128, 6, 196, 1][1176, 196, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_6);  any_dims_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:117 in forward, code: attn = attn.softmax(dim=-1)
        where_self_6: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_13, mul_tensor_26, sub_tensor_13);  mul_tensor_26 = sub_tensor_13 = None
        exp_1: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.exp.default(where_self_6);  where_self_6 = None
        sum_2: "f32[128, 6, 196, 1][1176, 196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        convert_element_type_88: "bf16[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        expand_6: "bf16[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_88, [128, 6, 196, 196]);  convert_element_type_88 = None
        view_12: "bf16[768, 196, 196][38416, 196, 1]cuda:0" = torch.ops.aten.reshape.default(expand_6, [768, 196, 196]);  expand_6 = None
        expand_7: "bf16[128, 6, 196, 64][225792, 64, 1152, 1]cuda:0" = torch.ops.aten.expand.default(getitem_31, [128, 6, 196, 64]);  getitem_31 = None
        clone_27: "bf16[128, 6, 196, 64][75264, 12544, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_13: "bf16[768, 196, 64][12544, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_27, [768, 196, 64]);  clone_27 = None
        constant_pad_nd_default_16: "bf16[768, 200, 200][40000, 200, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_12, [0, 4, 0, 4, 0, 0])
        constant_pad_nd_default_17: "bf16[768, 200, 64][12800, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_13, [0, 0, 0, 4, 0, 0])
        bmm_default_8: "bf16[768, 200, 64][12800, 64, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_16, constant_pad_nd_default_17);  constant_pad_nd_default_16 = constant_pad_nd_default_17 = None
        slice_tensor_14: "bf16[768, 196, 64][12800, 64, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_8, 1, 0, -4);  bmm_default_8 = None
        view_14: "bf16[128, 6, 196, 64][76800, 12800, 64, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_14, [128, 6, 196, 64]);  slice_tensor_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:121 in forward, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
        permute_5: "bf16[128, 6, 64, 196][76800, 12800, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_14, [0, 1, 3, 2]);  view_14 = None
        clone_28: "bf16[128, 6, 64, 196][75264, 12544, 196, 1]cuda:0" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_15: "bf16[128, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(clone_28, [128, 384, 14, 14]);  clone_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:122 in forward, code: x = self.proj(x)
        convert_element_type_91: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_100, torch.bfloat16);  primals_100 = None
        convolution_29: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(view_15, convert_element_type_91, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        add_91: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_85, convolution_29);  convolution_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_92: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_101, 1)
        var_mean_13 = torch.ops.aten.var_mean.correction(add_91, [0, 2, 3], correction = 0, keepdim = True)
        getitem_32: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_13[0]
        getitem_33: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        add_93: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
        rsqrt_13: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        sub_15: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(add_91, getitem_33)
        mul_138: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_13);  sub_15 = None
        squeeze_39: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        squeeze_40: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
        mul_139: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
        mul_140: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_102, 0.9)
        add_94: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_139, mul_140);  mul_139 = mul_140 = None
        squeeze_41: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
        mul_141: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0000398612827361);  squeeze_41 = None
        mul_142: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_141, 0.1);  mul_141 = None
        mul_143: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_103, 0.9)
        add_95: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_142, mul_143);  mul_142 = mul_143 = None
        unsqueeze_52: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_104, -1)
        unsqueeze_53: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_144: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_138, unsqueeze_53);  mul_138 = unsqueeze_53 = None
        unsqueeze_54: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_105, -1);  primals_105 = None
        unsqueeze_55: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_96: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_144, unsqueeze_55);  mul_144 = unsqueeze_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.conv1(x)
        convert_element_type_92: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_106, torch.bfloat16);  primals_106 = None
        convert_element_type_93: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_96, torch.bfloat16);  add_96 = None
        convolution_30: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_93, convert_element_type_92, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.act1(x)
        convert_element_type_94: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_30, torch.float32)
        mul_145: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_94, 0.5)
        mul_146: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_94, 0.7071067811865476);  convert_element_type_94 = None
        erf_15: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.erf.default(mul_146);  mul_146 = None
        add_97: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_147: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_145, add_97);  mul_145 = add_97 = None
        convert_element_type_95: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(mul_147, torch.bfloat16);  mul_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:73 in forward, code: x = self.conv3(x)
        convert_element_type_96: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(primals_107, torch.bfloat16);  primals_107 = None
        convolution_31: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_95, convert_element_type_96, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_98: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_91, convolution_31);  convolution_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        add_99: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_108, 1)
        var_mean_14 = torch.ops.aten.var_mean.correction(add_98, [0, 2, 3], correction = 0, keepdim = True)
        getitem_34: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_14[0]
        getitem_35: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        add_100: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
        rsqrt_14: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        sub_16: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(add_98, getitem_35)
        mul_148: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_14);  sub_16 = None
        squeeze_42: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
        squeeze_43: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_149: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
        mul_150: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_109, 0.9)
        add_101: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_149, mul_150);  mul_149 = mul_150 = None
        squeeze_44: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
        mul_151: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0000398612827361);  squeeze_44 = None
        mul_152: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_151, 0.1);  mul_151 = None
        mul_153: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_110, 0.9)
        add_102: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_152, mul_153);  mul_152 = mul_153 = None
        unsqueeze_56: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_111, -1)
        unsqueeze_57: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        mul_154: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_148, unsqueeze_57);  mul_148 = unsqueeze_57 = None
        unsqueeze_58: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_112, -1);  primals_112 = None
        unsqueeze_59: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        add_103: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_154, unsqueeze_59);  mul_154 = unsqueeze_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:107 in forward, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
        convert_element_type_97: "bf16[1152, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_113, torch.bfloat16);  primals_113 = None
        convert_element_type_98: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_103, torch.bfloat16);  add_103 = None
        convolution_32: "bf16[128, 1152, 14, 14][225792, 1, 16128, 1152]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_98, convert_element_type_97, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        view_16: "bf16[128, 3, 6, 64, 196][225792, 384, 64, 1, 1152]cuda:0" = torch.ops.aten.reshape.default(convolution_32, [128, 3, 6, 64, 196]);  convolution_32 = None
        permute_6: "bf16[3, 128, 6, 196, 64][384, 225792, 64, 1152, 1]cuda:0" = torch.ops.aten.permute.default(view_16, [1, 0, 2, 4, 3]);  view_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:108 in forward, code: q, k, v = x.unbind(0)
        unbind_2 = torch.ops.aten.unbind.int(permute_6);  permute_6 = None
        getitem_36: "bf16[128, 6, 196, 64][225792, 64, 1152, 1]cuda:0" = unbind_2[0]
        getitem_37: "bf16[128, 6, 196, 64][225792, 64, 1152, 1]cuda:0" = unbind_2[1]
        getitem_38: "bf16[128, 6, 196, 64][225792, 64, 1152, 1]cuda:0" = unbind_2[2];  unbind_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_7: "bf16[128, 6, 64, 196][225792, 64, 1, 1152]cuda:0" = torch.ops.aten.permute.default(getitem_37, [0, 1, 3, 2]);  getitem_37 = None
        expand_8: "bf16[128, 6, 196, 64][225792, 64, 1152, 1]cuda:0" = torch.ops.aten.expand.default(getitem_36, [128, 6, 196, 64]);  getitem_36 = None
        clone_32: "bf16[128, 6, 196, 64][75264, 12544, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_17: "bf16[768, 196, 64][12544, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_32, [768, 196, 64]);  clone_32 = None
        expand_9: "bf16[128, 6, 64, 196][225792, 64, 1, 1152]cuda:0" = torch.ops.aten.expand.default(permute_7, [128, 6, 64, 196]);  permute_7 = None
        clone_33: "bf16[128, 6, 64, 196][75264, 12544, 196, 1]cuda:0" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_18: "bf16[768, 64, 196][12544, 196, 1]cuda:0" = torch.ops.aten.reshape.default(clone_33, [768, 64, 196]);  clone_33 = None
        constant_pad_nd_default_14: "bf16[768, 200, 64][12800, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_17, [0, 0, 0, 4, 0, 0])
        constant_pad_nd_default_15: "bf16[768, 64, 200][12800, 200, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_18, [0, 4, 0, 0, 0, 0])
        bmm_default_7: "bf16[768, 200, 200][40000, 200, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_14, constant_pad_nd_default_15);  constant_pad_nd_default_14 = constant_pad_nd_default_15 = None
        slice_tensor_12: "bf16[768, 196, 200][40000, 200, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_7, 1, 0, -4)
        slice_tensor_13: "bf16[768, 196, 196][40000, 200, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_12, 2, 0, -4);  slice_tensor_12 = None
        view_19: "bf16[128, 6, 196, 196][240000, 40000, 200, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_13, [128, 6, 196, 196]);  slice_tensor_13 = None

        # No stacktrace found for following nodes
        mul_tensor_20: "bf16[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_19, 0.125)
        convert_element_type_default_10: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_20, torch.float32);  mul_tensor_20 = None
        convert_element_type_default_11: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        mul_tensor_21: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_11, 1);  convert_element_type_default_11 = None
        amax_default_10: "f32[128, 6, 196, 1][1176, 196, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_21, [-1], True)
        sub_tensor_10: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_21, amax_default_10);  mul_tensor_21 = None
        mul_tensor_22: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_10, 0.125);  sub_tensor_10 = None
        amax_default_11: "f32[128, 6, 196, 1][1176, 196, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_10, [-1], True)
        sub_tensor_11: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_10, amax_default_11)
        abs_default_5: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_10)
        ne_scalar_5: "b8[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_5, inf);  abs_default_5 = None
        eq_tensor_5: "b8[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_10, convert_element_type_default_10);  convert_element_type_default_10 = None
        mul_tensor_23: "b8[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_5, ne_scalar_5);  eq_tensor_5 = ne_scalar_5 = None
        logical_not_default_10: "b8[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_23);  mul_tensor_23 = None
        any_dims_5: "b8[128, 6, 196, 1][1176, 196, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_10, [-1], True);  logical_not_default_10 = None
        logical_not_default_11: "b8[128, 6, 196, 1][1176, 196, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_5);  any_dims_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:117 in forward, code: attn = attn.softmax(dim=-1)
        where_self_5: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_11, mul_tensor_22, sub_tensor_11);  mul_tensor_22 = sub_tensor_11 = None
        exp_2: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.exp.default(where_self_5);  where_self_5 = None
        sum_3: "f32[128, 6, 196, 1][1176, 196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        convert_element_type_102: "bf16[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None
        expand_10: "bf16[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_102, [128, 6, 196, 196]);  convert_element_type_102 = None
        view_20: "bf16[768, 196, 196][38416, 196, 1]cuda:0" = torch.ops.aten.reshape.default(expand_10, [768, 196, 196]);  expand_10 = None
        expand_11: "bf16[128, 6, 196, 64][225792, 64, 1152, 1]cuda:0" = torch.ops.aten.expand.default(getitem_38, [128, 6, 196, 64]);  getitem_38 = None
        clone_35: "bf16[128, 6, 196, 64][75264, 12544, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_21: "bf16[768, 196, 64][12544, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_35, [768, 196, 64]);  clone_35 = None
        constant_pad_nd_default_12: "bf16[768, 200, 200][40000, 200, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_20, [0, 4, 0, 4, 0, 0])
        constant_pad_nd_default_13: "bf16[768, 200, 64][12800, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_21, [0, 0, 0, 4, 0, 0])
        bmm_default_6: "bf16[768, 200, 64][12800, 64, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_12, constant_pad_nd_default_13);  constant_pad_nd_default_12 = constant_pad_nd_default_13 = None
        slice_tensor_11: "bf16[768, 196, 64][12800, 64, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_6, 1, 0, -4);  bmm_default_6 = None
        view_22: "bf16[128, 6, 196, 64][76800, 12800, 64, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_11, [128, 6, 196, 64]);  slice_tensor_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:121 in forward, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
        permute_8: "bf16[128, 6, 64, 196][76800, 12800, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_22, [0, 1, 3, 2]);  view_22 = None
        clone_36: "bf16[128, 6, 64, 196][75264, 12544, 196, 1]cuda:0" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
        view_23: "bf16[128, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(clone_36, [128, 384, 14, 14]);  clone_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:122 in forward, code: x = self.proj(x)
        convert_element_type_105: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_114, torch.bfloat16);  primals_114 = None
        convolution_33: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(view_23, convert_element_type_105, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        add_104: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_98, convolution_33);  convolution_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_105: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_115, 1)
        var_mean_15 = torch.ops.aten.var_mean.correction(add_104, [0, 2, 3], correction = 0, keepdim = True)
        getitem_39: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_15[0]
        getitem_40: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        add_106: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_39, 1e-05)
        rsqrt_15: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        sub_18: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(add_104, getitem_40)
        mul_156: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_15);  sub_18 = None
        squeeze_45: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
        squeeze_46: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_157: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
        mul_158: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_116, 0.9)
        add_107: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_157, mul_158);  mul_157 = mul_158 = None
        squeeze_47: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
        mul_159: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0000398612827361);  squeeze_47 = None
        mul_160: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_159, 0.1);  mul_159 = None
        mul_161: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_117, 0.9)
        add_108: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_160, mul_161);  mul_160 = mul_161 = None
        unsqueeze_60: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_118, -1)
        unsqueeze_61: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_162: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_156, unsqueeze_61);  mul_156 = unsqueeze_61 = None
        unsqueeze_62: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_119, -1);  primals_119 = None
        unsqueeze_63: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_109: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_162, unsqueeze_63);  mul_162 = unsqueeze_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.conv1(x)
        convert_element_type_106: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_120, torch.bfloat16);  primals_120 = None
        convert_element_type_107: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_109, torch.bfloat16);  add_109 = None
        convolution_34: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_107, convert_element_type_106, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.act1(x)
        convert_element_type_108: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_34, torch.float32)
        mul_163: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_108, 0.5)
        mul_164: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_108, 0.7071067811865476);  convert_element_type_108 = None
        erf_16: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.erf.default(mul_164);  mul_164 = None
        add_110: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_165: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_163, add_110);  mul_163 = add_110 = None
        convert_element_type_109: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(mul_165, torch.bfloat16);  mul_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:73 in forward, code: x = self.conv3(x)
        convert_element_type_110: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(primals_121, torch.bfloat16);  primals_121 = None
        convolution_35: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_109, convert_element_type_110, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_111: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_104, convolution_35);  convolution_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        add_112: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_122, 1)
        var_mean_16 = torch.ops.aten.var_mean.correction(add_111, [0, 2, 3], correction = 0, keepdim = True)
        getitem_41: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_16[0]
        getitem_42: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        add_113: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_41, 1e-05)
        rsqrt_16: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
        sub_19: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(add_111, getitem_42)
        mul_166: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_16);  sub_19 = None
        squeeze_48: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
        squeeze_49: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_167: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
        mul_168: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_123, 0.9)
        add_114: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_167, mul_168);  mul_167 = mul_168 = None
        squeeze_50: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
        mul_169: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0000398612827361);  squeeze_50 = None
        mul_170: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_169, 0.1);  mul_169 = None
        mul_171: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_124, 0.9)
        add_115: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_170, mul_171);  mul_170 = mul_171 = None
        unsqueeze_64: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_125, -1)
        unsqueeze_65: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        mul_172: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_166, unsqueeze_65);  mul_166 = unsqueeze_65 = None
        unsqueeze_66: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_126, -1);  primals_126 = None
        unsqueeze_67: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        add_116: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_172, unsqueeze_67);  mul_172 = unsqueeze_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:107 in forward, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
        convert_element_type_111: "bf16[1152, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_127, torch.bfloat16);  primals_127 = None
        convert_element_type_112: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_116, torch.bfloat16);  add_116 = None
        convolution_36: "bf16[128, 1152, 14, 14][225792, 1, 16128, 1152]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_112, convert_element_type_111, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        view_24: "bf16[128, 3, 6, 64, 196][225792, 384, 64, 1, 1152]cuda:0" = torch.ops.aten.reshape.default(convolution_36, [128, 3, 6, 64, 196]);  convolution_36 = None
        permute_9: "bf16[3, 128, 6, 196, 64][384, 225792, 64, 1152, 1]cuda:0" = torch.ops.aten.permute.default(view_24, [1, 0, 2, 4, 3]);  view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:108 in forward, code: q, k, v = x.unbind(0)
        unbind_3 = torch.ops.aten.unbind.int(permute_9);  permute_9 = None
        getitem_43: "bf16[128, 6, 196, 64][225792, 64, 1152, 1]cuda:0" = unbind_3[0]
        getitem_44: "bf16[128, 6, 196, 64][225792, 64, 1152, 1]cuda:0" = unbind_3[1]
        getitem_45: "bf16[128, 6, 196, 64][225792, 64, 1152, 1]cuda:0" = unbind_3[2];  unbind_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_10: "bf16[128, 6, 64, 196][225792, 64, 1, 1152]cuda:0" = torch.ops.aten.permute.default(getitem_44, [0, 1, 3, 2]);  getitem_44 = None
        expand_12: "bf16[128, 6, 196, 64][225792, 64, 1152, 1]cuda:0" = torch.ops.aten.expand.default(getitem_43, [128, 6, 196, 64]);  getitem_43 = None
        clone_40: "bf16[128, 6, 196, 64][75264, 12544, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_25: "bf16[768, 196, 64][12544, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_40, [768, 196, 64]);  clone_40 = None
        expand_13: "bf16[128, 6, 64, 196][225792, 64, 1, 1152]cuda:0" = torch.ops.aten.expand.default(permute_10, [128, 6, 64, 196]);  permute_10 = None
        clone_41: "bf16[128, 6, 64, 196][75264, 12544, 196, 1]cuda:0" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_26: "bf16[768, 64, 196][12544, 196, 1]cuda:0" = torch.ops.aten.reshape.default(clone_41, [768, 64, 196]);  clone_41 = None
        constant_pad_nd_default_10: "bf16[768, 200, 64][12800, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_25, [0, 0, 0, 4, 0, 0])
        constant_pad_nd_default_11: "bf16[768, 64, 200][12800, 200, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_26, [0, 4, 0, 0, 0, 0])
        bmm_default_5: "bf16[768, 200, 200][40000, 200, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_10, constant_pad_nd_default_11);  constant_pad_nd_default_10 = constant_pad_nd_default_11 = None
        slice_tensor_9: "bf16[768, 196, 200][40000, 200, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_5, 1, 0, -4)
        slice_tensor_10: "bf16[768, 196, 196][40000, 200, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_9, 2, 0, -4);  slice_tensor_9 = None
        view_27: "bf16[128, 6, 196, 196][240000, 40000, 200, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_10, [128, 6, 196, 196]);  slice_tensor_10 = None

        # No stacktrace found for following nodes
        mul_tensor_16: "bf16[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_27, 0.125)
        convert_element_type_default_8: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_16, torch.float32);  mul_tensor_16 = None
        convert_element_type_default_9: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_27, torch.float32);  view_27 = None
        mul_tensor_17: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_9, 1);  convert_element_type_default_9 = None
        amax_default_8: "f32[128, 6, 196, 1][1176, 196, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_17, [-1], True)
        sub_tensor_8: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_17, amax_default_8);  mul_tensor_17 = None
        mul_tensor_18: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_8, 0.125);  sub_tensor_8 = None
        amax_default_9: "f32[128, 6, 196, 1][1176, 196, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_8, [-1], True)
        sub_tensor_9: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_8, amax_default_9)
        abs_default_4: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_8)
        ne_scalar_4: "b8[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_4, inf);  abs_default_4 = None
        eq_tensor_4: "b8[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_8, convert_element_type_default_8);  convert_element_type_default_8 = None
        mul_tensor_19: "b8[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_4, ne_scalar_4);  eq_tensor_4 = ne_scalar_4 = None
        logical_not_default_8: "b8[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_19);  mul_tensor_19 = None
        any_dims_4: "b8[128, 6, 196, 1][1176, 196, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_8, [-1], True);  logical_not_default_8 = None
        logical_not_default_9: "b8[128, 6, 196, 1][1176, 196, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_4);  any_dims_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:117 in forward, code: attn = attn.softmax(dim=-1)
        where_self_4: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_9, mul_tensor_18, sub_tensor_9);  mul_tensor_18 = sub_tensor_9 = None
        exp_3: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.exp.default(where_self_4);  where_self_4 = None
        sum_4: "f32[128, 6, 196, 1][1176, 196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        convert_element_type_116: "bf16[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None
        expand_14: "bf16[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_116, [128, 6, 196, 196]);  convert_element_type_116 = None
        view_28: "bf16[768, 196, 196][38416, 196, 1]cuda:0" = torch.ops.aten.reshape.default(expand_14, [768, 196, 196]);  expand_14 = None
        expand_15: "bf16[128, 6, 196, 64][225792, 64, 1152, 1]cuda:0" = torch.ops.aten.expand.default(getitem_45, [128, 6, 196, 64]);  getitem_45 = None
        clone_43: "bf16[128, 6, 196, 64][75264, 12544, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_29: "bf16[768, 196, 64][12544, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_43, [768, 196, 64]);  clone_43 = None
        constant_pad_nd_default_8: "bf16[768, 200, 200][40000, 200, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_28, [0, 4, 0, 4, 0, 0])
        constant_pad_nd_default_9: "bf16[768, 200, 64][12800, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_29, [0, 0, 0, 4, 0, 0])
        bmm_default_4: "bf16[768, 200, 64][12800, 64, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_8, constant_pad_nd_default_9);  constant_pad_nd_default_8 = constant_pad_nd_default_9 = None
        slice_tensor_8: "bf16[768, 196, 64][12800, 64, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_4, 1, 0, -4);  bmm_default_4 = None
        view_30: "bf16[128, 6, 196, 64][76800, 12800, 64, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_8, [128, 6, 196, 64]);  slice_tensor_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:121 in forward, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
        permute_11: "bf16[128, 6, 64, 196][76800, 12800, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_30, [0, 1, 3, 2]);  view_30 = None
        clone_44: "bf16[128, 6, 64, 196][75264, 12544, 196, 1]cuda:0" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
        view_31: "bf16[128, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(clone_44, [128, 384, 14, 14]);  clone_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:122 in forward, code: x = self.proj(x)
        convert_element_type_119: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_128, torch.bfloat16);  primals_128 = None
        convolution_37: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(view_31, convert_element_type_119, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        add_117: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_111, convolution_37);  convolution_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_118: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_129, 1)
        var_mean_17 = torch.ops.aten.var_mean.correction(add_117, [0, 2, 3], correction = 0, keepdim = True)
        getitem_46: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_17[0]
        getitem_47: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        add_119: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
        rsqrt_17: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
        sub_21: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(add_117, getitem_47)
        mul_174: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_17);  sub_21 = None
        squeeze_51: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
        squeeze_52: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
        mul_175: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
        mul_176: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_130, 0.9)
        add_120: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_175, mul_176);  mul_175 = mul_176 = None
        squeeze_53: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
        mul_177: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0000398612827361);  squeeze_53 = None
        mul_178: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_177, 0.1);  mul_177 = None
        mul_179: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_131, 0.9)
        add_121: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_178, mul_179);  mul_178 = mul_179 = None
        unsqueeze_68: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_132, -1)
        unsqueeze_69: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_180: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_174, unsqueeze_69);  mul_174 = unsqueeze_69 = None
        unsqueeze_70: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_133, -1);  primals_133 = None
        unsqueeze_71: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_122: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_180, unsqueeze_71);  mul_180 = unsqueeze_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.conv1(x)
        convert_element_type_120: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_134, torch.bfloat16);  primals_134 = None
        convert_element_type_121: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_122, torch.bfloat16);  add_122 = None
        convolution_38: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_121, convert_element_type_120, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.act1(x)
        convert_element_type_122: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_38, torch.float32)
        mul_181: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_122, 0.5)
        mul_182: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_122, 0.7071067811865476);  convert_element_type_122 = None
        erf_17: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.erf.default(mul_182);  mul_182 = None
        add_123: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_183: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_181, add_123);  mul_181 = add_123 = None
        convert_element_type_123: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(mul_183, torch.bfloat16);  mul_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:73 in forward, code: x = self.conv3(x)
        convert_element_type_124: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(primals_135, torch.bfloat16);  primals_135 = None
        convolution_39: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_123, convert_element_type_124, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_124: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_117, convolution_39);  convolution_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.proj(x)
        convert_element_type_125: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_137, torch.bfloat16);  primals_137 = None
        convert_element_type_126: "bf16[768, 384, 2, 2][1536, 1, 768, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_136, torch.bfloat16);  primals_136 = None
        convert_element_type_127: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_124, torch.bfloat16);  add_124 = None
        convolution_40: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_127, convert_element_type_126, convert_element_type_125, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:141 in forward, code: x = self.norm(x)
        add_125: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_138, 1)
        convert_element_type_128: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_40, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_128, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_128 = None
        getitem_48: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = var_mean_18[0]
        getitem_49: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        add_126: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_48, 1e-05)
        rsqrt_18: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        sub_22: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(convolution_40, getitem_49)
        mul_184: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_18);  sub_22 = None
        squeeze_54: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3])
        mul_185: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1);  squeeze_54 = None
        mul_186: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_139, 0.9)
        add_127: "f32[768][1]cuda:0" = torch.ops.aten.add.Tensor(mul_185, mul_186);  mul_185 = mul_186 = None
        squeeze_56: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
        mul_187: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0001594642002871);  squeeze_56 = None
        mul_188: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_187, 0.1);  mul_187 = None
        mul_189: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_140, 0.9)
        add_128: "f32[768][1]cuda:0" = torch.ops.aten.add.Tensor(mul_188, mul_189);  mul_188 = mul_189 = None
        unsqueeze_72: "f32[768, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_141, -1)
        unsqueeze_73: "f32[768, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        mul_190: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(mul_184, unsqueeze_73);  mul_184 = unsqueeze_73 = None
        unsqueeze_74: "f32[768, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_142, -1)
        unsqueeze_75: "f32[768, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        add_129: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.add.Tensor(mul_190, unsqueeze_75);  mul_190 = unsqueeze_75 = None
        convert_element_type_129: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.prims.convert_element_type.default(add_129, torch.bfloat16);  add_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:462 in forward_features, code: x = self.pos_drop(x + self.pos_embed3)
        add_130: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_129, primals_143);  convert_element_type_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        add_131: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_144, 1)
        var_mean_19 = torch.ops.aten.var_mean.correction(add_130, [0, 2, 3], correction = 0, keepdim = True)
        getitem_50: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = var_mean_19[0]
        getitem_51: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        add_132: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_50, 1e-05)
        rsqrt_19: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
        sub_23: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(add_130, getitem_51)
        mul_191: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_19);  sub_23 = None
        squeeze_57: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
        squeeze_58: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
        mul_192: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
        mul_193: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_145, 0.9)
        add_133: "f32[768][1]cuda:0" = torch.ops.aten.add.Tensor(mul_192, mul_193);  mul_192 = mul_193 = None
        squeeze_59: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
        mul_194: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0001594642002871);  squeeze_59 = None
        mul_195: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_194, 0.1);  mul_194 = None
        mul_196: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_146, 0.9)
        add_134: "f32[768][1]cuda:0" = torch.ops.aten.add.Tensor(mul_195, mul_196);  mul_195 = mul_196 = None
        unsqueeze_76: "f32[768, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_147, -1)
        unsqueeze_77: "f32[768, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_197: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(mul_191, unsqueeze_77);  mul_191 = unsqueeze_77 = None
        unsqueeze_78: "f32[768, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_148, -1);  primals_148 = None
        unsqueeze_79: "f32[768, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_135: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.add.Tensor(mul_197, unsqueeze_79);  mul_197 = unsqueeze_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:107 in forward, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
        convert_element_type_130: "bf16[2304, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(primals_149, torch.bfloat16);  primals_149 = None
        convert_element_type_131: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.prims.convert_element_type.default(add_135, torch.bfloat16);  add_135 = None
        convolution_41: "bf16[128, 2304, 7, 7][112896, 1, 16128, 2304]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_131, convert_element_type_130, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        view_32: "bf16[128, 3, 6, 128, 49][112896, 768, 128, 1, 2304]cuda:0" = torch.ops.aten.reshape.default(convolution_41, [128, 3, 6, 128, 49]);  convolution_41 = None
        permute_12: "bf16[3, 128, 6, 49, 128][768, 112896, 128, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_32, [1, 0, 2, 4, 3]);  view_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:108 in forward, code: q, k, v = x.unbind(0)
        unbind_4 = torch.ops.aten.unbind.int(permute_12);  permute_12 = None
        getitem_52: "bf16[128, 6, 49, 128][112896, 128, 2304, 1]cuda:0" = unbind_4[0]
        getitem_53: "bf16[128, 6, 49, 128][112896, 128, 2304, 1]cuda:0" = unbind_4[1]
        getitem_54: "bf16[128, 6, 49, 128][112896, 128, 2304, 1]cuda:0" = unbind_4[2];  unbind_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_13: "bf16[128, 6, 128, 49][112896, 128, 1, 2304]cuda:0" = torch.ops.aten.permute.default(getitem_53, [0, 1, 3, 2]);  getitem_53 = None
        expand_16: "bf16[128, 6, 49, 128][112896, 128, 2304, 1]cuda:0" = torch.ops.aten.expand.default(getitem_52, [128, 6, 49, 128]);  getitem_52 = None
        clone_49: "bf16[128, 6, 49, 128][37632, 6272, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_33: "bf16[768, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_49, [768, 49, 128]);  clone_49 = None
        expand_17: "bf16[128, 6, 128, 49][112896, 128, 1, 2304]cuda:0" = torch.ops.aten.expand.default(permute_13, [128, 6, 128, 49]);  permute_13 = None
        clone_50: "bf16[128, 6, 128, 49][37632, 6272, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_34: "bf16[768, 128, 49][6272, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_50, [768, 128, 49]);  clone_50 = None
        constant_pad_nd_default_6: "bf16[768, 56, 128][7168, 128, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_33, [0, 0, 0, 7, 0, 0])
        constant_pad_nd_default_7: "bf16[768, 128, 56][7168, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_34, [0, 7, 0, 0, 0, 0])
        bmm_default_3: "bf16[768, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_6, constant_pad_nd_default_7);  constant_pad_nd_default_6 = constant_pad_nd_default_7 = None
        slice_tensor_6: "bf16[768, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_3, 1, 0, -7)
        slice_tensor_7: "bf16[768, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_6, 2, 0, -7);  slice_tensor_6 = None
        view_35: "bf16[128, 6, 49, 49][18816, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_7, [128, 6, 49, 49]);  slice_tensor_7 = None

        # No stacktrace found for following nodes
        mul_tensor_12: "bf16[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_35, 0.08838834764831845)
        convert_element_type_default_6: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_12, torch.float32);  mul_tensor_12 = None
        convert_element_type_default_7: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_35, torch.float32);  view_35 = None
        mul_tensor_13: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_7, 1);  convert_element_type_default_7 = None
        amax_default_6: "f32[128, 6, 49, 1][294, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_13, [-1], True)
        sub_tensor_6: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_13, amax_default_6);  mul_tensor_13 = None
        mul_tensor_14: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_6, 0.08838834764831845);  sub_tensor_6 = None
        amax_default_7: "f32[128, 6, 49, 1][294, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_6, [-1], True)
        sub_tensor_7: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_6, amax_default_7)
        abs_default_3: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_6)
        ne_scalar_3: "b8[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_3, inf);  abs_default_3 = None
        eq_tensor_3: "b8[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_6, convert_element_type_default_6);  convert_element_type_default_6 = None
        mul_tensor_15: "b8[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_3, ne_scalar_3);  eq_tensor_3 = ne_scalar_3 = None
        logical_not_default_6: "b8[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_15);  mul_tensor_15 = None
        any_dims_3: "b8[128, 6, 49, 1][294, 49, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_6, [-1], True);  logical_not_default_6 = None
        logical_not_default_7: "b8[128, 6, 49, 1][294, 49, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_3);  any_dims_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:117 in forward, code: attn = attn.softmax(dim=-1)
        where_self_3: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_7, mul_tensor_14, sub_tensor_7);  mul_tensor_14 = sub_tensor_7 = None
        exp_4: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(where_self_3);  where_self_3 = None
        sum_5: "f32[128, 6, 49, 1][294, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        convert_element_type_135: "bf16[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None
        expand_18: "bf16[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_135, [128, 6, 49, 49]);  convert_element_type_135 = None
        view_36: "bf16[768, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_18, [768, 49, 49]);  expand_18 = None
        expand_19: "bf16[128, 6, 49, 128][112896, 128, 2304, 1]cuda:0" = torch.ops.aten.expand.default(getitem_54, [128, 6, 49, 128]);  getitem_54 = None
        clone_52: "bf16[128, 6, 49, 128][37632, 6272, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_37: "bf16[768, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_52, [768, 49, 128]);  clone_52 = None
        bmm_9: "bf16[768, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_36, view_37)
        view_38: "bf16[128, 6, 49, 128][37632, 6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_9, [128, 6, 49, 128]);  bmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:121 in forward, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
        permute_14: "bf16[128, 6, 128, 49][37632, 6272, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_38, [0, 1, 3, 2]);  view_38 = None
        clone_53: "bf16[128, 6, 128, 49][37632, 6272, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
        view_39: "bf16[128, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.reshape.default(clone_53, [128, 768, 7, 7]);  clone_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:122 in forward, code: x = self.proj(x)
        convert_element_type_138: "bf16[768, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(primals_150, torch.bfloat16);  primals_150 = None
        convolution_42: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.convolution.default(view_39, convert_element_type_138, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        add_136: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.add.Tensor(add_130, convolution_42);  add_130 = convolution_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_137: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_151, 1)
        var_mean_20 = torch.ops.aten.var_mean.correction(add_136, [0, 2, 3], correction = 0, keepdim = True)
        getitem_55: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = var_mean_20[0]
        getitem_56: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None
        add_138: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_55, 1e-05)
        rsqrt_20: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
        sub_25: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(add_136, getitem_56)
        mul_199: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_20);  sub_25 = None
        squeeze_60: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
        squeeze_61: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
        mul_200: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
        mul_201: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_152, 0.9)
        add_139: "f32[768][1]cuda:0" = torch.ops.aten.add.Tensor(mul_200, mul_201);  mul_200 = mul_201 = None
        squeeze_62: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
        mul_202: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0001594642002871);  squeeze_62 = None
        mul_203: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_202, 0.1);  mul_202 = None
        mul_204: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_153, 0.9)
        add_140: "f32[768][1]cuda:0" = torch.ops.aten.add.Tensor(mul_203, mul_204);  mul_203 = mul_204 = None
        unsqueeze_80: "f32[768, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_154, -1)
        unsqueeze_81: "f32[768, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        mul_205: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(mul_199, unsqueeze_81);  mul_199 = unsqueeze_81 = None
        unsqueeze_82: "f32[768, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_155, -1);  primals_155 = None
        unsqueeze_83: "f32[768, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        add_141: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.add.Tensor(mul_205, unsqueeze_83);  mul_205 = unsqueeze_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.conv1(x)
        convert_element_type_139: "bf16[3072, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(primals_156, torch.bfloat16);  primals_156 = None
        convert_element_type_140: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.prims.convert_element_type.default(add_141, torch.bfloat16);  add_141 = None
        convolution_43: "bf16[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_140, convert_element_type_139, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.act1(x)
        convert_element_type_141: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_43, torch.float32)
        mul_206: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_141, 0.5)
        mul_207: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_141, 0.7071067811865476);  convert_element_type_141 = None
        erf_18: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.erf.default(mul_207);  mul_207 = None
        add_142: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_208: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(mul_206, add_142);  mul_206 = add_142 = None
        convert_element_type_142: "bf16[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.prims.convert_element_type.default(mul_208, torch.bfloat16);  mul_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:73 in forward, code: x = self.conv3(x)
        convert_element_type_143: "bf16[768, 3072, 1, 1][3072, 1, 3072, 3072]cuda:0" = torch.ops.prims.convert_element_type.default(primals_157, torch.bfloat16);  primals_157 = None
        convolution_44: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_142, convert_element_type_143, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_143: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.add.Tensor(add_136, convolution_44);  convolution_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        add_144: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_158, 1)
        var_mean_21 = torch.ops.aten.var_mean.correction(add_143, [0, 2, 3], correction = 0, keepdim = True)
        getitem_57: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = var_mean_21[0]
        getitem_58: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        add_145: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_57, 1e-05)
        rsqrt_21: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
        sub_26: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(add_143, getitem_58)
        mul_209: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_21);  sub_26 = None
        squeeze_63: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
        squeeze_64: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
        mul_210: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
        mul_211: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_159, 0.9)
        add_146: "f32[768][1]cuda:0" = torch.ops.aten.add.Tensor(mul_210, mul_211);  mul_210 = mul_211 = None
        squeeze_65: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
        mul_212: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0001594642002871);  squeeze_65 = None
        mul_213: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_212, 0.1);  mul_212 = None
        mul_214: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_160, 0.9)
        add_147: "f32[768][1]cuda:0" = torch.ops.aten.add.Tensor(mul_213, mul_214);  mul_213 = mul_214 = None
        unsqueeze_84: "f32[768, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_161, -1)
        unsqueeze_85: "f32[768, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_215: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(mul_209, unsqueeze_85);  mul_209 = unsqueeze_85 = None
        unsqueeze_86: "f32[768, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_162, -1);  primals_162 = None
        unsqueeze_87: "f32[768, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_148: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.add.Tensor(mul_215, unsqueeze_87);  mul_215 = unsqueeze_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:107 in forward, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
        convert_element_type_144: "bf16[2304, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(primals_163, torch.bfloat16);  primals_163 = None
        convert_element_type_145: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.prims.convert_element_type.default(add_148, torch.bfloat16);  add_148 = None
        convolution_45: "bf16[128, 2304, 7, 7][112896, 1, 16128, 2304]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_145, convert_element_type_144, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        view_40: "bf16[128, 3, 6, 128, 49][112896, 768, 128, 1, 2304]cuda:0" = torch.ops.aten.reshape.default(convolution_45, [128, 3, 6, 128, 49]);  convolution_45 = None
        permute_15: "bf16[3, 128, 6, 49, 128][768, 112896, 128, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_40, [1, 0, 2, 4, 3]);  view_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:108 in forward, code: q, k, v = x.unbind(0)
        unbind_5 = torch.ops.aten.unbind.int(permute_15);  permute_15 = None
        getitem_59: "bf16[128, 6, 49, 128][112896, 128, 2304, 1]cuda:0" = unbind_5[0]
        getitem_60: "bf16[128, 6, 49, 128][112896, 128, 2304, 1]cuda:0" = unbind_5[1]
        getitem_61: "bf16[128, 6, 49, 128][112896, 128, 2304, 1]cuda:0" = unbind_5[2];  unbind_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_16: "bf16[128, 6, 128, 49][112896, 128, 1, 2304]cuda:0" = torch.ops.aten.permute.default(getitem_60, [0, 1, 3, 2]);  getitem_60 = None
        expand_20: "bf16[128, 6, 49, 128][112896, 128, 2304, 1]cuda:0" = torch.ops.aten.expand.default(getitem_59, [128, 6, 49, 128]);  getitem_59 = None
        clone_57: "bf16[128, 6, 49, 128][37632, 6272, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_41: "bf16[768, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_57, [768, 49, 128]);  clone_57 = None
        expand_21: "bf16[128, 6, 128, 49][112896, 128, 1, 2304]cuda:0" = torch.ops.aten.expand.default(permute_16, [128, 6, 128, 49]);  permute_16 = None
        clone_58: "bf16[128, 6, 128, 49][37632, 6272, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_42: "bf16[768, 128, 49][6272, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_58, [768, 128, 49]);  clone_58 = None
        constant_pad_nd_default_4: "bf16[768, 56, 128][7168, 128, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_41, [0, 0, 0, 7, 0, 0])
        constant_pad_nd_default_5: "bf16[768, 128, 56][7168, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_42, [0, 7, 0, 0, 0, 0])
        bmm_default_2: "bf16[768, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_4, constant_pad_nd_default_5);  constant_pad_nd_default_4 = constant_pad_nd_default_5 = None
        slice_tensor_4: "bf16[768, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_2, 1, 0, -7)
        slice_tensor_5: "bf16[768, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_4, 2, 0, -7);  slice_tensor_4 = None
        view_43: "bf16[128, 6, 49, 49][18816, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_5, [128, 6, 49, 49]);  slice_tensor_5 = None

        # No stacktrace found for following nodes
        mul_tensor_8: "bf16[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_43, 0.08838834764831845)
        convert_element_type_default_4: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_8, torch.float32);  mul_tensor_8 = None
        convert_element_type_default_5: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_43, torch.float32);  view_43 = None
        mul_tensor_9: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_5, 1);  convert_element_type_default_5 = None
        amax_default_4: "f32[128, 6, 49, 1][294, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_9, [-1], True)
        sub_tensor_4: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_9, amax_default_4);  mul_tensor_9 = None
        mul_tensor_10: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_4, 0.08838834764831845);  sub_tensor_4 = None
        amax_default_5: "f32[128, 6, 49, 1][294, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_4, [-1], True)
        sub_tensor_5: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_4, amax_default_5)
        abs_default_2: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_4)
        ne_scalar_2: "b8[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_2, inf);  abs_default_2 = None
        eq_tensor_2: "b8[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_4, convert_element_type_default_4);  convert_element_type_default_4 = None
        mul_tensor_11: "b8[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_2, ne_scalar_2);  eq_tensor_2 = ne_scalar_2 = None
        logical_not_default_4: "b8[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_11);  mul_tensor_11 = None
        any_dims_2: "b8[128, 6, 49, 1][294, 49, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_4, [-1], True);  logical_not_default_4 = None
        logical_not_default_5: "b8[128, 6, 49, 1][294, 49, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_2);  any_dims_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:117 in forward, code: attn = attn.softmax(dim=-1)
        where_self_2: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_5, mul_tensor_10, sub_tensor_5);  mul_tensor_10 = sub_tensor_5 = None
        exp_5: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(where_self_2);  where_self_2 = None
        sum_6: "f32[128, 6, 49, 1][294, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        convert_element_type_149: "bf16[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None
        expand_22: "bf16[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_149, [128, 6, 49, 49]);  convert_element_type_149 = None
        view_44: "bf16[768, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_22, [768, 49, 49]);  expand_22 = None
        expand_23: "bf16[128, 6, 49, 128][112896, 128, 2304, 1]cuda:0" = torch.ops.aten.expand.default(getitem_61, [128, 6, 49, 128]);  getitem_61 = None
        clone_60: "bf16[128, 6, 49, 128][37632, 6272, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_45: "bf16[768, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_60, [768, 49, 128]);  clone_60 = None
        bmm_11: "bf16[768, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_44, view_45)
        view_46: "bf16[128, 6, 49, 128][37632, 6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_11, [128, 6, 49, 128]);  bmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:121 in forward, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
        permute_17: "bf16[128, 6, 128, 49][37632, 6272, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_46, [0, 1, 3, 2]);  view_46 = None
        clone_61: "bf16[128, 6, 128, 49][37632, 6272, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
        view_47: "bf16[128, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.reshape.default(clone_61, [128, 768, 7, 7]);  clone_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:122 in forward, code: x = self.proj(x)
        convert_element_type_152: "bf16[768, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(primals_164, torch.bfloat16);  primals_164 = None
        convolution_46: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.convolution.default(view_47, convert_element_type_152, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        add_149: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.add.Tensor(add_143, convolution_46);  convolution_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_150: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_165, 1)
        var_mean_22 = torch.ops.aten.var_mean.correction(add_149, [0, 2, 3], correction = 0, keepdim = True)
        getitem_62: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = var_mean_22[0]
        getitem_63: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None
        add_151: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
        rsqrt_22: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
        sub_28: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(add_149, getitem_63)
        mul_217: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_22);  sub_28 = None
        squeeze_66: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
        squeeze_67: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
        mul_218: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
        mul_219: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_166, 0.9)
        add_152: "f32[768][1]cuda:0" = torch.ops.aten.add.Tensor(mul_218, mul_219);  mul_218 = mul_219 = None
        squeeze_68: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
        mul_220: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0001594642002871);  squeeze_68 = None
        mul_221: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_220, 0.1);  mul_220 = None
        mul_222: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_167, 0.9)
        add_153: "f32[768][1]cuda:0" = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
        unsqueeze_88: "f32[768, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_168, -1)
        unsqueeze_89: "f32[768, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        mul_223: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_89);  mul_217 = unsqueeze_89 = None
        unsqueeze_90: "f32[768, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_169, -1);  primals_169 = None
        unsqueeze_91: "f32[768, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        add_154: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.add.Tensor(mul_223, unsqueeze_91);  mul_223 = unsqueeze_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.conv1(x)
        convert_element_type_153: "bf16[3072, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(primals_170, torch.bfloat16);  primals_170 = None
        convert_element_type_154: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.prims.convert_element_type.default(add_154, torch.bfloat16);  add_154 = None
        convolution_47: "bf16[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_154, convert_element_type_153, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.act1(x)
        convert_element_type_155: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_47, torch.float32)
        mul_224: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_155, 0.5)
        mul_225: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_155, 0.7071067811865476);  convert_element_type_155 = None
        erf_19: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.erf.default(mul_225);  mul_225 = None
        add_155: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_226: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(mul_224, add_155);  mul_224 = add_155 = None
        convert_element_type_156: "bf16[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.prims.convert_element_type.default(mul_226, torch.bfloat16);  mul_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:73 in forward, code: x = self.conv3(x)
        convert_element_type_157: "bf16[768, 3072, 1, 1][3072, 1, 3072, 3072]cuda:0" = torch.ops.prims.convert_element_type.default(primals_171, torch.bfloat16);  primals_171 = None
        convolution_48: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_156, convert_element_type_157, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_156: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.add.Tensor(add_149, convolution_48);  convolution_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        add_157: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_172, 1)
        var_mean_23 = torch.ops.aten.var_mean.correction(add_156, [0, 2, 3], correction = 0, keepdim = True)
        getitem_64: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = var_mean_23[0]
        getitem_65: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        add_158: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
        rsqrt_23: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
        sub_29: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(add_156, getitem_65)
        mul_227: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_23);  sub_29 = None
        squeeze_69: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
        squeeze_70: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
        mul_228: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
        mul_229: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_173, 0.9)
        add_159: "f32[768][1]cuda:0" = torch.ops.aten.add.Tensor(mul_228, mul_229);  mul_228 = mul_229 = None
        squeeze_71: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
        mul_230: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0001594642002871);  squeeze_71 = None
        mul_231: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_230, 0.1);  mul_230 = None
        mul_232: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_174, 0.9)
        add_160: "f32[768][1]cuda:0" = torch.ops.aten.add.Tensor(mul_231, mul_232);  mul_231 = mul_232 = None
        unsqueeze_92: "f32[768, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_175, -1)
        unsqueeze_93: "f32[768, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        mul_233: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(mul_227, unsqueeze_93);  mul_227 = unsqueeze_93 = None
        unsqueeze_94: "f32[768, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_176, -1);  primals_176 = None
        unsqueeze_95: "f32[768, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        add_161: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.add.Tensor(mul_233, unsqueeze_95);  mul_233 = unsqueeze_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:107 in forward, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
        convert_element_type_158: "bf16[2304, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(primals_177, torch.bfloat16);  primals_177 = None
        convert_element_type_159: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.prims.convert_element_type.default(add_161, torch.bfloat16);  add_161 = None
        convolution_49: "bf16[128, 2304, 7, 7][112896, 1, 16128, 2304]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_159, convert_element_type_158, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        view_48: "bf16[128, 3, 6, 128, 49][112896, 768, 128, 1, 2304]cuda:0" = torch.ops.aten.reshape.default(convolution_49, [128, 3, 6, 128, 49]);  convolution_49 = None
        permute_18: "bf16[3, 128, 6, 49, 128][768, 112896, 128, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_48, [1, 0, 2, 4, 3]);  view_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:108 in forward, code: q, k, v = x.unbind(0)
        unbind_6 = torch.ops.aten.unbind.int(permute_18);  permute_18 = None
        getitem_66: "bf16[128, 6, 49, 128][112896, 128, 2304, 1]cuda:0" = unbind_6[0]
        getitem_67: "bf16[128, 6, 49, 128][112896, 128, 2304, 1]cuda:0" = unbind_6[1]
        getitem_68: "bf16[128, 6, 49, 128][112896, 128, 2304, 1]cuda:0" = unbind_6[2];  unbind_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_19: "bf16[128, 6, 128, 49][112896, 128, 1, 2304]cuda:0" = torch.ops.aten.permute.default(getitem_67, [0, 1, 3, 2]);  getitem_67 = None
        expand_24: "bf16[128, 6, 49, 128][112896, 128, 2304, 1]cuda:0" = torch.ops.aten.expand.default(getitem_66, [128, 6, 49, 128]);  getitem_66 = None
        clone_65: "bf16[128, 6, 49, 128][37632, 6272, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_49: "bf16[768, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_65, [768, 49, 128]);  clone_65 = None
        expand_25: "bf16[128, 6, 128, 49][112896, 128, 1, 2304]cuda:0" = torch.ops.aten.expand.default(permute_19, [128, 6, 128, 49]);  permute_19 = None
        clone_66: "bf16[128, 6, 128, 49][37632, 6272, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_50: "bf16[768, 128, 49][6272, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_66, [768, 128, 49]);  clone_66 = None
        constant_pad_nd_default_2: "bf16[768, 56, 128][7168, 128, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_49, [0, 0, 0, 7, 0, 0])
        constant_pad_nd_default_3: "bf16[768, 128, 56][7168, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_50, [0, 7, 0, 0, 0, 0])
        bmm_default_1: "bf16[768, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_2, constant_pad_nd_default_3);  constant_pad_nd_default_2 = constant_pad_nd_default_3 = None
        slice_tensor_2: "bf16[768, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_1, 1, 0, -7)
        slice_tensor_3: "bf16[768, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_2, 2, 0, -7);  slice_tensor_2 = None
        view_51: "bf16[128, 6, 49, 49][18816, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_3, [128, 6, 49, 49]);  slice_tensor_3 = None

        # No stacktrace found for following nodes
        mul_tensor_4: "bf16[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_51, 0.08838834764831845)
        convert_element_type_default_2: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.float32);  mul_tensor_4 = None
        convert_element_type_default_3: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_51, torch.float32);  view_51 = None
        mul_tensor_5: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_3, 1);  convert_element_type_default_3 = None
        amax_default_2: "f32[128, 6, 49, 1][294, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_5, [-1], True)
        sub_tensor_2: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_5, amax_default_2);  mul_tensor_5 = None
        mul_tensor_6: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_2, 0.08838834764831845);  sub_tensor_2 = None
        amax_default_3: "f32[128, 6, 49, 1][294, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_2, [-1], True)
        sub_tensor_3: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_2, amax_default_3)
        abs_default_1: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_2)
        ne_scalar_1: "b8[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_1, inf);  abs_default_1 = None
        eq_tensor_1: "b8[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_2, convert_element_type_default_2);  convert_element_type_default_2 = None
        mul_tensor_7: "b8[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_1, ne_scalar_1);  eq_tensor_1 = ne_scalar_1 = None
        logical_not_default_2: "b8[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_7);  mul_tensor_7 = None
        any_dims_1: "b8[128, 6, 49, 1][294, 49, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_2, [-1], True);  logical_not_default_2 = None
        logical_not_default_3: "b8[128, 6, 49, 1][294, 49, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_1);  any_dims_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:117 in forward, code: attn = attn.softmax(dim=-1)
        where_self_1: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_3, mul_tensor_6, sub_tensor_3);  mul_tensor_6 = sub_tensor_3 = None
        exp_6: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(where_self_1);  where_self_1 = None
        sum_7: "f32[128, 6, 49, 1][294, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        convert_element_type_163: "bf16[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None
        expand_26: "bf16[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_163, [128, 6, 49, 49]);  convert_element_type_163 = None
        view_52: "bf16[768, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_26, [768, 49, 49]);  expand_26 = None
        expand_27: "bf16[128, 6, 49, 128][112896, 128, 2304, 1]cuda:0" = torch.ops.aten.expand.default(getitem_68, [128, 6, 49, 128]);  getitem_68 = None
        clone_68: "bf16[128, 6, 49, 128][37632, 6272, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_53: "bf16[768, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_68, [768, 49, 128]);  clone_68 = None
        bmm_13: "bf16[768, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_52, view_53)
        view_54: "bf16[128, 6, 49, 128][37632, 6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_13, [128, 6, 49, 128]);  bmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:121 in forward, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
        permute_20: "bf16[128, 6, 128, 49][37632, 6272, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_54, [0, 1, 3, 2]);  view_54 = None
        clone_69: "bf16[128, 6, 128, 49][37632, 6272, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_20, memory_format = torch.contiguous_format);  permute_20 = None
        view_55: "bf16[128, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.reshape.default(clone_69, [128, 768, 7, 7]);  clone_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:122 in forward, code: x = self.proj(x)
        convert_element_type_166: "bf16[768, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(primals_178, torch.bfloat16);  primals_178 = None
        convolution_50: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.convolution.default(view_55, convert_element_type_166, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        add_162: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.add.Tensor(add_156, convolution_50);  convolution_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_163: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_179, 1)
        var_mean_24 = torch.ops.aten.var_mean.correction(add_162, [0, 2, 3], correction = 0, keepdim = True)
        getitem_69: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = var_mean_24[0]
        getitem_70: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None
        add_164: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_69, 1e-05)
        rsqrt_24: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
        sub_31: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(add_162, getitem_70)
        mul_235: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_24);  sub_31 = None
        squeeze_72: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
        squeeze_73: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
        mul_236: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
        mul_237: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_180, 0.9)
        add_165: "f32[768][1]cuda:0" = torch.ops.aten.add.Tensor(mul_236, mul_237);  mul_236 = mul_237 = None
        squeeze_74: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
        mul_238: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0001594642002871);  squeeze_74 = None
        mul_239: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_238, 0.1);  mul_238 = None
        mul_240: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_181, 0.9)
        add_166: "f32[768][1]cuda:0" = torch.ops.aten.add.Tensor(mul_239, mul_240);  mul_239 = mul_240 = None
        unsqueeze_96: "f32[768, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_182, -1)
        unsqueeze_97: "f32[768, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        mul_241: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(mul_235, unsqueeze_97);  mul_235 = unsqueeze_97 = None
        unsqueeze_98: "f32[768, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_183, -1);  primals_183 = None
        unsqueeze_99: "f32[768, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        add_167: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.add.Tensor(mul_241, unsqueeze_99);  mul_241 = unsqueeze_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.conv1(x)
        convert_element_type_167: "bf16[3072, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(primals_184, torch.bfloat16);  primals_184 = None
        convert_element_type_168: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.prims.convert_element_type.default(add_167, torch.bfloat16);  add_167 = None
        convolution_51: "bf16[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_168, convert_element_type_167, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.act1(x)
        convert_element_type_169: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_51, torch.float32)
        mul_242: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_169, 0.5)
        mul_243: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_169, 0.7071067811865476);  convert_element_type_169 = None
        erf_20: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.erf.default(mul_243);  mul_243 = None
        add_168: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_244: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(mul_242, add_168);  mul_242 = add_168 = None
        convert_element_type_170: "bf16[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.prims.convert_element_type.default(mul_244, torch.bfloat16);  mul_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:73 in forward, code: x = self.conv3(x)
        convert_element_type_171: "bf16[768, 3072, 1, 1][3072, 1, 3072, 3072]cuda:0" = torch.ops.prims.convert_element_type.default(primals_185, torch.bfloat16);  primals_185 = None
        convolution_52: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_170, convert_element_type_171, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_169: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.add.Tensor(add_162, convolution_52);  convolution_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        add_170: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_186, 1)
        var_mean_25 = torch.ops.aten.var_mean.correction(add_169, [0, 2, 3], correction = 0, keepdim = True)
        getitem_71: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = var_mean_25[0]
        getitem_72: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = var_mean_25[1];  var_mean_25 = None
        add_171: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_71, 1e-05)
        rsqrt_25: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
        sub_32: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(add_169, getitem_72)
        mul_245: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_25);  sub_32 = None
        squeeze_75: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
        squeeze_76: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
        mul_246: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
        mul_247: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_187, 0.9)
        add_172: "f32[768][1]cuda:0" = torch.ops.aten.add.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
        squeeze_77: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
        mul_248: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0001594642002871);  squeeze_77 = None
        mul_249: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_248, 0.1);  mul_248 = None
        mul_250: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_188, 0.9)
        add_173: "f32[768][1]cuda:0" = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
        unsqueeze_100: "f32[768, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_189, -1)
        unsqueeze_101: "f32[768, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_251: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_101);  mul_245 = unsqueeze_101 = None
        unsqueeze_102: "f32[768, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_190, -1);  primals_190 = None
        unsqueeze_103: "f32[768, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_174: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.add.Tensor(mul_251, unsqueeze_103);  mul_251 = unsqueeze_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:107 in forward, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
        convert_element_type_172: "bf16[2304, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(primals_191, torch.bfloat16);  primals_191 = None
        convert_element_type_173: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.prims.convert_element_type.default(add_174, torch.bfloat16);  add_174 = None
        convolution_53: "bf16[128, 2304, 7, 7][112896, 1, 16128, 2304]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_173, convert_element_type_172, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        view_56: "bf16[128, 3, 6, 128, 49][112896, 768, 128, 1, 2304]cuda:0" = torch.ops.aten.reshape.default(convolution_53, [128, 3, 6, 128, 49]);  convolution_53 = None
        permute_21: "bf16[3, 128, 6, 49, 128][768, 112896, 128, 2304, 1]cuda:0" = torch.ops.aten.permute.default(view_56, [1, 0, 2, 4, 3]);  view_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:108 in forward, code: q, k, v = x.unbind(0)
        unbind_7 = torch.ops.aten.unbind.int(permute_21);  permute_21 = None
        getitem_73: "bf16[128, 6, 49, 128][112896, 128, 2304, 1]cuda:0" = unbind_7[0]
        getitem_74: "bf16[128, 6, 49, 128][112896, 128, 2304, 1]cuda:0" = unbind_7[1]
        getitem_75: "bf16[128, 6, 49, 128][112896, 128, 2304, 1]cuda:0" = unbind_7[2];  unbind_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_22: "bf16[128, 6, 128, 49][112896, 128, 1, 2304]cuda:0" = torch.ops.aten.permute.default(getitem_74, [0, 1, 3, 2]);  getitem_74 = None
        expand_28: "bf16[128, 6, 49, 128][112896, 128, 2304, 1]cuda:0" = torch.ops.aten.expand.default(getitem_73, [128, 6, 49, 128]);  getitem_73 = None
        clone_73: "bf16[128, 6, 49, 128][37632, 6272, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_57: "bf16[768, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_73, [768, 49, 128]);  clone_73 = None
        expand_29: "bf16[128, 6, 128, 49][112896, 128, 1, 2304]cuda:0" = torch.ops.aten.expand.default(permute_22, [128, 6, 128, 49]);  permute_22 = None
        clone_74: "bf16[128, 6, 128, 49][37632, 6272, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_58: "bf16[768, 128, 49][6272, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_74, [768, 128, 49]);  clone_74 = None
        constant_pad_nd_default: "bf16[768, 56, 128][7168, 128, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_57, [0, 0, 0, 7, 0, 0])
        constant_pad_nd_default_1: "bf16[768, 128, 56][7168, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_58, [0, 7, 0, 0, 0, 0])
        bmm_default: "bf16[768, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default, constant_pad_nd_default_1);  constant_pad_nd_default = constant_pad_nd_default_1 = None
        slice_tensor: "bf16[768, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default, 1, 0, -7)
        slice_tensor_1: "bf16[768, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor, 2, 0, -7);  slice_tensor = None
        view_59: "bf16[128, 6, 49, 49][18816, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_1, [128, 6, 49, 49]);  slice_tensor_1 = None

        # No stacktrace found for following nodes
        mul_tensor: "bf16[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_59, 0.08838834764831845)
        convert_element_type_default: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float32);  mul_tensor = None
        convert_element_type_default_1: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_59, torch.float32);  view_59 = None
        mul_tensor_1: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1);  convert_element_type_default_1 = None
        amax_default: "f32[128, 6, 49, 1][294, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_1, [-1], True)
        sub_tensor: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_1, amax_default);  mul_tensor_1 = None
        mul_tensor_2: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor, 0.08838834764831845);  sub_tensor = None
        amax_default_1: "f32[128, 6, 49, 1][294, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default, [-1], True)
        sub_tensor_1: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default, amax_default_1)
        abs_default: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default)
        ne_scalar: "b8[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default, inf);  abs_default = None
        eq_tensor: "b8[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default, convert_element_type_default);  convert_element_type_default = None
        mul_tensor_3: "b8[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor, ne_scalar);  eq_tensor = ne_scalar = None
        logical_not_default: "b8[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_3);  mul_tensor_3 = None
        any_dims: "b8[128, 6, 49, 1][294, 49, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default, [-1], True);  logical_not_default = None
        logical_not_default_1: "b8[128, 6, 49, 1][294, 49, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims);  any_dims = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:117 in forward, code: attn = attn.softmax(dim=-1)
        where_self: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_1, mul_tensor_2, sub_tensor_1);  mul_tensor_2 = sub_tensor_1 = None
        exp_7: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(where_self);  where_self = None
        sum_8: "f32[128, 6, 49, 1][294, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        convert_element_type_177: "bf16[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None
        expand_30: "bf16[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_177, [128, 6, 49, 49]);  convert_element_type_177 = None
        view_60: "bf16[768, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_30, [768, 49, 49]);  expand_30 = None
        expand_31: "bf16[128, 6, 49, 128][112896, 128, 2304, 1]cuda:0" = torch.ops.aten.expand.default(getitem_75, [128, 6, 49, 128]);  getitem_75 = None
        clone_76: "bf16[128, 6, 49, 128][37632, 6272, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_61: "bf16[768, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_76, [768, 49, 128]);  clone_76 = None
        bmm_15: "bf16[768, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_60, view_61)
        view_62: "bf16[128, 6, 49, 128][37632, 6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_15, [128, 6, 49, 128]);  bmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:121 in forward, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
        permute_23: "bf16[128, 6, 128, 49][37632, 6272, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_62, [0, 1, 3, 2]);  view_62 = None
        clone_77: "bf16[128, 6, 128, 49][37632, 6272, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
        view_63: "bf16[128, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.reshape.default(clone_77, [128, 768, 7, 7]);  clone_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:122 in forward, code: x = self.proj(x)
        convert_element_type_180: "bf16[768, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(primals_192, torch.bfloat16);  primals_192 = None
        convolution_54: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.convolution.default(view_63, convert_element_type_180, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        add_175: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.add.Tensor(add_169, convolution_54);  convolution_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_176: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_193, 1)
        var_mean_26 = torch.ops.aten.var_mean.correction(add_175, [0, 2, 3], correction = 0, keepdim = True)
        getitem_76: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = var_mean_26[0]
        getitem_77: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = var_mean_26[1];  var_mean_26 = None
        add_177: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_76, 1e-05)
        rsqrt_26: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_177);  add_177 = None
        sub_34: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(add_175, getitem_77)
        mul_253: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_26);  sub_34 = None
        squeeze_78: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
        squeeze_79: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
        mul_254: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
        mul_255: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_194, 0.9)
        add_178: "f32[768][1]cuda:0" = torch.ops.aten.add.Tensor(mul_254, mul_255);  mul_254 = mul_255 = None
        squeeze_80: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
        mul_256: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0001594642002871);  squeeze_80 = None
        mul_257: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_256, 0.1);  mul_256 = None
        mul_258: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_195, 0.9)
        add_179: "f32[768][1]cuda:0" = torch.ops.aten.add.Tensor(mul_257, mul_258);  mul_257 = mul_258 = None
        unsqueeze_104: "f32[768, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_196, -1)
        unsqueeze_105: "f32[768, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        mul_259: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(mul_253, unsqueeze_105);  mul_253 = unsqueeze_105 = None
        unsqueeze_106: "f32[768, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_197, -1);  primals_197 = None
        unsqueeze_107: "f32[768, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
        add_180: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.add.Tensor(mul_259, unsqueeze_107);  mul_259 = unsqueeze_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.conv1(x)
        convert_element_type_181: "bf16[3072, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(primals_198, torch.bfloat16);  primals_198 = None
        convert_element_type_182: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.prims.convert_element_type.default(add_180, torch.bfloat16);  add_180 = None
        convolution_55: "bf16[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_182, convert_element_type_181, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.act1(x)
        convert_element_type_183: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_55, torch.float32)
        mul_260: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_183, 0.5)
        mul_261: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_183, 0.7071067811865476);  convert_element_type_183 = None
        erf_21: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.erf.default(mul_261);  mul_261 = None
        add_181: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_262: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(mul_260, add_181);  mul_260 = add_181 = None
        convert_element_type_184: "bf16[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.prims.convert_element_type.default(mul_262, torch.bfloat16);  mul_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:73 in forward, code: x = self.conv3(x)
        convert_element_type_185: "bf16[768, 3072, 1, 1][3072, 1, 3072, 3072]cuda:0" = torch.ops.prims.convert_element_type.default(primals_199, torch.bfloat16);  primals_199 = None
        convolution_56: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_184, convert_element_type_185, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_182: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.add.Tensor(add_175, convolution_56);  convolution_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:468 in forward_features, code: x = self.norm(x)
        add_183: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_200, 1)
        var_mean_27 = torch.ops.aten.var_mean.correction(add_182, [0, 2, 3], correction = 0, keepdim = True)
        getitem_78: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = var_mean_27[0]
        getitem_79: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = var_mean_27[1];  var_mean_27 = None
        add_184: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_78, 1e-05)
        rsqrt_27: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_184);  add_184 = None
        sub_35: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(add_182, getitem_79)
        mul_263: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_27);  sub_35 = None
        squeeze_81: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
        squeeze_82: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
        mul_264: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
        mul_265: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_201, 0.9)
        add_185: "f32[768][1]cuda:0" = torch.ops.aten.add.Tensor(mul_264, mul_265);  mul_264 = mul_265 = None
        squeeze_83: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
        mul_266: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0001594642002871);  squeeze_83 = None
        mul_267: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_266, 0.1);  mul_266 = None
        mul_268: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_202, 0.9)
        add_186: "f32[768][1]cuda:0" = torch.ops.aten.add.Tensor(mul_267, mul_268);  mul_267 = mul_268 = None
        unsqueeze_108: "f32[768, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_203, -1)
        unsqueeze_109: "f32[768, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
        mul_269: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(mul_263, unsqueeze_109);  mul_263 = unsqueeze_109 = None
        unsqueeze_110: "f32[768, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_204, -1);  primals_204 = None
        unsqueeze_111: "f32[768, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
        add_187: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.add.Tensor(mul_269, unsqueeze_111);  mul_269 = unsqueeze_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean: "f32[128, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(add_187, [-1, -2], True);  add_187 = None
        as_strided: "f32[128, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.aten.as_strided.default(mean, [128, 768, 1, 1], [768, 1, 768, 768]);  mean = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_64: "f32[128, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(as_strided, [128, 768]);  as_strided = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:474 in forward_head, code: return x if pre_logits else self.head(x)
        convert_element_type_186: "bf16[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_206, torch.bfloat16);  primals_206 = None
        convert_element_type_187: "bf16[1000, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_205, torch.bfloat16);  primals_205 = None
        convert_element_type_188: "bf16[128, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_64, torch.bfloat16);  view_64 = None
        permute_24: "bf16[768, 1000][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_187, [1, 0]);  convert_element_type_187 = None
        addmm: "bf16[128, 1000][1000, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_186, convert_element_type_188, permute_24);  convert_element_type_186 = None
        permute_25: "bf16[1000, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:468 in forward_features, code: x = self.norm(x)
        unsqueeze_112: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
        unsqueeze_113: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_112, 2);  unsqueeze_112 = None
        unsqueeze_114: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_113, 3);  unsqueeze_113 = None
        sub_36: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(add_182, unsqueeze_114);  add_182 = unsqueeze_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        unsqueeze_124: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
        unsqueeze_125: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_124, 2);  unsqueeze_124 = None
        unsqueeze_126: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_125, 3);  unsqueeze_125 = None
        sub_40: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(add_175, unsqueeze_126);  add_175 = unsqueeze_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        permute_30: "bf16[768, 49, 49][2401, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_60, [0, 2, 1]);  view_60 = None
        permute_31: "bf16[768, 128, 49][6272, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_61, [0, 2, 1]);  view_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_32: "bf16[768, 128, 49][6272, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_57, [0, 2, 1]);  view_57 = None
        permute_33: "bf16[768, 49, 128][6272, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_58, [0, 2, 1]);  view_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        unsqueeze_136: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
        unsqueeze_137: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_136, 2);  unsqueeze_136 = None
        unsqueeze_138: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_137, 3);  unsqueeze_137 = None
        sub_44: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(add_169, unsqueeze_138);  add_169 = unsqueeze_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        unsqueeze_148: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
        unsqueeze_149: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_148, 2);  unsqueeze_148 = None
        unsqueeze_150: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_149, 3);  unsqueeze_149 = None
        sub_48: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(add_162, unsqueeze_150);  add_162 = unsqueeze_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        permute_37: "bf16[768, 49, 49][2401, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_52, [0, 2, 1]);  view_52 = None
        permute_38: "bf16[768, 128, 49][6272, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_39: "bf16[768, 128, 49][6272, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_49, [0, 2, 1]);  view_49 = None
        permute_40: "bf16[768, 49, 128][6272, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_50, [0, 2, 1]);  view_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        unsqueeze_160: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
        unsqueeze_161: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_160, 2);  unsqueeze_160 = None
        unsqueeze_162: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_161, 3);  unsqueeze_161 = None
        sub_52: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(add_156, unsqueeze_162);  add_156 = unsqueeze_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        unsqueeze_172: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
        unsqueeze_173: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_172, 2);  unsqueeze_172 = None
        unsqueeze_174: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_173, 3);  unsqueeze_173 = None
        sub_56: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(add_149, unsqueeze_174);  add_149 = unsqueeze_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        permute_44: "bf16[768, 49, 49][2401, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_44, [0, 2, 1]);  view_44 = None
        permute_45: "bf16[768, 128, 49][6272, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_45, [0, 2, 1]);  view_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_46: "bf16[768, 128, 49][6272, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_41, [0, 2, 1]);  view_41 = None
        permute_47: "bf16[768, 49, 128][6272, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_42, [0, 2, 1]);  view_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        unsqueeze_184: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
        unsqueeze_185: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_184, 2);  unsqueeze_184 = None
        unsqueeze_186: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 3);  unsqueeze_185 = None
        sub_60: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(add_143, unsqueeze_186);  add_143 = unsqueeze_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        unsqueeze_196: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
        unsqueeze_197: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_196, 2);  unsqueeze_196 = None
        unsqueeze_198: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 3);  unsqueeze_197 = None
        sub_64: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(add_136, unsqueeze_198);  add_136 = unsqueeze_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        permute_51: "bf16[768, 49, 49][2401, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_36, [0, 2, 1]);  view_36 = None
        permute_52: "bf16[768, 128, 49][6272, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_37, [0, 2, 1]);  view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_53: "bf16[768, 128, 49][6272, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None
        permute_54: "bf16[768, 49, 128][6272, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        unsqueeze_208: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
        unsqueeze_209: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_208, 2);  unsqueeze_208 = None
        unsqueeze_210: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 3);  unsqueeze_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        unsqueeze_232: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
        unsqueeze_233: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 2);  unsqueeze_232 = None
        unsqueeze_234: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 3);  unsqueeze_233 = None
        sub_76: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(add_117, unsqueeze_234);  add_117 = unsqueeze_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        permute_58: "bf16[768, 196, 196][38416, 1, 196]cuda:0" = torch.ops.aten.permute.default(view_28, [0, 2, 1]);  view_28 = None
        permute_59: "bf16[768, 64, 196][12544, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_29, [0, 2, 1]);  view_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_60: "bf16[768, 64, 196][12544, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_25, [0, 2, 1]);  view_25 = None
        permute_61: "bf16[768, 196, 64][12544, 1, 196]cuda:0" = torch.ops.aten.permute.default(view_26, [0, 2, 1]);  view_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        unsqueeze_244: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
        unsqueeze_245: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 2);  unsqueeze_244 = None
        unsqueeze_246: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 3);  unsqueeze_245 = None
        sub_80: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(add_111, unsqueeze_246);  add_111 = unsqueeze_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        unsqueeze_256: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
        unsqueeze_257: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 2);  unsqueeze_256 = None
        unsqueeze_258: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 3);  unsqueeze_257 = None
        sub_84: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(add_104, unsqueeze_258);  add_104 = unsqueeze_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        permute_65: "bf16[768, 196, 196][38416, 1, 196]cuda:0" = torch.ops.aten.permute.default(view_20, [0, 2, 1]);  view_20 = None
        permute_66: "bf16[768, 64, 196][12544, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_21, [0, 2, 1]);  view_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_67: "bf16[768, 64, 196][12544, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_17, [0, 2, 1]);  view_17 = None
        permute_68: "bf16[768, 196, 64][12544, 1, 196]cuda:0" = torch.ops.aten.permute.default(view_18, [0, 2, 1]);  view_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        unsqueeze_268: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
        unsqueeze_269: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 2);  unsqueeze_268 = None
        unsqueeze_270: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 3);  unsqueeze_269 = None
        sub_88: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(add_98, unsqueeze_270);  add_98 = unsqueeze_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        unsqueeze_280: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
        unsqueeze_281: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 2);  unsqueeze_280 = None
        unsqueeze_282: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 3);  unsqueeze_281 = None
        sub_92: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(add_91, unsqueeze_282);  add_91 = unsqueeze_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        permute_72: "bf16[768, 196, 196][38416, 1, 196]cuda:0" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
        permute_73: "bf16[768, 64, 196][12544, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_74: "bf16[768, 64, 196][12544, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        permute_75: "bf16[768, 196, 64][12544, 1, 196]cuda:0" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        unsqueeze_292: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
        unsqueeze_293: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 2);  unsqueeze_292 = None
        unsqueeze_294: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 3);  unsqueeze_293 = None
        sub_96: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(add_85, unsqueeze_294);  add_85 = unsqueeze_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        unsqueeze_304: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
        unsqueeze_305: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_304, 2);  unsqueeze_304 = None
        unsqueeze_306: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 3);  unsqueeze_305 = None
        sub_100: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(add_78, unsqueeze_306);  add_78 = unsqueeze_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        permute_79: "bf16[768, 196, 196][38416, 1, 196]cuda:0" = torch.ops.aten.permute.default(view_4, [0, 2, 1]);  view_4 = None
        permute_80: "bf16[768, 64, 196][12544, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_5, [0, 2, 1]);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_81: "bf16[768, 64, 196][12544, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_1, [0, 2, 1]);  view_1 = None
        permute_82: "bf16[768, 196, 64][12544, 1, 196]cuda:0" = torch.ops.aten.permute.default(view_2, [0, 2, 1]);  view_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        unsqueeze_316: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
        unsqueeze_317: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_316, 2);  unsqueeze_316 = None
        unsqueeze_318: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 3);  unsqueeze_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        unsqueeze_340: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
        unsqueeze_341: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_340, 2);  unsqueeze_340 = None
        unsqueeze_342: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 3);  unsqueeze_341 = None
        sub_112: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(add_58, unsqueeze_342);  add_58 = unsqueeze_342 = None
        unsqueeze_352: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
        unsqueeze_353: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_352, 2);  unsqueeze_352 = None
        unsqueeze_354: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 3);  unsqueeze_353 = None
        sub_116: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(add_50, unsqueeze_354);  add_50 = unsqueeze_354 = None
        unsqueeze_364: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
        unsqueeze_365: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 2);  unsqueeze_364 = None
        unsqueeze_366: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 3);  unsqueeze_365 = None
        sub_120: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(add_42, unsqueeze_366);  add_42 = unsqueeze_366 = None
        unsqueeze_376: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
        unsqueeze_377: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 2);  unsqueeze_376 = None
        unsqueeze_378: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 3);  unsqueeze_377 = None
        sub_124: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(add_34, unsqueeze_378);  add_34 = unsqueeze_378 = None
        unsqueeze_388: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
        unsqueeze_389: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 2);  unsqueeze_388 = None
        unsqueeze_390: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 3);  unsqueeze_389 = None
        sub_128: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(add_26, unsqueeze_390);  add_26 = unsqueeze_390 = None
        unsqueeze_400: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
        unsqueeze_401: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 2);  unsqueeze_400 = None
        unsqueeze_402: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 3);  unsqueeze_401 = None
        sub_132: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(add_18, unsqueeze_402);  add_18 = unsqueeze_402 = None
        unsqueeze_412: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
        unsqueeze_413: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 2);  unsqueeze_412 = None
        unsqueeze_414: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 3);  unsqueeze_413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:437 in forward_features, code: x = self.stem(x)
        unsqueeze_436: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_437: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 2);  unsqueeze_436 = None
        unsqueeze_438: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 3);  unsqueeze_437 = None

        # No stacktrace found for following nodes
        copy_: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_3, add);  primals_3 = add = copy_ = None
        copy__1: "f32[32][1]cuda:0" = torch.ops.aten.copy_.default(primals_4, add_2);  primals_4 = add_2 = copy__1 = None
        copy__2: "f32[32][1]cuda:0" = torch.ops.aten.copy_.default(primals_5, add_3);  primals_5 = add_3 = copy__2 = None
        copy__3: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_10, add_5);  primals_10 = add_5 = copy__3 = None
        copy__4: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_11, add_7);  primals_11 = add_7 = copy__4 = None
        copy__5: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_12, add_8);  primals_12 = add_8 = copy__5 = None
        copy__6: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_16, add_11);  primals_16 = add_11 = copy__6 = None
        copy__7: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_17, add_13);  primals_17 = add_13 = copy__7 = None
        copy__8: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_18, add_14);  primals_18 = add_14 = copy__8 = None
        copy__9: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_24, add_19);  primals_24 = add_19 = copy__9 = None
        copy__10: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_25, add_21);  primals_25 = add_21 = copy__10 = None
        copy__11: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_26, add_22);  primals_26 = add_22 = copy__11 = None
        copy__12: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_32, add_27);  primals_32 = add_27 = copy__12 = None
        copy__13: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_33, add_29);  primals_33 = add_29 = copy__13 = None
        copy__14: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_34, add_30);  primals_34 = add_30 = copy__14 = None
        copy__15: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_40, add_35);  primals_40 = add_35 = copy__15 = None
        copy__16: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_41, add_37);  primals_41 = add_37 = copy__16 = None
        copy__17: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_42, add_38);  primals_42 = add_38 = copy__17 = None
        copy__18: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_48, add_43);  primals_48 = add_43 = copy__18 = None
        copy__19: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_49, add_45);  primals_49 = add_45 = copy__19 = None
        copy__20: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_50, add_46);  primals_50 = add_46 = copy__20 = None
        copy__21: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_56, add_51);  primals_56 = add_51 = copy__21 = None
        copy__22: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_57, add_53);  primals_57 = add_53 = copy__22 = None
        copy__23: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_58, add_54);  primals_58 = add_54 = copy__23 = None
        copy__24: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_64, add_59);  primals_64 = add_59 = copy__24 = None
        copy__25: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_65, add_61);  primals_65 = add_61 = copy__25 = None
        copy__26: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_66, add_62);  primals_66 = add_62 = copy__26 = None
        copy__27: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_74, add_67);  primals_74 = add_67 = copy__27 = None
        copy__28: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_75, add_69);  primals_75 = add_69 = copy__28 = None
        copy__29: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_76, add_70);  primals_76 = add_70 = copy__29 = None
        copy__30: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_80, add_73);  primals_80 = add_73 = copy__30 = None
        copy__31: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_81, add_75);  primals_81 = add_75 = copy__31 = None
        copy__32: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_82, add_76);  primals_82 = add_76 = copy__32 = None
        copy__33: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_87, add_79);  primals_87 = add_79 = copy__33 = None
        copy__34: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_88, add_81);  primals_88 = add_81 = copy__34 = None
        copy__35: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_89, add_82);  primals_89 = add_82 = copy__35 = None
        copy__36: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_94, add_86);  primals_94 = add_86 = copy__36 = None
        copy__37: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_95, add_88);  primals_95 = add_88 = copy__37 = None
        copy__38: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_96, add_89);  primals_96 = add_89 = copy__38 = None
        copy__39: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_101, add_92);  primals_101 = add_92 = copy__39 = None
        copy__40: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_102, add_94);  primals_102 = add_94 = copy__40 = None
        copy__41: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_103, add_95);  primals_103 = add_95 = copy__41 = None
        copy__42: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_108, add_99);  primals_108 = add_99 = copy__42 = None
        copy__43: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_109, add_101);  primals_109 = add_101 = copy__43 = None
        copy__44: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_110, add_102);  primals_110 = add_102 = copy__44 = None
        copy__45: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_115, add_105);  primals_115 = add_105 = copy__45 = None
        copy__46: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_116, add_107);  primals_116 = add_107 = copy__46 = None
        copy__47: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_117, add_108);  primals_117 = add_108 = copy__47 = None
        copy__48: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_122, add_112);  primals_122 = add_112 = copy__48 = None
        copy__49: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_123, add_114);  primals_123 = add_114 = copy__49 = None
        copy__50: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_124, add_115);  primals_124 = add_115 = copy__50 = None
        copy__51: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_129, add_118);  primals_129 = add_118 = copy__51 = None
        copy__52: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_130, add_120);  primals_130 = add_120 = copy__52 = None
        copy__53: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_131, add_121);  primals_131 = add_121 = copy__53 = None
        copy__54: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_138, add_125);  primals_138 = add_125 = copy__54 = None
        copy__55: "f32[768][1]cuda:0" = torch.ops.aten.copy_.default(primals_139, add_127);  primals_139 = add_127 = copy__55 = None
        copy__56: "f32[768][1]cuda:0" = torch.ops.aten.copy_.default(primals_140, add_128);  primals_140 = add_128 = copy__56 = None
        copy__57: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_144, add_131);  primals_144 = add_131 = copy__57 = None
        copy__58: "f32[768][1]cuda:0" = torch.ops.aten.copy_.default(primals_145, add_133);  primals_145 = add_133 = copy__58 = None
        copy__59: "f32[768][1]cuda:0" = torch.ops.aten.copy_.default(primals_146, add_134);  primals_146 = add_134 = copy__59 = None
        copy__60: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_151, add_137);  primals_151 = add_137 = copy__60 = None
        copy__61: "f32[768][1]cuda:0" = torch.ops.aten.copy_.default(primals_152, add_139);  primals_152 = add_139 = copy__61 = None
        copy__62: "f32[768][1]cuda:0" = torch.ops.aten.copy_.default(primals_153, add_140);  primals_153 = add_140 = copy__62 = None
        copy__63: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_158, add_144);  primals_158 = add_144 = copy__63 = None
        copy__64: "f32[768][1]cuda:0" = torch.ops.aten.copy_.default(primals_159, add_146);  primals_159 = add_146 = copy__64 = None
        copy__65: "f32[768][1]cuda:0" = torch.ops.aten.copy_.default(primals_160, add_147);  primals_160 = add_147 = copy__65 = None
        copy__66: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_165, add_150);  primals_165 = add_150 = copy__66 = None
        copy__67: "f32[768][1]cuda:0" = torch.ops.aten.copy_.default(primals_166, add_152);  primals_166 = add_152 = copy__67 = None
        copy__68: "f32[768][1]cuda:0" = torch.ops.aten.copy_.default(primals_167, add_153);  primals_167 = add_153 = copy__68 = None
        copy__69: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_172, add_157);  primals_172 = add_157 = copy__69 = None
        copy__70: "f32[768][1]cuda:0" = torch.ops.aten.copy_.default(primals_173, add_159);  primals_173 = add_159 = copy__70 = None
        copy__71: "f32[768][1]cuda:0" = torch.ops.aten.copy_.default(primals_174, add_160);  primals_174 = add_160 = copy__71 = None
        copy__72: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_179, add_163);  primals_179 = add_163 = copy__72 = None
        copy__73: "f32[768][1]cuda:0" = torch.ops.aten.copy_.default(primals_180, add_165);  primals_180 = add_165 = copy__73 = None
        copy__74: "f32[768][1]cuda:0" = torch.ops.aten.copy_.default(primals_181, add_166);  primals_181 = add_166 = copy__74 = None
        copy__75: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_186, add_170);  primals_186 = add_170 = copy__75 = None
        copy__76: "f32[768][1]cuda:0" = torch.ops.aten.copy_.default(primals_187, add_172);  primals_187 = add_172 = copy__76 = None
        copy__77: "f32[768][1]cuda:0" = torch.ops.aten.copy_.default(primals_188, add_173);  primals_188 = add_173 = copy__77 = None
        copy__78: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_193, add_176);  primals_193 = add_176 = copy__78 = None
        copy__79: "f32[768][1]cuda:0" = torch.ops.aten.copy_.default(primals_194, add_178);  primals_194 = add_178 = copy__79 = None
        copy__80: "f32[768][1]cuda:0" = torch.ops.aten.copy_.default(primals_195, add_179);  primals_195 = add_179 = copy__80 = None
        copy__81: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_200, add_183);  primals_200 = add_183 = copy__81 = None
        copy__82: "f32[768][1]cuda:0" = torch.ops.aten.copy_.default(primals_201, add_185);  primals_201 = add_185 = copy__82 = None
        copy__83: "f32[768][1]cuda:0" = torch.ops.aten.copy_.default(primals_202, add_186);  primals_202 = add_186 = copy__83 = None
        return (addmm, primals_6, primals_13, primals_14, primals_15, primals_19, primals_27, primals_35, primals_43, primals_51, primals_59, primals_67, primals_77, primals_78, primals_79, primals_83, primals_90, primals_97, primals_104, primals_111, primals_118, primals_125, primals_132, primals_141, primals_142, primals_143, primals_147, primals_154, primals_161, primals_168, primals_175, primals_182, primals_189, primals_196, primals_203, convert_element_type, convert_element_type_1, convolution, squeeze_1, relu, convert_element_type_5, convolution_1, getitem_3, rsqrt_1, squeeze_7, convert_element_type_8, convert_element_type_9, convolution_2, convert_element_type_11, convert_element_type_12, convolution_3, convert_element_type_14, convert_element_type_15, squeeze_10, convert_element_type_16, convert_element_type_17, convolution_5, convert_element_type_19, convert_element_type_20, convolution_6, convert_element_type_22, convert_element_type_23, squeeze_13, convert_element_type_24, convert_element_type_25, convolution_8, convert_element_type_27, convert_element_type_28, convolution_9, convert_element_type_30, convert_element_type_31, squeeze_16, convert_element_type_32, convert_element_type_33, convolution_11, convert_element_type_35, convert_element_type_36, convolution_12, convert_element_type_38, convert_element_type_39, squeeze_19, convert_element_type_40, convert_element_type_41, convolution_14, convert_element_type_43, convert_element_type_44, convolution_15, convert_element_type_46, convert_element_type_47, squeeze_22, convert_element_type_48, convert_element_type_49, convolution_17, convert_element_type_51, convert_element_type_52, convolution_18, convert_element_type_54, convert_element_type_55, squeeze_25, convert_element_type_56, convert_element_type_57, convolution_20, convert_element_type_59, convert_element_type_60, convolution_21, convert_element_type_62, convert_element_type_63, convert_element_type_65, convert_element_type_66, convolution_23, getitem_19, rsqrt_9, squeeze_31, convert_element_type_69, convert_element_type_70, bmm_default_11, amax_default_14, amax_default_15, logical_not_default_15, sum_1, view_7, convert_element_type_77, squeeze_34, convert_element_type_78, convert_element_type_79, convolution_26, convert_element_type_81, convert_element_type_82, squeeze_37, convert_element_type_83, convert_element_type_84, bmm_default_9, amax_default_12, amax_default_13, logical_not_default_13, sum_2, view_15, convert_element_type_91, squeeze_40, convert_element_type_92, convert_element_type_93, convolution_30, convert_element_type_95, convert_element_type_96, squeeze_43, convert_element_type_97, convert_element_type_98, bmm_default_7, amax_default_10, amax_default_11, logical_not_default_11, sum_3, view_23, convert_element_type_105, squeeze_46, convert_element_type_106, convert_element_type_107, convolution_34, convert_element_type_109, convert_element_type_110, squeeze_49, convert_element_type_111, convert_element_type_112, bmm_default_5, amax_default_8, amax_default_9, logical_not_default_9, sum_4, view_31, convert_element_type_119, squeeze_52, convert_element_type_120, convert_element_type_121, convolution_38, convert_element_type_123, convert_element_type_124, convert_element_type_126, convert_element_type_127, convolution_40, getitem_49, rsqrt_18, squeeze_58, convert_element_type_130, convert_element_type_131, bmm_default_3, amax_default_6, amax_default_7, logical_not_default_7, sum_5, view_39, convert_element_type_138, squeeze_61, convert_element_type_139, convert_element_type_140, convolution_43, convert_element_type_142, convert_element_type_143, squeeze_64, convert_element_type_144, convert_element_type_145, bmm_default_2, amax_default_4, amax_default_5, logical_not_default_5, sum_6, view_47, convert_element_type_152, squeeze_67, convert_element_type_153, convert_element_type_154, convolution_47, convert_element_type_156, convert_element_type_157, squeeze_70, convert_element_type_158, convert_element_type_159, bmm_default_1, amax_default_2, amax_default_3, logical_not_default_3, sum_7, view_55, convert_element_type_166, squeeze_73, convert_element_type_167, convert_element_type_168, convolution_51, convert_element_type_170, convert_element_type_171, squeeze_76, convert_element_type_172, convert_element_type_173, bmm_default, amax_default, amax_default_1, logical_not_default_1, sum_8, view_63, convert_element_type_180, squeeze_79, convert_element_type_181, convert_element_type_182, convolution_55, convert_element_type_184, convert_element_type_185, squeeze_82, convert_element_type_188, permute_25, sub_36, sub_40, permute_30, permute_31, permute_32, permute_33, sub_44, sub_48, permute_37, permute_38, permute_39, permute_40, sub_52, sub_56, permute_44, permute_45, permute_46, permute_47, sub_60, sub_64, permute_51, permute_52, permute_53, permute_54, unsqueeze_210, sub_76, permute_58, permute_59, permute_60, permute_61, sub_80, sub_84, permute_65, permute_66, permute_67, permute_68, sub_88, sub_92, permute_72, permute_73, permute_74, permute_75, sub_96, sub_100, permute_79, permute_80, permute_81, permute_82, unsqueeze_318, sub_112, sub_116, sub_120, sub_124, sub_128, sub_132, unsqueeze_414, unsqueeze_438)
