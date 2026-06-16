class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[16, 3, 3, 3][27, 1, 9, 3]cuda:0", primals_2: "f32[16, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_3: "f32[16][1]cuda:0", primals_4: "f32[128, 3, 224, 224][150528, 1, 672, 3]cuda:0", primals_5: "f32[32, 16, 3, 3][144, 1, 48, 16]cuda:0", primals_6: "f32[32, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_7: "f32[32][1]cuda:0", primals_8: "f32[64, 32, 3, 3][288, 1, 96, 32]cuda:0", primals_9: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_10: "f32[64][1]cuda:0", primals_11: "f32[128, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_12: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_13: "f32[128][1]cuda:0", primals_14: "f32[256, 128, 1, 1][128, 1, 128, 128]cuda:0", primals_15: "f32[256, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_16: "f32[256][1]cuda:0", primals_17: "f32[64, 128, 1, 1][128, 1, 128, 128]cuda:0", primals_18: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_19: "f32[64][1]cuda:0", primals_20: "f32[64, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_21: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_22: "f32[64][1]cuda:0", primals_23: "f32[64, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_24: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_25: "f32[64][1]cuda:0", primals_26: "f32[256, 64, 1, 1][64, 1, 64, 64]cuda:0", primals_27: "f32[256, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_28: "f32[256][1]cuda:0", primals_29: "f32[64, 256, 1, 1][256, 1, 256, 256]cuda:0", primals_30: "f32[64][1]cuda:0", primals_31: "f32[256, 64, 1, 1][64, 1, 64, 64]cuda:0", primals_32: "f32[256][1]cuda:0", primals_33: "f32[512, 256, 1, 1][256, 1, 256, 256]cuda:0", primals_34: "f32[512, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_35: "f32[512][1]cuda:0", primals_36: "f32[128, 256, 1, 1][256, 1, 256, 256]cuda:0", primals_37: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_38: "f32[128][1]cuda:0", primals_39: "f32[128, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_40: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_41: "f32[128][1]cuda:0", primals_42: "f32[128, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_43: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_44: "f32[128][1]cuda:0", primals_45: "f32[512, 128, 1, 1][128, 1, 128, 128]cuda:0", primals_46: "f32[512, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_47: "f32[512][1]cuda:0", primals_48: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0", primals_49: "f32[128][1]cuda:0", primals_50: "f32[512, 128, 1, 1][128, 1, 128, 128]cuda:0", primals_51: "f32[512][1]cuda:0", primals_52: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0", primals_53: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_54: "f32[128][1]cuda:0", primals_55: "f32[128, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_56: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_57: "f32[128][1]cuda:0", primals_58: "f32[128, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_59: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_60: "f32[128][1]cuda:0", primals_61: "f32[512, 128, 1, 1][128, 1, 128, 128]cuda:0", primals_62: "f32[512, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_63: "f32[512][1]cuda:0", primals_64: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0", primals_65: "f32[128][1]cuda:0", primals_66: "f32[512, 128, 1, 1][128, 1, 128, 128]cuda:0", primals_67: "f32[512][1]cuda:0", primals_68: "f32[1536, 512, 1, 1][512, 1, 512, 512]cuda:0", primals_69: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_70: "f32[1536][1]cuda:0", primals_71: "f32[384, 512, 1, 1][512, 1, 512, 512]cuda:0", primals_72: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_73: "f32[384][1]cuda:0", primals_74: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_75: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_76: "f32[384][1]cuda:0", primals_77: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_78: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_79: "f32[384][1]cuda:0", primals_80: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_81: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_82: "f32[1536][1]cuda:0", primals_83: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_84: "f32[384][1]cuda:0", primals_85: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_86: "f32[1536][1]cuda:0", primals_87: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_88: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_89: "f32[384][1]cuda:0", primals_90: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_91: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_92: "f32[384][1]cuda:0", primals_93: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_94: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_95: "f32[384][1]cuda:0", primals_96: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_97: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_98: "f32[1536][1]cuda:0", primals_99: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_100: "f32[384][1]cuda:0", primals_101: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_102: "f32[1536][1]cuda:0", primals_103: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_104: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_105: "f32[384][1]cuda:0", primals_106: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_107: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_108: "f32[384][1]cuda:0", primals_109: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_110: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_111: "f32[384][1]cuda:0", primals_112: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_113: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_114: "f32[1536][1]cuda:0", primals_115: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_116: "f32[384][1]cuda:0", primals_117: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_118: "f32[1536][1]cuda:0", primals_119: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_120: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_121: "f32[384][1]cuda:0", primals_122: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_123: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_124: "f32[384][1]cuda:0", primals_125: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_126: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_127: "f32[384][1]cuda:0", primals_128: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_129: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_130: "f32[1536][1]cuda:0", primals_131: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_132: "f32[384][1]cuda:0", primals_133: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_134: "f32[1536][1]cuda:0", primals_135: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_136: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_137: "f32[384][1]cuda:0", primals_138: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_139: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_140: "f32[384][1]cuda:0", primals_141: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_142: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_143: "f32[384][1]cuda:0", primals_144: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_145: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_146: "f32[1536][1]cuda:0", primals_147: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_148: "f32[384][1]cuda:0", primals_149: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_150: "f32[1536][1]cuda:0", primals_151: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_152: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_153: "f32[384][1]cuda:0", primals_154: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_155: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_156: "f32[384][1]cuda:0", primals_157: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_158: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_159: "f32[384][1]cuda:0", primals_160: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_161: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_162: "f32[1536][1]cuda:0", primals_163: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_164: "f32[384][1]cuda:0", primals_165: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_166: "f32[1536][1]cuda:0", primals_167: "f32[1536, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_168: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_169: "f32[1536][1]cuda:0", primals_170: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_171: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_172: "f32[384][1]cuda:0", primals_173: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_174: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_175: "f32[384][1]cuda:0", primals_176: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_177: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_178: "f32[384][1]cuda:0", primals_179: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_180: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_181: "f32[1536][1]cuda:0", primals_182: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_183: "f32[384][1]cuda:0", primals_184: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_185: "f32[1536][1]cuda:0", primals_186: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_187: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_188: "f32[384][1]cuda:0", primals_189: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_190: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_191: "f32[384][1]cuda:0", primals_192: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_193: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_194: "f32[384][1]cuda:0", primals_195: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_196: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_197: "f32[1536][1]cuda:0", primals_198: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_199: "f32[384][1]cuda:0", primals_200: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_201: "f32[1536][1]cuda:0", primals_202: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_203: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_204: "f32[384][1]cuda:0", primals_205: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_206: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_207: "f32[384][1]cuda:0", primals_208: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_209: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_210: "f32[384][1]cuda:0", primals_211: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_212: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_213: "f32[1536][1]cuda:0", primals_214: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_215: "f32[384][1]cuda:0", primals_216: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_217: "f32[1536][1]cuda:0", primals_218: "f32[2304, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_219: "f32[2304, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_220: "f32[2304][1]cuda:0", primals_221: "f32[1000, 2304][2304, 1]cuda:0", primals_222: "f32[1000][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone: "f32[16, 3, 3, 3][27, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_1, memory_format = torch.contiguous_format)
        view: "f32[1, 16, 27][432, 27, 1]cuda:0" = torch.ops.aten.reshape.default(clone, [1, 16, 27]);  clone = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul: "f32[16, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_2, 0.34412564994580647)
        view_1: "f32[16][1]cuda:0" = torch.ops.aten.reshape.default(mul, [-1]);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean = torch.ops.aten.var_mean.correction(view, [0, 2], correction = 0, keepdim = True)
        getitem: "f32[1, 16, 1][16, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[1, 16, 1][16, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        add: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[1, 16, 27][432, 27, 1]cuda:0" = torch.ops.aten.sub.Tensor(view, getitem_1);  view = None
        mul_1: "f32[1, 16, 27][432, 27, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze: "f32[16][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2]);  getitem_1 = None
        squeeze_1: "f32[16][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2]);  rsqrt = None
        unsqueeze: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_1, -1);  view_1 = None
        mul_2: "f32[1, 16, 27][432, 27, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze);  mul_1 = unsqueeze = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_2: "f32[16, 3, 3, 3][27, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_2, [16, 3, 3, 3]);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type: "bf16[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_3, torch.bfloat16);  primals_3 = None
        convert_element_type_1: "bf16[16, 3, 3, 3][27, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16);  view_2 = None
        convert_element_type_2: "bf16[128, 3, 224, 224][150528, 1, 672, 3]cuda:0" = torch.ops.prims.convert_element_type.default(primals_4, torch.bfloat16);  primals_4 = None
        convolution: "bf16[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_2, convert_element_type_1, convert_element_type, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:563 in forward_features, code: x = self.stem(x)
        convert_element_type_3: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.prims.convert_element_type.default(convolution, torch.float32)
        neg: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.neg.default(convert_element_type_3)
        exp: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.exp.default(neg);  neg = None
        add_1: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_3, add_1);  convert_element_type_3 = add_1 = None
        convert_element_type_4: "bf16[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_2: "f32[32, 16, 3, 3][144, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_5, memory_format = torch.contiguous_format)
        view_3: "f32[1, 32, 144][4608, 144, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [1, 32, 144]);  clone_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_3: "f32[32, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_6, 0.1490107774734497)
        view_4: "f32[32][1]cuda:0" = torch.ops.aten.reshape.default(mul_3, [-1]);  mul_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_1 = torch.ops.aten.var_mean.correction(view_3, [0, 2], correction = 0, keepdim = True)
        getitem_2: "f32[1, 32, 1][32, 1, 1]cuda:0" = var_mean_1[0]
        getitem_3: "f32[1, 32, 1][32, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        add_2: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub_1: "f32[1, 32, 144][4608, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_3, getitem_3);  view_3 = None
        mul_4: "f32[1, 32, 144][4608, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        squeeze_2: "f32[32][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2]);  getitem_3 = None
        squeeze_3: "f32[32][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2]);  rsqrt_1 = None
        unsqueeze_1: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_4, -1);  view_4 = None
        mul_5: "f32[1, 32, 144][4608, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_1);  mul_4 = unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_5: "f32[32, 16, 3, 3][144, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_5, [32, 16, 3, 3]);  mul_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_5: "bf16[32][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_7, torch.bfloat16);  primals_7 = None
        convert_element_type_6: "bf16[32, 16, 3, 3][144, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16);  view_5 = None
        convolution_1: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_4, convert_element_type_6, convert_element_type_5, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:563 in forward_features, code: x = self.stem(x)
        convert_element_type_7: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32)
        neg_1: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.neg.default(convert_element_type_7)
        exp_1: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.exp.default(neg_1);  neg_1 = None
        add_3: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.add.Tensor(exp_1, 1);  exp_1 = None
        div_1: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_7, add_3);  convert_element_type_7 = add_3 = None
        convert_element_type_8: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_4: "f32[64, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_8, memory_format = torch.contiguous_format)
        view_6: "f32[1, 64, 288][18432, 288, 1]cuda:0" = torch.ops.aten.reshape.default(clone_4, [1, 64, 288]);  clone_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_6: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_9, 0.10536653122135592)
        view_7: "f32[64][1]cuda:0" = torch.ops.aten.reshape.default(mul_6, [-1]);  mul_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_2 = torch.ops.aten.var_mean.correction(view_6, [0, 2], correction = 0, keepdim = True)
        getitem_4: "f32[1, 64, 1][64, 1, 1]cuda:0" = var_mean_2[0]
        getitem_5: "f32[1, 64, 1][64, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        add_4: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_2: "f32[1, 64, 288][18432, 288, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_6, getitem_5);  view_6 = None
        mul_7: "f32[1, 64, 288][18432, 288, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        squeeze_4: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2]);  getitem_5 = None
        squeeze_5: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2]);  rsqrt_2 = None
        unsqueeze_2: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_7, -1);  view_7 = None
        mul_8: "f32[1, 64, 288][18432, 288, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_2);  mul_7 = unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_8: "f32[64, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_8, [64, 32, 3, 3]);  mul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_9: "bf16[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_10, torch.bfloat16);  primals_10 = None
        convert_element_type_10: "bf16[64, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_8, torch.bfloat16);  view_8 = None
        convolution_2: "bf16[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_8, convert_element_type_10, convert_element_type_9, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:563 in forward_features, code: x = self.stem(x)
        convert_element_type_11: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_2, torch.float32)
        neg_2: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.neg.default(convert_element_type_11)
        exp_2: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.exp.default(neg_2);  neg_2 = None
        add_5: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.add.Tensor(exp_2, 1);  exp_2 = None
        div_2: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_11, add_5);  convert_element_type_11 = add_5 = None
        convert_element_type_12: "bf16[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_6: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_11, memory_format = torch.contiguous_format)
        view_9: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_6, [1, 128, 576]);  clone_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_9: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_12, 0.07450538873672485)
        view_10: "f32[128][1]cuda:0" = torch.ops.aten.reshape.default(mul_9, [-1]);  mul_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_3 = torch.ops.aten.var_mean.correction(view_9, [0, 2], correction = 0, keepdim = True)
        getitem_6: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_3[0]
        getitem_7: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        add_6: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_3: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_9, getitem_7);  view_9 = None
        mul_10: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        squeeze_6: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2]);  getitem_7 = None
        squeeze_7: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2]);  rsqrt_3 = None
        unsqueeze_3: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_10, -1);  view_10 = None
        mul_11: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, unsqueeze_3);  mul_10 = unsqueeze_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_11: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_11, [128, 64, 3, 3]);  mul_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_13: "bf16[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_13, torch.bfloat16);  primals_13 = None
        convert_element_type_14: "bf16[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_11, torch.bfloat16);  view_11 = None
        convolution_3: "bf16[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_12, convert_element_type_14, convert_element_type_13, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        convert_element_type_15: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_3, torch.float32)
        neg_3: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.neg.default(convert_element_type_15)
        exp_3: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.exp.default(neg_3);  neg_3 = None
        add_7: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.add.Tensor(exp_3, 1);  exp_3 = None
        div_3: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_15, add_7);  convert_element_type_15 = add_7 = None
        convert_element_type_16: "bf16[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None
        mul_12: "bf16[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.0);  convert_element_type_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_12: "f32[1, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(primals_14, [1, 256, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_13: "f32[256, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_15, 0.1580497968320339)
        view_13: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(mul_13, [-1]);  mul_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_4 = torch.ops.aten.var_mean.correction(view_12, [0, 2], correction = 0, keepdim = True)
        getitem_8: "f32[1, 256, 1][256, 1, 1]cuda:0" = var_mean_4[0]
        getitem_9: "f32[1, 256, 1][256, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        add_8: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_4: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_4: "f32[1, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_12, getitem_9);  view_12 = None
        mul_14: "f32[1, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        squeeze_8: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2]);  getitem_9 = None
        squeeze_9: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2]);  rsqrt_4 = None
        unsqueeze_4: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_13, -1);  view_13 = None
        mul_15: "f32[1, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_4);  mul_14 = unsqueeze_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_14: "f32[256, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_15, [256, 128, 1, 1]);  mul_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_17: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_16, torch.bfloat16);  primals_16 = None
        convert_element_type_18: "bf16[256, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_14, torch.bfloat16);  view_14 = None
        convolution_4: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.convolution.default(mul_12, convert_element_type_18, convert_element_type_17, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_15: "f32[1, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(primals_17, [1, 64, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_16: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_18, 0.1580497968320339)
        view_16: "f32[64][1]cuda:0" = torch.ops.aten.reshape.default(mul_16, [-1]);  mul_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_5 = torch.ops.aten.var_mean.correction(view_15, [0, 2], correction = 0, keepdim = True)
        getitem_10: "f32[1, 64, 1][64, 1, 1]cuda:0" = var_mean_5[0]
        getitem_11: "f32[1, 64, 1][64, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        add_9: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_5: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        sub_5: "f32[1, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_15, getitem_11);  view_15 = None
        mul_17: "f32[1, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        squeeze_10: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2]);  getitem_11 = None
        squeeze_11: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2]);  rsqrt_5 = None
        unsqueeze_5: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_16, -1);  view_16 = None
        mul_18: "f32[1, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_17, unsqueeze_5);  mul_17 = unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_17: "f32[64, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_18, [64, 128, 1, 1]);  mul_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_19: "bf16[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_19, torch.bfloat16);  primals_19 = None
        convert_element_type_20: "bf16[64, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_17, torch.bfloat16);  view_17 = None
        convolution_5: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.convolution.default(mul_12, convert_element_type_20, convert_element_type_19, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        convert_element_type_21: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_5, torch.float32)
        neg_4: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.neg.default(convert_element_type_21)
        exp_4: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.exp.default(neg_4);  neg_4 = None
        add_10: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.add.Tensor(exp_4, 1);  exp_4 = None
        div_4: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_21, add_10);  convert_element_type_21 = add_10 = None
        convert_element_type_22: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_8: "f32[64, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_20, memory_format = torch.contiguous_format)
        view_18: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_8, [1, 64, 576]);  clone_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_19: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_21, 0.07450538873672485)
        view_19: "f32[64][1]cuda:0" = torch.ops.aten.reshape.default(mul_19, [-1]);  mul_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_6 = torch.ops.aten.var_mean.correction(view_18, [0, 2], correction = 0, keepdim = True)
        getitem_12: "f32[1, 64, 1][64, 1, 1]cuda:0" = var_mean_6[0]
        getitem_13: "f32[1, 64, 1][64, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        add_11: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_6: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_6: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_18, getitem_13);  view_18 = None
        mul_20: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        squeeze_12: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2]);  getitem_13 = None
        squeeze_13: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2]);  rsqrt_6 = None
        unsqueeze_6: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_19, -1);  view_19 = None
        mul_21: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_20, unsqueeze_6);  mul_20 = unsqueeze_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_20: "f32[64, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_21, [64, 64, 3, 3]);  mul_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_23: "bf16[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_22, torch.bfloat16);  primals_22 = None
        convert_element_type_24: "bf16[64, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_20, torch.bfloat16);  view_20 = None
        convolution_6: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_22, convert_element_type_24, convert_element_type_23, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        convert_element_type_25: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_6, torch.float32)
        neg_5: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.neg.default(convert_element_type_25)
        exp_5: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.exp.default(neg_5);  neg_5 = None
        add_12: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.add.Tensor(exp_5, 1);  exp_5 = None
        div_5: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_25, add_12);  convert_element_type_25 = add_12 = None
        convert_element_type_26: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_10: "f32[64, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_23, memory_format = torch.contiguous_format)
        view_21: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_10, [1, 64, 576]);  clone_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_22: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_24, 0.07450538873672485)
        view_22: "f32[64][1]cuda:0" = torch.ops.aten.reshape.default(mul_22, [-1]);  mul_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_7 = torch.ops.aten.var_mean.correction(view_21, [0, 2], correction = 0, keepdim = True)
        getitem_14: "f32[1, 64, 1][64, 1, 1]cuda:0" = var_mean_7[0]
        getitem_15: "f32[1, 64, 1][64, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        add_13: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_7: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_7: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_21, getitem_15);  view_21 = None
        mul_23: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        squeeze_14: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2]);  getitem_15 = None
        squeeze_15: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2]);  rsqrt_7 = None
        unsqueeze_7: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_22, -1);  view_22 = None
        mul_24: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_23, unsqueeze_7);  mul_23 = unsqueeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_23: "f32[64, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_24, [64, 64, 3, 3]);  mul_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_27: "bf16[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_25, torch.bfloat16);  primals_25 = None
        convert_element_type_28: "bf16[64, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_23, torch.bfloat16);  view_23 = None
        convolution_7: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_26, convert_element_type_28, convert_element_type_27, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        convert_element_type_29: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_7, torch.float32)
        neg_6: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.neg.default(convert_element_type_29)
        exp_6: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.exp.default(neg_6);  neg_6 = None
        add_14: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.add.Tensor(exp_6, 1);  exp_6 = None
        div_6: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_29, add_14);  convert_element_type_29 = add_14 = None
        convert_element_type_30: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_24: "f32[1, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(primals_26, [1, 256, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_25: "f32[256, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_27, 0.22351616621017456)
        view_25: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(mul_25, [-1]);  mul_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_8 = torch.ops.aten.var_mean.correction(view_24, [0, 2], correction = 0, keepdim = True)
        getitem_16: "f32[1, 256, 1][256, 1, 1]cuda:0" = var_mean_8[0]
        getitem_17: "f32[1, 256, 1][256, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        add_15: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_8: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        sub_8: "f32[1, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_24, getitem_17);  view_24 = None
        mul_26: "f32[1, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        squeeze_16: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2]);  getitem_17 = None
        squeeze_17: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2]);  rsqrt_8 = None
        unsqueeze_8: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_25, -1);  view_25 = None
        mul_27: "f32[1, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_26, unsqueeze_8);  mul_26 = unsqueeze_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_26: "f32[256, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_27, [256, 64, 1, 1]);  mul_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_31: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_28, torch.bfloat16);  primals_28 = None
        convert_element_type_32: "bf16[256, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_26, torch.bfloat16);  view_26 = None
        convolution_8: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_30, convert_element_type_32, convert_element_type_31, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean: "bf16[128, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convolution_8, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convert_element_type_33: "bf16[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_30, torch.bfloat16);  primals_30 = None
        convert_element_type_34: "bf16[64, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(primals_29, torch.bfloat16);  primals_29 = None
        convolution_9: "bf16[128, 64, 1, 1][64, 1, 64, 64]cuda:0" = torch.ops.aten.convolution.default(mean, convert_element_type_34, convert_element_type_33, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu: "bf16[128, 64, 1, 1][64, 1, 64, 64]cuda:0" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convert_element_type_35: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_32, torch.bfloat16);  primals_32 = None
        convert_element_type_36: "bf16[256, 64, 1, 1][64, 1, 64, 64]cuda:0" = torch.ops.prims.convert_element_type.default(primals_31, torch.bfloat16);  primals_31 = None
        convolution_10: "bf16[128, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.aten.convolution.default(relu, convert_element_type_36, convert_element_type_35, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid: "bf16[128, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.aten.sigmoid.default(convolution_10)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_28: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(convolution_8, sigmoid);  sigmoid = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_29: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, 2.0);  mul_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_30: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_29, 0.2);  mul_29 = None
        add_16: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_30, convolution_4);  mul_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        convert_element_type_37: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.prims.convert_element_type.default(add_16, torch.float32);  add_16 = None
        neg_7: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.neg.default(convert_element_type_37)
        exp_7: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.exp.default(neg_7);  neg_7 = None
        add_17: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.add.Tensor(exp_7, 1);  exp_7 = None
        div_7: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_37, add_17);  convert_element_type_37 = add_17 = None
        convert_element_type_38: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None
        mul_31: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_38, 0.9805806756909201);  convert_element_type_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:149 in forward, code: return self.conv(self.pool(x))
        avg_pool2d: "bf16[128, 256, 28, 28][200704, 1, 7168, 256]cuda:0" = torch.ops.aten.avg_pool2d.default(mul_31, [2, 2], [2, 2], [0, 0], True, False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_27: "f32[1, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(primals_33, [1, 512, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_32: "f32[512, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_34, 0.11175808310508728)
        view_28: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(mul_32, [-1]);  mul_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_9 = torch.ops.aten.var_mean.correction(view_27, [0, 2], correction = 0, keepdim = True)
        getitem_18: "f32[1, 512, 1][512, 1, 1]cuda:0" = var_mean_9[0]
        getitem_19: "f32[1, 512, 1][512, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        add_18: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_9: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_9: "f32[1, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_27, getitem_19);  view_27 = None
        mul_33: "f32[1, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        squeeze_18: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2]);  getitem_19 = None
        squeeze_19: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2]);  rsqrt_9 = None
        unsqueeze_9: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_28, -1);  view_28 = None
        mul_34: "f32[1, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_33, unsqueeze_9);  mul_33 = unsqueeze_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_29: "f32[512, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_34, [512, 256, 1, 1]);  mul_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_39: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_35, torch.bfloat16);  primals_35 = None
        convert_element_type_40: "bf16[512, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_29, torch.bfloat16);  view_29 = None
        convolution_11: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.convolution.default(avg_pool2d, convert_element_type_40, convert_element_type_39, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_30: "f32[1, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(primals_36, [1, 128, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_35: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_37, 0.11175808310508728)
        view_31: "f32[128][1]cuda:0" = torch.ops.aten.reshape.default(mul_35, [-1]);  mul_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_10 = torch.ops.aten.var_mean.correction(view_30, [0, 2], correction = 0, keepdim = True)
        getitem_20: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_10[0]
        getitem_21: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        add_19: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_10: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        sub_10: "f32[1, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_30, getitem_21);  view_30 = None
        mul_36: "f32[1, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        squeeze_20: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2]);  getitem_21 = None
        squeeze_21: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2]);  rsqrt_10 = None
        unsqueeze_10: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_31, -1);  view_31 = None
        mul_37: "f32[1, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_36, unsqueeze_10);  mul_36 = unsqueeze_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_32: "f32[128, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_37, [128, 256, 1, 1]);  mul_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_41: "bf16[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_38, torch.bfloat16);  primals_38 = None
        convert_element_type_42: "bf16[128, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_32, torch.bfloat16);  view_32 = None
        convolution_12: "bf16[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.convolution.default(mul_31, convert_element_type_42, convert_element_type_41, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        convert_element_type_43: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_12, torch.float32)
        neg_8: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.neg.default(convert_element_type_43)
        exp_8: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.exp.default(neg_8);  neg_8 = None
        add_20: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.add.Tensor(exp_8, 1);  exp_8 = None
        div_8: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_43, add_20);  convert_element_type_43 = add_20 = None
        convert_element_type_44: "bf16[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_12: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_39, memory_format = torch.contiguous_format)
        view_33: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_12, [1, 128, 576]);  clone_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_38: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_40, 0.07450538873672485)
        view_34: "f32[128][1]cuda:0" = torch.ops.aten.reshape.default(mul_38, [-1]);  mul_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_11 = torch.ops.aten.var_mean.correction(view_33, [0, 2], correction = 0, keepdim = True)
        getitem_22: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_11[0]
        getitem_23: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        add_21: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_11: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_11: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_33, getitem_23);  view_33 = None
        mul_39: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        squeeze_22: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2]);  getitem_23 = None
        squeeze_23: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2]);  rsqrt_11 = None
        unsqueeze_11: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_34, -1);  view_34 = None
        mul_40: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_39, unsqueeze_11);  mul_39 = unsqueeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_35: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_40, [128, 64, 3, 3]);  mul_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_45: "bf16[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_41, torch.bfloat16);  primals_41 = None
        convert_element_type_46: "bf16[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_35, torch.bfloat16);  view_35 = None
        convolution_13: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_44, convert_element_type_46, convert_element_type_45, [2, 2], [1, 1], [1, 1], False, [0, 0], 2);  convert_element_type_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        convert_element_type_47: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_13, torch.float32)
        neg_9: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.neg.default(convert_element_type_47)
        exp_9: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.exp.default(neg_9);  neg_9 = None
        add_22: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.add.Tensor(exp_9, 1);  exp_9 = None
        div_9: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_47, add_22);  convert_element_type_47 = add_22 = None
        convert_element_type_48: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_14: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_42, memory_format = torch.contiguous_format)
        view_36: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_14, [1, 128, 576]);  clone_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_41: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_43, 0.07450538873672485)
        view_37: "f32[128][1]cuda:0" = torch.ops.aten.reshape.default(mul_41, [-1]);  mul_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_12 = torch.ops.aten.var_mean.correction(view_36, [0, 2], correction = 0, keepdim = True)
        getitem_24: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_12[0]
        getitem_25: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        add_23: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_12: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        sub_12: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_36, getitem_25);  view_36 = None
        mul_42: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        squeeze_24: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2]);  getitem_25 = None
        squeeze_25: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2]);  rsqrt_12 = None
        unsqueeze_12: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_37, -1);  view_37 = None
        mul_43: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_12);  mul_42 = unsqueeze_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_38: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_43, [128, 64, 3, 3]);  mul_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_49: "bf16[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16);  primals_44 = None
        convert_element_type_50: "bf16[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_38, torch.bfloat16);  view_38 = None
        convolution_14: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_48, convert_element_type_50, convert_element_type_49, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  convert_element_type_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        convert_element_type_51: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_14, torch.float32)
        neg_10: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.neg.default(convert_element_type_51)
        exp_10: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.exp.default(neg_10);  neg_10 = None
        add_24: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.add.Tensor(exp_10, 1);  exp_10 = None
        div_10: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_51, add_24);  convert_element_type_51 = add_24 = None
        convert_element_type_52: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_39: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.reshape.default(primals_45, [1, 512, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_44: "f32[512, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_46, 0.1580497968320339)
        view_40: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(mul_44, [-1]);  mul_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_13 = torch.ops.aten.var_mean.correction(view_39, [0, 2], correction = 0, keepdim = True)
        getitem_26: "f32[1, 512, 1][512, 1, 1]cuda:0" = var_mean_13[0]
        getitem_27: "f32[1, 512, 1][512, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        add_25: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_13: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        sub_13: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_39, getitem_27);  view_39 = None
        mul_45: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        squeeze_26: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2]);  getitem_27 = None
        squeeze_27: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2]);  rsqrt_13 = None
        unsqueeze_13: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_40, -1);  view_40 = None
        mul_46: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_45, unsqueeze_13);  mul_45 = unsqueeze_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_41: "f32[512, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_46, [512, 128, 1, 1]);  mul_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_53: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_47, torch.bfloat16);  primals_47 = None
        convert_element_type_54: "bf16[512, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_41, torch.bfloat16);  view_41 = None
        convolution_15: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_52, convert_element_type_54, convert_element_type_53, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_1: "bf16[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convolution_15, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convert_element_type_55: "bf16[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_49, torch.bfloat16);  primals_49 = None
        convert_element_type_56: "bf16[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.prims.convert_element_type.default(primals_48, torch.bfloat16);  primals_48 = None
        convolution_16: "bf16[128, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.aten.convolution.default(mean_1, convert_element_type_56, convert_element_type_55, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_1: "bf16[128, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.aten.relu.default(convolution_16);  convolution_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convert_element_type_57: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_51, torch.bfloat16);  primals_51 = None
        convert_element_type_58: "bf16[512, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.prims.convert_element_type.default(primals_50, torch.bfloat16);  primals_50 = None
        convolution_17: "bf16[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.aten.convolution.default(relu_1, convert_element_type_58, convert_element_type_57, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_1: "bf16[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.aten.sigmoid.default(convolution_17)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_47: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(convolution_15, sigmoid_1);  sigmoid_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_48: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_47, 2.0);  mul_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_49: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_48, 0.2);  mul_48 = None
        add_26: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.add.Tensor(mul_49, convolution_11);  mul_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        convert_element_type_59: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(add_26, torch.float32)
        neg_11: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.neg.default(convert_element_type_59)
        exp_11: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.exp.default(neg_11);  neg_11 = None
        add_27: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.add.Tensor(exp_11, 1);  exp_11 = None
        div_11: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_59, add_27);  convert_element_type_59 = add_27 = None
        convert_element_type_60: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None
        mul_50: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_60, 0.9805806756909201);  convert_element_type_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_42: "f32[1, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(primals_52, [1, 128, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_51: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_53, 0.07902489841601695)
        view_43: "f32[128][1]cuda:0" = torch.ops.aten.reshape.default(mul_51, [-1]);  mul_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_14 = torch.ops.aten.var_mean.correction(view_42, [0, 2], correction = 0, keepdim = True)
        getitem_28: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_14[0]
        getitem_29: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        add_28: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_14: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_14: "f32[1, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_42, getitem_29);  view_42 = None
        mul_52: "f32[1, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        squeeze_28: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2]);  getitem_29 = None
        squeeze_29: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2]);  rsqrt_14 = None
        unsqueeze_14: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_43, -1);  view_43 = None
        mul_53: "f32[1, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_52, unsqueeze_14);  mul_52 = unsqueeze_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_44: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_53, [128, 512, 1, 1]);  mul_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_61: "bf16[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_54, torch.bfloat16);  primals_54 = None
        convert_element_type_62: "bf16[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_44, torch.bfloat16);  view_44 = None
        convolution_18: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.convolution.default(mul_50, convert_element_type_62, convert_element_type_61, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        convert_element_type_63: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_18, torch.float32)
        neg_12: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.neg.default(convert_element_type_63)
        exp_12: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.exp.default(neg_12);  neg_12 = None
        add_29: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.add.Tensor(exp_12, 1);  exp_12 = None
        div_12: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_63, add_29);  convert_element_type_63 = add_29 = None
        convert_element_type_64: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(div_12, torch.bfloat16);  div_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_16: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_55, memory_format = torch.contiguous_format)
        view_45: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_16, [1, 128, 576]);  clone_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_54: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_56, 0.07450538873672485)
        view_46: "f32[128][1]cuda:0" = torch.ops.aten.reshape.default(mul_54, [-1]);  mul_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_15 = torch.ops.aten.var_mean.correction(view_45, [0, 2], correction = 0, keepdim = True)
        getitem_30: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_15[0]
        getitem_31: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        add_30: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_15: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        sub_15: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_45, getitem_31);  view_45 = None
        mul_55: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        squeeze_30: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2]);  getitem_31 = None
        squeeze_31: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2]);  rsqrt_15 = None
        unsqueeze_15: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_46, -1);  view_46 = None
        mul_56: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_55, unsqueeze_15);  mul_55 = unsqueeze_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_47: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_56, [128, 64, 3, 3]);  mul_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_65: "bf16[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_57, torch.bfloat16);  primals_57 = None
        convert_element_type_66: "bf16[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_47, torch.bfloat16);  view_47 = None
        convolution_19: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_64, convert_element_type_66, convert_element_type_65, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  convert_element_type_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        convert_element_type_67: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_19, torch.float32)
        neg_13: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.neg.default(convert_element_type_67)
        exp_13: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.exp.default(neg_13);  neg_13 = None
        add_31: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.add.Tensor(exp_13, 1);  exp_13 = None
        div_13: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_67, add_31);  convert_element_type_67 = add_31 = None
        convert_element_type_68: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_18: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_58, memory_format = torch.contiguous_format)
        view_48: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_18, [1, 128, 576]);  clone_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_57: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_59, 0.07450538873672485)
        view_49: "f32[128][1]cuda:0" = torch.ops.aten.reshape.default(mul_57, [-1]);  mul_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_16 = torch.ops.aten.var_mean.correction(view_48, [0, 2], correction = 0, keepdim = True)
        getitem_32: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_16[0]
        getitem_33: "f32[1, 128, 1][128, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        add_32: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_16: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        sub_16: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_48, getitem_33);  view_48 = None
        mul_58: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        squeeze_32: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2]);  getitem_33 = None
        squeeze_33: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2]);  rsqrt_16 = None
        unsqueeze_16: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_49, -1);  view_49 = None
        mul_59: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_58, unsqueeze_16);  mul_58 = unsqueeze_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_50: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_59, [128, 64, 3, 3]);  mul_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_69: "bf16[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_60, torch.bfloat16);  primals_60 = None
        convert_element_type_70: "bf16[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_50, torch.bfloat16);  view_50 = None
        convolution_20: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_68, convert_element_type_70, convert_element_type_69, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  convert_element_type_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        convert_element_type_71: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_20, torch.float32)
        neg_14: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.neg.default(convert_element_type_71)
        exp_14: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.exp.default(neg_14);  neg_14 = None
        add_33: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.add.Tensor(exp_14, 1);  exp_14 = None
        div_14: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_71, add_33);  convert_element_type_71 = add_33 = None
        convert_element_type_72: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(div_14, torch.bfloat16);  div_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_51: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.reshape.default(primals_61, [1, 512, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_60: "f32[512, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_62, 0.1580497968320339)
        view_52: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(mul_60, [-1]);  mul_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_17 = torch.ops.aten.var_mean.correction(view_51, [0, 2], correction = 0, keepdim = True)
        getitem_34: "f32[1, 512, 1][512, 1, 1]cuda:0" = var_mean_17[0]
        getitem_35: "f32[1, 512, 1][512, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        add_34: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_17: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        sub_17: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_51, getitem_35);  view_51 = None
        mul_61: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        squeeze_34: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2]);  getitem_35 = None
        squeeze_35: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2]);  rsqrt_17 = None
        unsqueeze_17: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_52, -1);  view_52 = None
        mul_62: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_61, unsqueeze_17);  mul_61 = unsqueeze_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_53: "f32[512, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_62, [512, 128, 1, 1]);  mul_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_73: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_63, torch.bfloat16);  primals_63 = None
        convert_element_type_74: "bf16[512, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_53, torch.bfloat16);  view_53 = None
        convolution_21: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_72, convert_element_type_74, convert_element_type_73, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_2: "bf16[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convolution_21, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convert_element_type_75: "bf16[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_65, torch.bfloat16);  primals_65 = None
        convert_element_type_76: "bf16[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.prims.convert_element_type.default(primals_64, torch.bfloat16);  primals_64 = None
        convolution_22: "bf16[128, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.aten.convolution.default(mean_2, convert_element_type_76, convert_element_type_75, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_2: "bf16[128, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.aten.relu.default(convolution_22);  convolution_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convert_element_type_77: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_67, torch.bfloat16);  primals_67 = None
        convert_element_type_78: "bf16[512, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.prims.convert_element_type.default(primals_66, torch.bfloat16);  primals_66 = None
        convolution_23: "bf16[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.aten.convolution.default(relu_2, convert_element_type_78, convert_element_type_77, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_2: "bf16[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.aten.sigmoid.default(convolution_23)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_63: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(convolution_21, sigmoid_2);  sigmoid_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_64: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_63, 2.0);  mul_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_65: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_64, 0.2);  mul_64 = None
        add_35: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.add.Tensor(mul_65, add_26);  mul_65 = add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        convert_element_type_79: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(add_35, torch.float32)
        neg_15: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.neg.default(convert_element_type_79)
        exp_15: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.exp.default(neg_15);  neg_15 = None
        add_36: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.add.Tensor(exp_15, 1);  exp_15 = None
        div_15: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_79, add_36);  convert_element_type_79 = add_36 = None
        convert_element_type_80: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None
        mul_66: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_80, 0.9622504486493761);  convert_element_type_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:149 in forward, code: return self.conv(self.pool(x))
        avg_pool2d_1: "bf16[128, 512, 14, 14][100352, 1, 7168, 512]cuda:0" = torch.ops.aten.avg_pool2d.default(mul_66, [2, 2], [2, 2], [0, 0], True, False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_54: "f32[1, 1536, 512][786432, 512, 1]cuda:0" = torch.ops.aten.reshape.default(primals_68, [1, 1536, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_67: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_69, 0.07902489841601695)
        view_55: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(mul_67, [-1]);  mul_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_18 = torch.ops.aten.var_mean.correction(view_54, [0, 2], correction = 0, keepdim = True)
        getitem_36: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = var_mean_18[0]
        getitem_37: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        add_37: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_18: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        sub_18: "f32[1, 1536, 512][786432, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_54, getitem_37);  view_54 = None
        mul_68: "f32[1, 1536, 512][786432, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        squeeze_36: "f32[1536][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2]);  getitem_37 = None
        squeeze_37: "f32[1536][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2]);  rsqrt_18 = None
        unsqueeze_18: "f32[1536, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_55, -1);  view_55 = None
        mul_69: "f32[1, 1536, 512][786432, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, unsqueeze_18);  mul_68 = unsqueeze_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_56: "f32[1536, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_69, [1536, 512, 1, 1]);  mul_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_81: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_70, torch.bfloat16);  primals_70 = None
        convert_element_type_82: "bf16[1536, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_56, torch.bfloat16);  view_56 = None
        convolution_24: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.convolution.default(avg_pool2d_1, convert_element_type_82, convert_element_type_81, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_57: "f32[1, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.reshape.default(primals_71, [1, 384, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_70: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_72, 0.07902489841601695)
        view_58: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_70, [-1]);  mul_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_19 = torch.ops.aten.var_mean.correction(view_57, [0, 2], correction = 0, keepdim = True)
        getitem_38: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_19[0]
        getitem_39: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        add_38: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_19: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        sub_19: "f32[1, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_57, getitem_39);  view_57 = None
        mul_71: "f32[1, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        squeeze_38: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2]);  getitem_39 = None
        squeeze_39: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2]);  rsqrt_19 = None
        unsqueeze_19: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_58, -1);  view_58 = None
        mul_72: "f32[1, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_71, unsqueeze_19);  mul_71 = unsqueeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_59: "f32[384, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_72, [384, 512, 1, 1]);  mul_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_83: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_73, torch.bfloat16);  primals_73 = None
        convert_element_type_84: "bf16[384, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_59, torch.bfloat16);  view_59 = None
        convolution_25: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.convolution.default(mul_66, convert_element_type_84, convert_element_type_83, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        convert_element_type_85: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_25, torch.float32)
        neg_16: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.neg.default(convert_element_type_85)
        exp_16: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.exp.default(neg_16);  neg_16 = None
        add_39: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_16, 1);  exp_16 = None
        div_16: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_85, add_39);  convert_element_type_85 = add_39 = None
        convert_element_type_86: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(div_16, torch.bfloat16);  div_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_20: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_74, memory_format = torch.contiguous_format)
        view_60: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_20, [1, 384, 576]);  clone_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_73: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_75, 0.07450538873672485)
        view_61: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_73, [-1]);  mul_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_20 = torch.ops.aten.var_mean.correction(view_60, [0, 2], correction = 0, keepdim = True)
        getitem_40: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_20[0]
        getitem_41: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None
        add_40: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_20: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        sub_20: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_60, getitem_41);  view_60 = None
        mul_74: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
        squeeze_40: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2]);  getitem_41 = None
        squeeze_41: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2]);  rsqrt_20 = None
        unsqueeze_20: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_61, -1);  view_61 = None
        mul_75: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_74, unsqueeze_20);  mul_74 = unsqueeze_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_62: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_75, [384, 64, 3, 3]);  mul_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_87: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_76, torch.bfloat16);  primals_76 = None
        convert_element_type_88: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_62, torch.bfloat16);  view_62 = None
        convolution_26: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_86, convert_element_type_88, convert_element_type_87, [2, 2], [1, 1], [1, 1], False, [0, 0], 6);  convert_element_type_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        convert_element_type_89: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_26, torch.float32)
        neg_17: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convert_element_type_89)
        exp_17: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_17);  neg_17 = None
        add_41: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_17, 1);  exp_17 = None
        div_17: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_89, add_41);  convert_element_type_89 = add_41 = None
        convert_element_type_90: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(div_17, torch.bfloat16);  div_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_22: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_77, memory_format = torch.contiguous_format)
        view_63: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_22, [1, 384, 576]);  clone_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_76: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_78, 0.07450538873672485)
        view_64: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_76, [-1]);  mul_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_21 = torch.ops.aten.var_mean.correction(view_63, [0, 2], correction = 0, keepdim = True)
        getitem_42: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_21[0]
        getitem_43: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        add_42: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_21: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_21: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_63, getitem_43);  view_63 = None
        mul_77: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
        squeeze_42: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2]);  getitem_43 = None
        squeeze_43: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2]);  rsqrt_21 = None
        unsqueeze_21: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_64, -1);  view_64 = None
        mul_78: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_21);  mul_77 = unsqueeze_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_65: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_78, [384, 64, 3, 3]);  mul_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_91: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_79, torch.bfloat16);  primals_79 = None
        convert_element_type_92: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_65, torch.bfloat16);  view_65 = None
        convolution_27: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_90, convert_element_type_92, convert_element_type_91, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  convert_element_type_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        convert_element_type_93: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_27, torch.float32)
        neg_18: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convert_element_type_93)
        exp_18: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_18);  neg_18 = None
        add_43: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_18, 1);  exp_18 = None
        div_18: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_93, add_43);  convert_element_type_93 = add_43 = None
        convert_element_type_94: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(div_18, torch.bfloat16);  div_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_66: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(primals_80, [1, 1536, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_79: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_81, 0.09125009274634042)
        view_67: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(mul_79, [-1]);  mul_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_22 = torch.ops.aten.var_mean.correction(view_66, [0, 2], correction = 0, keepdim = True)
        getitem_44: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = var_mean_22[0]
        getitem_45: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None
        add_44: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_22: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_22: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_66, getitem_45);  view_66 = None
        mul_80: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        squeeze_44: "f32[1536][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2]);  getitem_45 = None
        squeeze_45: "f32[1536][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2]);  rsqrt_22 = None
        unsqueeze_22: "f32[1536, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_67, -1);  view_67 = None
        mul_81: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_80, unsqueeze_22);  mul_80 = unsqueeze_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_68: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_81, [1536, 384, 1, 1]);  mul_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_95: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_82, torch.bfloat16);  primals_82 = None
        convert_element_type_96: "bf16[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_68, torch.bfloat16);  view_68 = None
        convolution_28: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_94, convert_element_type_96, convert_element_type_95, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_3: "bf16[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convolution_28, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convert_element_type_97: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_84, torch.bfloat16);  primals_84 = None
        convert_element_type_98: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(primals_83, torch.bfloat16);  primals_83 = None
        convolution_29: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.convolution.default(mean_3, convert_element_type_98, convert_element_type_97, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_3: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.relu.default(convolution_29);  convolution_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convert_element_type_99: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_86, torch.bfloat16);  primals_86 = None
        convert_element_type_100: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_85, torch.bfloat16);  primals_85 = None
        convolution_30: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.convolution.default(relu_3, convert_element_type_100, convert_element_type_99, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_3: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sigmoid.default(convolution_30)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_82: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convolution_28, sigmoid_3);  sigmoid_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_83: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_82, 2.0);  mul_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_84: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_83, 0.2);  mul_83 = None
        add_45: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_84, convolution_24);  mul_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        convert_element_type_101: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(add_45, torch.float32)
        neg_19: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.neg.default(convert_element_type_101)
        exp_19: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.exp.default(neg_19);  neg_19 = None
        add_46: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(exp_19, 1);  exp_19 = None
        div_19: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_101, add_46);  convert_element_type_101 = add_46 = None
        convert_element_type_102: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16);  div_19 = None
        mul_85: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_102, 0.9805806756909201);  convert_element_type_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_69: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(primals_87, [1, 384, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_86: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_88, 0.04562504637317021)
        view_70: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_86, [-1]);  mul_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_23 = torch.ops.aten.var_mean.correction(view_69, [0, 2], correction = 0, keepdim = True)
        getitem_46: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_23[0]
        getitem_47: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        add_47: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_23: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        sub_23: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_69, getitem_47);  view_69 = None
        mul_87: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
        squeeze_46: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2]);  getitem_47 = None
        squeeze_47: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2]);  rsqrt_23 = None
        unsqueeze_23: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_70, -1);  view_70 = None
        mul_88: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_87, unsqueeze_23);  mul_87 = unsqueeze_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_71: "f32[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_88, [384, 1536, 1, 1]);  mul_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_103: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_89, torch.bfloat16);  primals_89 = None
        convert_element_type_104: "bf16[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_71, torch.bfloat16);  view_71 = None
        convolution_31: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(mul_85, convert_element_type_104, convert_element_type_103, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        convert_element_type_105: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_31, torch.float32)
        neg_20: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convert_element_type_105)
        exp_20: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_20);  neg_20 = None
        add_48: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_20, 1);  exp_20 = None
        div_20: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_105, add_48);  convert_element_type_105 = add_48 = None
        convert_element_type_106: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(div_20, torch.bfloat16);  div_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_24: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_90, memory_format = torch.contiguous_format)
        view_72: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_24, [1, 384, 576]);  clone_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_89: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_91, 0.07450538873672485)
        view_73: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_89, [-1]);  mul_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_24 = torch.ops.aten.var_mean.correction(view_72, [0, 2], correction = 0, keepdim = True)
        getitem_48: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_24[0]
        getitem_49: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None
        add_49: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_24: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        sub_24: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_72, getitem_49);  view_72 = None
        mul_90: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        squeeze_48: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2]);  getitem_49 = None
        squeeze_49: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2]);  rsqrt_24 = None
        unsqueeze_24: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_73, -1);  view_73 = None
        mul_91: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_90, unsqueeze_24);  mul_90 = unsqueeze_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_74: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_91, [384, 64, 3, 3]);  mul_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_107: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_92, torch.bfloat16);  primals_92 = None
        convert_element_type_108: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_74, torch.bfloat16);  view_74 = None
        convolution_32: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_106, convert_element_type_108, convert_element_type_107, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  convert_element_type_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        convert_element_type_109: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_32, torch.float32)
        neg_21: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convert_element_type_109)
        exp_21: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_21);  neg_21 = None
        add_50: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_21, 1);  exp_21 = None
        div_21: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_109, add_50);  convert_element_type_109 = add_50 = None
        convert_element_type_110: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16);  div_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_26: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_93, memory_format = torch.contiguous_format)
        view_75: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_26, [1, 384, 576]);  clone_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_92: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_94, 0.07450538873672485)
        view_76: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_92, [-1]);  mul_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_25 = torch.ops.aten.var_mean.correction(view_75, [0, 2], correction = 0, keepdim = True)
        getitem_50: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_25[0]
        getitem_51: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_25[1];  var_mean_25 = None
        add_51: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
        rsqrt_25: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        sub_25: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_75, getitem_51);  view_75 = None
        mul_93: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        squeeze_50: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2]);  getitem_51 = None
        squeeze_51: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2]);  rsqrt_25 = None
        unsqueeze_25: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_76, -1);  view_76 = None
        mul_94: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_93, unsqueeze_25);  mul_93 = unsqueeze_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_77: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_94, [384, 64, 3, 3]);  mul_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_111: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_95, torch.bfloat16);  primals_95 = None
        convert_element_type_112: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_77, torch.bfloat16);  view_77 = None
        convolution_33: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_110, convert_element_type_112, convert_element_type_111, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  convert_element_type_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        convert_element_type_113: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_33, torch.float32)
        neg_22: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convert_element_type_113)
        exp_22: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_22);  neg_22 = None
        add_52: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_22, 1);  exp_22 = None
        div_22: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_113, add_52);  convert_element_type_113 = add_52 = None
        convert_element_type_114: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(div_22, torch.bfloat16);  div_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_78: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(primals_96, [1, 1536, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_95: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_97, 0.09125009274634042)
        view_79: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(mul_95, [-1]);  mul_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_26 = torch.ops.aten.var_mean.correction(view_78, [0, 2], correction = 0, keepdim = True)
        getitem_52: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = var_mean_26[0]
        getitem_53: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = var_mean_26[1];  var_mean_26 = None
        add_53: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
        rsqrt_26: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_26: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_78, getitem_53);  view_78 = None
        mul_96: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
        squeeze_52: "f32[1536][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2]);  getitem_53 = None
        squeeze_53: "f32[1536][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2]);  rsqrt_26 = None
        unsqueeze_26: "f32[1536, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_79, -1);  view_79 = None
        mul_97: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_96, unsqueeze_26);  mul_96 = unsqueeze_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_80: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_97, [1536, 384, 1, 1]);  mul_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_115: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_98, torch.bfloat16);  primals_98 = None
        convert_element_type_116: "bf16[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_80, torch.bfloat16);  view_80 = None
        convolution_34: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_114, convert_element_type_116, convert_element_type_115, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_4: "bf16[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convolution_34, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convert_element_type_117: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_100, torch.bfloat16);  primals_100 = None
        convert_element_type_118: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(primals_99, torch.bfloat16);  primals_99 = None
        convolution_35: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.convolution.default(mean_4, convert_element_type_118, convert_element_type_117, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_4: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.relu.default(convolution_35);  convolution_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convert_element_type_119: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_102, torch.bfloat16);  primals_102 = None
        convert_element_type_120: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_101, torch.bfloat16);  primals_101 = None
        convolution_36: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.convolution.default(relu_4, convert_element_type_120, convert_element_type_119, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_4: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sigmoid.default(convolution_36)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_98: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convolution_34, sigmoid_4);  sigmoid_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_99: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_98, 2.0);  mul_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_100: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_99, 0.2);  mul_99 = None
        add_54: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_100, add_45);  mul_100 = add_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        convert_element_type_121: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(add_54, torch.float32)
        neg_23: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.neg.default(convert_element_type_121)
        exp_23: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.exp.default(neg_23);  neg_23 = None
        add_55: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(exp_23, 1);  exp_23 = None
        div_23: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_121, add_55);  convert_element_type_121 = add_55 = None
        convert_element_type_122: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(div_23, torch.bfloat16);  div_23 = None
        mul_101: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_122, 0.9622504486493761);  convert_element_type_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_81: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(primals_103, [1, 384, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_102: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_104, 0.04562504637317021)
        view_82: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_102, [-1]);  mul_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_27 = torch.ops.aten.var_mean.correction(view_81, [0, 2], correction = 0, keepdim = True)
        getitem_54: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_27[0]
        getitem_55: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_27[1];  var_mean_27 = None
        add_56: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
        rsqrt_27: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        sub_27: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_81, getitem_55);  view_81 = None
        mul_103: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
        squeeze_54: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2]);  getitem_55 = None
        squeeze_55: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2]);  rsqrt_27 = None
        unsqueeze_27: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_82, -1);  view_82 = None
        mul_104: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_103, unsqueeze_27);  mul_103 = unsqueeze_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_83: "f32[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_104, [384, 1536, 1, 1]);  mul_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_123: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_105, torch.bfloat16);  primals_105 = None
        convert_element_type_124: "bf16[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_83, torch.bfloat16);  view_83 = None
        convolution_37: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(mul_101, convert_element_type_124, convert_element_type_123, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        convert_element_type_125: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_37, torch.float32)
        neg_24: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convert_element_type_125)
        exp_24: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_24);  neg_24 = None
        add_57: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_24, 1);  exp_24 = None
        div_24: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_125, add_57);  convert_element_type_125 = add_57 = None
        convert_element_type_126: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(div_24, torch.bfloat16);  div_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_28: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_106, memory_format = torch.contiguous_format)
        view_84: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_28, [1, 384, 576]);  clone_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_105: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_107, 0.07450538873672485)
        view_85: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_105, [-1]);  mul_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_28 = torch.ops.aten.var_mean.correction(view_84, [0, 2], correction = 0, keepdim = True)
        getitem_56: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_28[0]
        getitem_57: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_28[1];  var_mean_28 = None
        add_58: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
        rsqrt_28: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_28: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_84, getitem_57);  view_84 = None
        mul_106: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
        squeeze_56: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2]);  getitem_57 = None
        squeeze_57: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2]);  rsqrt_28 = None
        unsqueeze_28: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_85, -1);  view_85 = None
        mul_107: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_106, unsqueeze_28);  mul_106 = unsqueeze_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_86: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_107, [384, 64, 3, 3]);  mul_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_127: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_108, torch.bfloat16);  primals_108 = None
        convert_element_type_128: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_86, torch.bfloat16);  view_86 = None
        convolution_38: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_126, convert_element_type_128, convert_element_type_127, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  convert_element_type_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        convert_element_type_129: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_38, torch.float32)
        neg_25: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convert_element_type_129)
        exp_25: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_25);  neg_25 = None
        add_59: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_25, 1);  exp_25 = None
        div_25: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_129, add_59);  convert_element_type_129 = add_59 = None
        convert_element_type_130: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(div_25, torch.bfloat16);  div_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_30: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_109, memory_format = torch.contiguous_format)
        view_87: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_30, [1, 384, 576]);  clone_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_108: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_110, 0.07450538873672485)
        view_88: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_108, [-1]);  mul_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_29 = torch.ops.aten.var_mean.correction(view_87, [0, 2], correction = 0, keepdim = True)
        getitem_58: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_29[0]
        getitem_59: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_29[1];  var_mean_29 = None
        add_60: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
        rsqrt_29: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_29: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_87, getitem_59);  view_87 = None
        mul_109: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
        squeeze_58: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2]);  getitem_59 = None
        squeeze_59: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2]);  rsqrt_29 = None
        unsqueeze_29: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_88, -1);  view_88 = None
        mul_110: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_109, unsqueeze_29);  mul_109 = unsqueeze_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_89: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_110, [384, 64, 3, 3]);  mul_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_131: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_111, torch.bfloat16);  primals_111 = None
        convert_element_type_132: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_89, torch.bfloat16);  view_89 = None
        convolution_39: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_130, convert_element_type_132, convert_element_type_131, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  convert_element_type_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        convert_element_type_133: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_39, torch.float32)
        neg_26: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convert_element_type_133)
        exp_26: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_26);  neg_26 = None
        add_61: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_26, 1);  exp_26 = None
        div_26: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_133, add_61);  convert_element_type_133 = add_61 = None
        convert_element_type_134: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(div_26, torch.bfloat16);  div_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_90: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(primals_112, [1, 1536, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_111: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_113, 0.09125009274634042)
        view_91: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(mul_111, [-1]);  mul_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_30 = torch.ops.aten.var_mean.correction(view_90, [0, 2], correction = 0, keepdim = True)
        getitem_60: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = var_mean_30[0]
        getitem_61: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = var_mean_30[1];  var_mean_30 = None
        add_62: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
        rsqrt_30: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        sub_30: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_90, getitem_61);  view_90 = None
        mul_112: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
        squeeze_60: "f32[1536][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2]);  getitem_61 = None
        squeeze_61: "f32[1536][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2]);  rsqrt_30 = None
        unsqueeze_30: "f32[1536, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_91, -1);  view_91 = None
        mul_113: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_30);  mul_112 = unsqueeze_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_92: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_113, [1536, 384, 1, 1]);  mul_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_135: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_114, torch.bfloat16);  primals_114 = None
        convert_element_type_136: "bf16[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_92, torch.bfloat16);  view_92 = None
        convolution_40: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_134, convert_element_type_136, convert_element_type_135, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_5: "bf16[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convolution_40, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convert_element_type_137: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_116, torch.bfloat16);  primals_116 = None
        convert_element_type_138: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(primals_115, torch.bfloat16);  primals_115 = None
        convolution_41: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.convolution.default(mean_5, convert_element_type_138, convert_element_type_137, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_5: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.relu.default(convolution_41);  convolution_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convert_element_type_139: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_118, torch.bfloat16);  primals_118 = None
        convert_element_type_140: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_117, torch.bfloat16);  primals_117 = None
        convolution_42: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.convolution.default(relu_5, convert_element_type_140, convert_element_type_139, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_5: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sigmoid.default(convolution_42)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_114: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convolution_40, sigmoid_5);  sigmoid_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_115: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_114, 2.0);  mul_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_116: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_115, 0.2);  mul_115 = None
        add_63: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_116, add_54);  mul_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        convert_element_type_141: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(add_63, torch.float32)
        neg_27: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.neg.default(convert_element_type_141)
        exp_27: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.exp.default(neg_27);  neg_27 = None
        add_64: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(exp_27, 1);  exp_27 = None
        div_27: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_141, add_64);  convert_element_type_141 = add_64 = None
        convert_element_type_142: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(div_27, torch.bfloat16);  div_27 = None
        mul_117: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_142, 0.9449111825230679);  convert_element_type_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_93: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(primals_119, [1, 384, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_118: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_120, 0.04562504637317021)
        view_94: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_118, [-1]);  mul_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_31 = torch.ops.aten.var_mean.correction(view_93, [0, 2], correction = 0, keepdim = True)
        getitem_62: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_31[0]
        getitem_63: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_31[1];  var_mean_31 = None
        add_65: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
        rsqrt_31: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        sub_31: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_93, getitem_63);  view_93 = None
        mul_119: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
        squeeze_62: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2]);  getitem_63 = None
        squeeze_63: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2]);  rsqrt_31 = None
        unsqueeze_31: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_94, -1);  view_94 = None
        mul_120: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_31);  mul_119 = unsqueeze_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_95: "f32[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_120, [384, 1536, 1, 1]);  mul_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_143: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_121, torch.bfloat16);  primals_121 = None
        convert_element_type_144: "bf16[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_95, torch.bfloat16);  view_95 = None
        convolution_43: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(mul_117, convert_element_type_144, convert_element_type_143, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        convert_element_type_145: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_43, torch.float32)
        neg_28: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convert_element_type_145)
        exp_28: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_28);  neg_28 = None
        add_66: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_28, 1);  exp_28 = None
        div_28: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_145, add_66);  convert_element_type_145 = add_66 = None
        convert_element_type_146: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(div_28, torch.bfloat16);  div_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_32: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_122, memory_format = torch.contiguous_format)
        view_96: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_32, [1, 384, 576]);  clone_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_121: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_123, 0.07450538873672485)
        view_97: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_121, [-1]);  mul_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_32 = torch.ops.aten.var_mean.correction(view_96, [0, 2], correction = 0, keepdim = True)
        getitem_64: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_32[0]
        getitem_65: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_32[1];  var_mean_32 = None
        add_67: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_32: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
        sub_32: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_96, getitem_65);  view_96 = None
        mul_122: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
        squeeze_64: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2]);  getitem_65 = None
        squeeze_65: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2]);  rsqrt_32 = None
        unsqueeze_32: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_97, -1);  view_97 = None
        mul_123: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_122, unsqueeze_32);  mul_122 = unsqueeze_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_98: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_123, [384, 64, 3, 3]);  mul_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_147: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_124, torch.bfloat16);  primals_124 = None
        convert_element_type_148: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_98, torch.bfloat16);  view_98 = None
        convolution_44: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_146, convert_element_type_148, convert_element_type_147, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  convert_element_type_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        convert_element_type_149: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_44, torch.float32)
        neg_29: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convert_element_type_149)
        exp_29: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_29);  neg_29 = None
        add_68: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_29, 1);  exp_29 = None
        div_29: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_149, add_68);  convert_element_type_149 = add_68 = None
        convert_element_type_150: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(div_29, torch.bfloat16);  div_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_34: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_125, memory_format = torch.contiguous_format)
        view_99: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_34, [1, 384, 576]);  clone_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_124: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_126, 0.07450538873672485)
        view_100: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_124, [-1]);  mul_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_33 = torch.ops.aten.var_mean.correction(view_99, [0, 2], correction = 0, keepdim = True)
        getitem_66: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_33[0]
        getitem_67: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_33[1];  var_mean_33 = None
        add_69: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_33: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_33: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_99, getitem_67);  view_99 = None
        mul_125: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
        squeeze_66: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2]);  getitem_67 = None
        squeeze_67: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2]);  rsqrt_33 = None
        unsqueeze_33: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_100, -1);  view_100 = None
        mul_126: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_125, unsqueeze_33);  mul_125 = unsqueeze_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_101: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_126, [384, 64, 3, 3]);  mul_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_151: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_127, torch.bfloat16);  primals_127 = None
        convert_element_type_152: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_101, torch.bfloat16);  view_101 = None
        convolution_45: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_150, convert_element_type_152, convert_element_type_151, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  convert_element_type_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        convert_element_type_153: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_45, torch.float32)
        neg_30: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convert_element_type_153)
        exp_30: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_30);  neg_30 = None
        add_70: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_30, 1);  exp_30 = None
        div_30: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_153, add_70);  convert_element_type_153 = add_70 = None
        convert_element_type_154: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(div_30, torch.bfloat16);  div_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_102: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(primals_128, [1, 1536, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_127: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_129, 0.09125009274634042)
        view_103: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(mul_127, [-1]);  mul_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_34 = torch.ops.aten.var_mean.correction(view_102, [0, 2], correction = 0, keepdim = True)
        getitem_68: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = var_mean_34[0]
        getitem_69: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = var_mean_34[1];  var_mean_34 = None
        add_71: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
        rsqrt_34: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        sub_34: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_102, getitem_69);  view_102 = None
        mul_128: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
        squeeze_68: "f32[1536][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2]);  getitem_69 = None
        squeeze_69: "f32[1536][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2]);  rsqrt_34 = None
        unsqueeze_34: "f32[1536, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_103, -1);  view_103 = None
        mul_129: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_128, unsqueeze_34);  mul_128 = unsqueeze_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_104: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_129, [1536, 384, 1, 1]);  mul_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_155: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_130, torch.bfloat16);  primals_130 = None
        convert_element_type_156: "bf16[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_104, torch.bfloat16);  view_104 = None
        convolution_46: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_154, convert_element_type_156, convert_element_type_155, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_6: "bf16[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convolution_46, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convert_element_type_157: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_132, torch.bfloat16);  primals_132 = None
        convert_element_type_158: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(primals_131, torch.bfloat16);  primals_131 = None
        convolution_47: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.convolution.default(mean_6, convert_element_type_158, convert_element_type_157, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_6: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.relu.default(convolution_47);  convolution_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convert_element_type_159: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_134, torch.bfloat16);  primals_134 = None
        convert_element_type_160: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_133, torch.bfloat16);  primals_133 = None
        convolution_48: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.convolution.default(relu_6, convert_element_type_160, convert_element_type_159, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_6: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sigmoid.default(convolution_48)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_130: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convolution_46, sigmoid_6);  sigmoid_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_131: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_130, 2.0);  mul_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_132: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_131, 0.2);  mul_131 = None
        add_72: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_132, add_63);  mul_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        convert_element_type_161: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(add_72, torch.float32)
        neg_31: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.neg.default(convert_element_type_161)
        exp_31: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.exp.default(neg_31);  neg_31 = None
        add_73: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(exp_31, 1);  exp_31 = None
        div_31: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_161, add_73);  convert_element_type_161 = add_73 = None
        convert_element_type_162: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(div_31, torch.bfloat16);  div_31 = None
        mul_133: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_162, 0.9284766908852592);  convert_element_type_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_105: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(primals_135, [1, 384, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_134: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_136, 0.04562504637317021)
        view_106: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_134, [-1]);  mul_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_35 = torch.ops.aten.var_mean.correction(view_105, [0, 2], correction = 0, keepdim = True)
        getitem_70: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_35[0]
        getitem_71: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_35[1];  var_mean_35 = None
        add_74: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
        rsqrt_35: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_35: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_105, getitem_71);  view_105 = None
        mul_135: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
        squeeze_70: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2]);  getitem_71 = None
        squeeze_71: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2]);  rsqrt_35 = None
        unsqueeze_35: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_106, -1);  view_106 = None
        mul_136: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_135, unsqueeze_35);  mul_135 = unsqueeze_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_107: "f32[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_136, [384, 1536, 1, 1]);  mul_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_163: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_137, torch.bfloat16);  primals_137 = None
        convert_element_type_164: "bf16[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_107, torch.bfloat16);  view_107 = None
        convolution_49: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(mul_133, convert_element_type_164, convert_element_type_163, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        convert_element_type_165: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_49, torch.float32)
        neg_32: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convert_element_type_165)
        exp_32: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_32);  neg_32 = None
        add_75: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_32, 1);  exp_32 = None
        div_32: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_165, add_75);  convert_element_type_165 = add_75 = None
        convert_element_type_166: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(div_32, torch.bfloat16);  div_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_36: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_138, memory_format = torch.contiguous_format)
        view_108: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_36, [1, 384, 576]);  clone_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_137: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_139, 0.07450538873672485)
        view_109: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_137, [-1]);  mul_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_36 = torch.ops.aten.var_mean.correction(view_108, [0, 2], correction = 0, keepdim = True)
        getitem_72: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_36[0]
        getitem_73: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_36[1];  var_mean_36 = None
        add_76: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
        rsqrt_36: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        sub_36: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_108, getitem_73);  view_108 = None
        mul_138: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
        squeeze_72: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2]);  getitem_73 = None
        squeeze_73: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2]);  rsqrt_36 = None
        unsqueeze_36: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_109, -1);  view_109 = None
        mul_139: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_138, unsqueeze_36);  mul_138 = unsqueeze_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_110: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_139, [384, 64, 3, 3]);  mul_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_167: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_140, torch.bfloat16);  primals_140 = None
        convert_element_type_168: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_110, torch.bfloat16);  view_110 = None
        convolution_50: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_166, convert_element_type_168, convert_element_type_167, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  convert_element_type_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        convert_element_type_169: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_50, torch.float32)
        neg_33: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convert_element_type_169)
        exp_33: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_33);  neg_33 = None
        add_77: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_33, 1);  exp_33 = None
        div_33: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_169, add_77);  convert_element_type_169 = add_77 = None
        convert_element_type_170: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(div_33, torch.bfloat16);  div_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_38: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_141, memory_format = torch.contiguous_format)
        view_111: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_38, [1, 384, 576]);  clone_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_140: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_142, 0.07450538873672485)
        view_112: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_140, [-1]);  mul_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_37 = torch.ops.aten.var_mean.correction(view_111, [0, 2], correction = 0, keepdim = True)
        getitem_74: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_37[0]
        getitem_75: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_37[1];  var_mean_37 = None
        add_78: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
        rsqrt_37: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        sub_37: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_111, getitem_75);  view_111 = None
        mul_141: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
        squeeze_74: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2]);  getitem_75 = None
        squeeze_75: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2]);  rsqrt_37 = None
        unsqueeze_37: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_112, -1);  view_112 = None
        mul_142: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_141, unsqueeze_37);  mul_141 = unsqueeze_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_113: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_142, [384, 64, 3, 3]);  mul_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_171: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_143, torch.bfloat16);  primals_143 = None
        convert_element_type_172: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_113, torch.bfloat16);  view_113 = None
        convolution_51: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_170, convert_element_type_172, convert_element_type_171, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  convert_element_type_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        convert_element_type_173: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_51, torch.float32)
        neg_34: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convert_element_type_173)
        exp_34: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_34);  neg_34 = None
        add_79: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_34, 1);  exp_34 = None
        div_34: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_173, add_79);  convert_element_type_173 = add_79 = None
        convert_element_type_174: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(div_34, torch.bfloat16);  div_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_114: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(primals_144, [1, 1536, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_143: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_145, 0.09125009274634042)
        view_115: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(mul_143, [-1]);  mul_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_38 = torch.ops.aten.var_mean.correction(view_114, [0, 2], correction = 0, keepdim = True)
        getitem_76: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = var_mean_38[0]
        getitem_77: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = var_mean_38[1];  var_mean_38 = None
        add_80: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
        rsqrt_38: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        sub_38: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_114, getitem_77);  view_114 = None
        mul_144: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
        squeeze_76: "f32[1536][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2]);  getitem_77 = None
        squeeze_77: "f32[1536][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2]);  rsqrt_38 = None
        unsqueeze_38: "f32[1536, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_115, -1);  view_115 = None
        mul_145: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_144, unsqueeze_38);  mul_144 = unsqueeze_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_116: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_145, [1536, 384, 1, 1]);  mul_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_175: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_146, torch.bfloat16);  primals_146 = None
        convert_element_type_176: "bf16[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_116, torch.bfloat16);  view_116 = None
        convolution_52: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_174, convert_element_type_176, convert_element_type_175, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_7: "bf16[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convolution_52, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convert_element_type_177: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_148, torch.bfloat16);  primals_148 = None
        convert_element_type_178: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(primals_147, torch.bfloat16);  primals_147 = None
        convolution_53: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.convolution.default(mean_7, convert_element_type_178, convert_element_type_177, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_7: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.relu.default(convolution_53);  convolution_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convert_element_type_179: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_150, torch.bfloat16);  primals_150 = None
        convert_element_type_180: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_149, torch.bfloat16);  primals_149 = None
        convolution_54: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.convolution.default(relu_7, convert_element_type_180, convert_element_type_179, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_7: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sigmoid.default(convolution_54)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_146: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convolution_52, sigmoid_7);  sigmoid_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_147: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_146, 2.0);  mul_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_148: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_147, 0.2);  mul_147 = None
        add_81: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_148, add_72);  mul_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        convert_element_type_181: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(add_81, torch.float32)
        neg_35: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.neg.default(convert_element_type_181)
        exp_35: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.exp.default(neg_35);  neg_35 = None
        add_82: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(exp_35, 1);  exp_35 = None
        div_35: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_181, add_82);  convert_element_type_181 = add_82 = None
        convert_element_type_182: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(div_35, torch.bfloat16);  div_35 = None
        mul_149: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_182, 0.9128709291752768);  convert_element_type_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_117: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(primals_151, [1, 384, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_150: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_152, 0.04562504637317021)
        view_118: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_150, [-1]);  mul_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_39 = torch.ops.aten.var_mean.correction(view_117, [0, 2], correction = 0, keepdim = True)
        getitem_78: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_39[0]
        getitem_79: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_39[1];  var_mean_39 = None
        add_83: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
        rsqrt_39: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
        sub_39: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_117, getitem_79);  view_117 = None
        mul_151: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
        squeeze_78: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2]);  getitem_79 = None
        squeeze_79: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2]);  rsqrt_39 = None
        unsqueeze_39: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_118, -1);  view_118 = None
        mul_152: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_151, unsqueeze_39);  mul_151 = unsqueeze_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_119: "f32[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_152, [384, 1536, 1, 1]);  mul_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_183: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_153, torch.bfloat16);  primals_153 = None
        convert_element_type_184: "bf16[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_119, torch.bfloat16);  view_119 = None
        convolution_55: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(mul_149, convert_element_type_184, convert_element_type_183, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        convert_element_type_185: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_55, torch.float32)
        neg_36: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convert_element_type_185)
        exp_36: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_36);  neg_36 = None
        add_84: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_36, 1);  exp_36 = None
        div_36: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_185, add_84);  convert_element_type_185 = add_84 = None
        convert_element_type_186: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(div_36, torch.bfloat16);  div_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_40: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_154, memory_format = torch.contiguous_format)
        view_120: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_40, [1, 384, 576]);  clone_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_153: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_155, 0.07450538873672485)
        view_121: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_153, [-1]);  mul_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_40 = torch.ops.aten.var_mean.correction(view_120, [0, 2], correction = 0, keepdim = True)
        getitem_80: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_40[0]
        getitem_81: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_40[1];  var_mean_40 = None
        add_85: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
        rsqrt_40: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        sub_40: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_120, getitem_81);  view_120 = None
        mul_154: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
        squeeze_80: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2]);  getitem_81 = None
        squeeze_81: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2]);  rsqrt_40 = None
        unsqueeze_40: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_121, -1);  view_121 = None
        mul_155: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_40);  mul_154 = unsqueeze_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_122: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_155, [384, 64, 3, 3]);  mul_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_187: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_156, torch.bfloat16);  primals_156 = None
        convert_element_type_188: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_122, torch.bfloat16);  view_122 = None
        convolution_56: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_186, convert_element_type_188, convert_element_type_187, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  convert_element_type_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        convert_element_type_189: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_56, torch.float32)
        neg_37: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convert_element_type_189)
        exp_37: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_37);  neg_37 = None
        add_86: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_37, 1);  exp_37 = None
        div_37: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_189, add_86);  convert_element_type_189 = add_86 = None
        convert_element_type_190: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(div_37, torch.bfloat16);  div_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_42: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_157, memory_format = torch.contiguous_format)
        view_123: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_42, [1, 384, 576]);  clone_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_156: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_158, 0.07450538873672485)
        view_124: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_156, [-1]);  mul_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_41 = torch.ops.aten.var_mean.correction(view_123, [0, 2], correction = 0, keepdim = True)
        getitem_82: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_41[0]
        getitem_83: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_41[1];  var_mean_41 = None
        add_87: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
        rsqrt_41: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
        sub_41: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_123, getitem_83);  view_123 = None
        mul_157: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
        squeeze_82: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2]);  getitem_83 = None
        squeeze_83: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2]);  rsqrt_41 = None
        unsqueeze_41: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_124, -1);  view_124 = None
        mul_158: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_157, unsqueeze_41);  mul_157 = unsqueeze_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_125: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_158, [384, 64, 3, 3]);  mul_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_191: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_159, torch.bfloat16);  primals_159 = None
        convert_element_type_192: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_125, torch.bfloat16);  view_125 = None
        convolution_57: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_190, convert_element_type_192, convert_element_type_191, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  convert_element_type_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        convert_element_type_193: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_57, torch.float32)
        neg_38: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convert_element_type_193)
        exp_38: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_38);  neg_38 = None
        add_88: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_38, 1);  exp_38 = None
        div_38: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_193, add_88);  convert_element_type_193 = add_88 = None
        convert_element_type_194: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(div_38, torch.bfloat16);  div_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_126: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(primals_160, [1, 1536, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_159: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_161, 0.09125009274634042)
        view_127: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(mul_159, [-1]);  mul_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_42 = torch.ops.aten.var_mean.correction(view_126, [0, 2], correction = 0, keepdim = True)
        getitem_84: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = var_mean_42[0]
        getitem_85: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = var_mean_42[1];  var_mean_42 = None
        add_89: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
        rsqrt_42: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
        sub_42: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_126, getitem_85);  view_126 = None
        mul_160: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
        squeeze_84: "f32[1536][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2]);  getitem_85 = None
        squeeze_85: "f32[1536][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2]);  rsqrt_42 = None
        unsqueeze_42: "f32[1536, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_127, -1);  view_127 = None
        mul_161: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_160, unsqueeze_42);  mul_160 = unsqueeze_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_128: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_161, [1536, 384, 1, 1]);  mul_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_195: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_162, torch.bfloat16);  primals_162 = None
        convert_element_type_196: "bf16[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_128, torch.bfloat16);  view_128 = None
        convolution_58: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_194, convert_element_type_196, convert_element_type_195, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_8: "bf16[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convolution_58, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convert_element_type_197: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_164, torch.bfloat16);  primals_164 = None
        convert_element_type_198: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(primals_163, torch.bfloat16);  primals_163 = None
        convolution_59: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.convolution.default(mean_8, convert_element_type_198, convert_element_type_197, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_8: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.relu.default(convolution_59);  convolution_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convert_element_type_199: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_166, torch.bfloat16);  primals_166 = None
        convert_element_type_200: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_165, torch.bfloat16);  primals_165 = None
        convolution_60: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.convolution.default(relu_8, convert_element_type_200, convert_element_type_199, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_8: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sigmoid.default(convolution_60)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_162: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convolution_58, sigmoid_8);  sigmoid_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_163: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_162, 2.0);  mul_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_164: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_163, 0.2);  mul_163 = None
        add_90: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_164, add_81);  mul_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        convert_element_type_201: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(add_90, torch.float32)
        neg_39: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.neg.default(convert_element_type_201)
        exp_39: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.exp.default(neg_39);  neg_39 = None
        add_91: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(exp_39, 1);  exp_39 = None
        div_39: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_201, add_91);  convert_element_type_201 = add_91 = None
        convert_element_type_202: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(div_39, torch.bfloat16);  div_39 = None
        mul_165: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_202, 0.8980265101338745);  convert_element_type_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:149 in forward, code: return self.conv(self.pool(x))
        avg_pool2d_2: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.avg_pool2d.default(mul_165, [2, 2], [2, 2], [0, 0], True, False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_129: "f32[1, 1536, 1536][2359296, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(primals_167, [1, 1536, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_166: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_168, 0.04562504637317021)
        view_130: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(mul_166, [-1]);  mul_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_43 = torch.ops.aten.var_mean.correction(view_129, [0, 2], correction = 0, keepdim = True)
        getitem_86: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = var_mean_43[0]
        getitem_87: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = var_mean_43[1];  var_mean_43 = None
        add_92: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
        rsqrt_43: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        sub_43: "f32[1, 1536, 1536][2359296, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_129, getitem_87);  view_129 = None
        mul_167: "f32[1, 1536, 1536][2359296, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
        squeeze_86: "f32[1536][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2]);  getitem_87 = None
        squeeze_87: "f32[1536][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2]);  rsqrt_43 = None
        unsqueeze_43: "f32[1536, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_130, -1);  view_130 = None
        mul_168: "f32[1, 1536, 1536][2359296, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_167, unsqueeze_43);  mul_167 = unsqueeze_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_131: "f32[1536, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_168, [1536, 1536, 1, 1]);  mul_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_203: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_169, torch.bfloat16);  primals_169 = None
        convert_element_type_204: "bf16[1536, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_131, torch.bfloat16);  view_131 = None
        convolution_61: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.convolution.default(avg_pool2d_2, convert_element_type_204, convert_element_type_203, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_132: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(primals_170, [1, 384, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_169: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_171, 0.04562504637317021)
        view_133: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_169, [-1]);  mul_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_44 = torch.ops.aten.var_mean.correction(view_132, [0, 2], correction = 0, keepdim = True)
        getitem_88: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_44[0]
        getitem_89: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_44[1];  var_mean_44 = None
        add_93: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_44: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        sub_44: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_132, getitem_89);  view_132 = None
        mul_170: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
        squeeze_88: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2]);  getitem_89 = None
        squeeze_89: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2]);  rsqrt_44 = None
        unsqueeze_44: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_133, -1);  view_133 = None
        mul_171: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_170, unsqueeze_44);  mul_170 = unsqueeze_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_134: "f32[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_171, [384, 1536, 1, 1]);  mul_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_205: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_172, torch.bfloat16);  primals_172 = None
        convert_element_type_206: "bf16[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_134, torch.bfloat16);  view_134 = None
        convolution_62: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.convolution.default(mul_165, convert_element_type_206, convert_element_type_205, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        convert_element_type_207: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_62, torch.float32)
        neg_40: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convert_element_type_207)
        exp_40: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_40);  neg_40 = None
        add_94: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_40, 1);  exp_40 = None
        div_40: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_207, add_94);  convert_element_type_207 = add_94 = None
        convert_element_type_208: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(div_40, torch.bfloat16);  div_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_44: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_173, memory_format = torch.contiguous_format)
        view_135: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_44, [1, 384, 576]);  clone_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_172: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_174, 0.07450538873672485)
        view_136: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_172, [-1]);  mul_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_45 = torch.ops.aten.var_mean.correction(view_135, [0, 2], correction = 0, keepdim = True)
        getitem_90: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_45[0]
        getitem_91: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_45[1];  var_mean_45 = None
        add_95: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
        rsqrt_45: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
        sub_45: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_135, getitem_91);  view_135 = None
        mul_173: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
        squeeze_90: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2]);  getitem_91 = None
        squeeze_91: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2]);  rsqrt_45 = None
        unsqueeze_45: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_136, -1);  view_136 = None
        mul_174: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_173, unsqueeze_45);  mul_173 = unsqueeze_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_137: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_174, [384, 64, 3, 3]);  mul_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_209: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_175, torch.bfloat16);  primals_175 = None
        convert_element_type_210: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_137, torch.bfloat16);  view_137 = None
        convolution_63: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_208, convert_element_type_210, convert_element_type_209, [2, 2], [1, 1], [1, 1], False, [0, 0], 6);  convert_element_type_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        convert_element_type_211: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_63, torch.float32)
        neg_41: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.neg.default(convert_element_type_211)
        exp_41: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.exp.default(neg_41);  neg_41 = None
        add_96: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_41, 1);  exp_41 = None
        div_41: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_211, add_96);  convert_element_type_211 = add_96 = None
        convert_element_type_212: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(div_41, torch.bfloat16);  div_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_46: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_176, memory_format = torch.contiguous_format)
        view_138: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_46, [1, 384, 576]);  clone_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_175: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_177, 0.07450538873672485)
        view_139: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_175, [-1]);  mul_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_46 = torch.ops.aten.var_mean.correction(view_138, [0, 2], correction = 0, keepdim = True)
        getitem_92: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_46[0]
        getitem_93: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_46[1];  var_mean_46 = None
        add_97: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
        rsqrt_46: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
        sub_46: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_138, getitem_93);  view_138 = None
        mul_176: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
        squeeze_92: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2]);  getitem_93 = None
        squeeze_93: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2]);  rsqrt_46 = None
        unsqueeze_46: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_139, -1);  view_139 = None
        mul_177: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_176, unsqueeze_46);  mul_176 = unsqueeze_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_140: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_177, [384, 64, 3, 3]);  mul_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_213: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_178, torch.bfloat16);  primals_178 = None
        convert_element_type_214: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_140, torch.bfloat16);  view_140 = None
        convolution_64: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_212, convert_element_type_214, convert_element_type_213, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  convert_element_type_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        convert_element_type_215: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_64, torch.float32)
        neg_42: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.neg.default(convert_element_type_215)
        exp_42: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.exp.default(neg_42);  neg_42 = None
        add_98: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_42, 1);  exp_42 = None
        div_42: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_215, add_98);  convert_element_type_215 = add_98 = None
        convert_element_type_216: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(div_42, torch.bfloat16);  div_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_141: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(primals_179, [1, 1536, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_178: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_180, 0.09125009274634042)
        view_142: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(mul_178, [-1]);  mul_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_47 = torch.ops.aten.var_mean.correction(view_141, [0, 2], correction = 0, keepdim = True)
        getitem_94: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = var_mean_47[0]
        getitem_95: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = var_mean_47[1];  var_mean_47 = None
        add_99: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
        rsqrt_47: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
        sub_47: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_141, getitem_95);  view_141 = None
        mul_179: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
        squeeze_94: "f32[1536][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2]);  getitem_95 = None
        squeeze_95: "f32[1536][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2]);  rsqrt_47 = None
        unsqueeze_47: "f32[1536, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_142, -1);  view_142 = None
        mul_180: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_179, unsqueeze_47);  mul_179 = unsqueeze_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_143: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_180, [1536, 384, 1, 1]);  mul_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_217: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_181, torch.bfloat16);  primals_181 = None
        convert_element_type_218: "bf16[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_143, torch.bfloat16);  view_143 = None
        convolution_65: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_216, convert_element_type_218, convert_element_type_217, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_9: "bf16[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convolution_65, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convert_element_type_219: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_183, torch.bfloat16);  primals_183 = None
        convert_element_type_220: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(primals_182, torch.bfloat16);  primals_182 = None
        convolution_66: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.convolution.default(mean_9, convert_element_type_220, convert_element_type_219, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_9: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.relu.default(convolution_66);  convolution_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convert_element_type_221: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_185, torch.bfloat16);  primals_185 = None
        convert_element_type_222: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_184, torch.bfloat16);  primals_184 = None
        convolution_67: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.convolution.default(relu_9, convert_element_type_222, convert_element_type_221, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_9: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sigmoid.default(convolution_67)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_181: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convolution_65, sigmoid_9);  sigmoid_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_182: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_181, 2.0);  mul_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_183: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_182, 0.2);  mul_182 = None
        add_100: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_183, convolution_61);  mul_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        convert_element_type_223: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(add_100, torch.float32)
        neg_43: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.neg.default(convert_element_type_223)
        exp_43: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.exp.default(neg_43);  neg_43 = None
        add_101: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.add.Tensor(exp_43, 1);  exp_43 = None
        div_43: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_223, add_101);  convert_element_type_223 = add_101 = None
        convert_element_type_224: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(div_43, torch.bfloat16);  div_43 = None
        mul_184: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_224, 0.9805806756909201);  convert_element_type_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_144: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(primals_186, [1, 384, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_185: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_187, 0.04562504637317021)
        view_145: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_185, [-1]);  mul_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_48 = torch.ops.aten.var_mean.correction(view_144, [0, 2], correction = 0, keepdim = True)
        getitem_96: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_48[0]
        getitem_97: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_48[1];  var_mean_48 = None
        add_102: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
        rsqrt_48: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        sub_48: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_144, getitem_97);  view_144 = None
        mul_186: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
        squeeze_96: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2]);  getitem_97 = None
        squeeze_97: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2]);  rsqrt_48 = None
        unsqueeze_48: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_145, -1);  view_145 = None
        mul_187: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_186, unsqueeze_48);  mul_186 = unsqueeze_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_146: "f32[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_187, [384, 1536, 1, 1]);  mul_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_225: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_188, torch.bfloat16);  primals_188 = None
        convert_element_type_226: "bf16[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_146, torch.bfloat16);  view_146 = None
        convolution_68: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.convolution.default(mul_184, convert_element_type_226, convert_element_type_225, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        convert_element_type_227: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_68, torch.float32)
        neg_44: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.neg.default(convert_element_type_227)
        exp_44: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.exp.default(neg_44);  neg_44 = None
        add_103: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_44, 1);  exp_44 = None
        div_44: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_227, add_103);  convert_element_type_227 = add_103 = None
        convert_element_type_228: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(div_44, torch.bfloat16);  div_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_48: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_189, memory_format = torch.contiguous_format)
        view_147: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_48, [1, 384, 576]);  clone_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_188: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_190, 0.07450538873672485)
        view_148: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_188, [-1]);  mul_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_49 = torch.ops.aten.var_mean.correction(view_147, [0, 2], correction = 0, keepdim = True)
        getitem_98: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_49[0]
        getitem_99: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_49[1];  var_mean_49 = None
        add_104: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_98, 1e-05);  getitem_98 = None
        rsqrt_49: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        sub_49: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_147, getitem_99);  view_147 = None
        mul_189: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
        squeeze_98: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2]);  getitem_99 = None
        squeeze_99: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2]);  rsqrt_49 = None
        unsqueeze_49: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_148, -1);  view_148 = None
        mul_190: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_189, unsqueeze_49);  mul_189 = unsqueeze_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_149: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_190, [384, 64, 3, 3]);  mul_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_229: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_191, torch.bfloat16);  primals_191 = None
        convert_element_type_230: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_149, torch.bfloat16);  view_149 = None
        convolution_69: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_228, convert_element_type_230, convert_element_type_229, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  convert_element_type_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        convert_element_type_231: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_69, torch.float32)
        neg_45: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.neg.default(convert_element_type_231)
        exp_45: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.exp.default(neg_45);  neg_45 = None
        add_105: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_45, 1);  exp_45 = None
        div_45: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_231, add_105);  convert_element_type_231 = add_105 = None
        convert_element_type_232: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(div_45, torch.bfloat16);  div_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_50: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_192, memory_format = torch.contiguous_format)
        view_150: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_50, [1, 384, 576]);  clone_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_191: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_193, 0.07450538873672485)
        view_151: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_191, [-1]);  mul_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_50 = torch.ops.aten.var_mean.correction(view_150, [0, 2], correction = 0, keepdim = True)
        getitem_100: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_50[0]
        getitem_101: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_50[1];  var_mean_50 = None
        add_106: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_100, 1e-05);  getitem_100 = None
        rsqrt_50: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        sub_50: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_150, getitem_101);  view_150 = None
        mul_192: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
        squeeze_100: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2]);  getitem_101 = None
        squeeze_101: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2]);  rsqrt_50 = None
        unsqueeze_50: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_151, -1);  view_151 = None
        mul_193: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_192, unsqueeze_50);  mul_192 = unsqueeze_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_152: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_193, [384, 64, 3, 3]);  mul_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_233: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_194, torch.bfloat16);  primals_194 = None
        convert_element_type_234: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_152, torch.bfloat16);  view_152 = None
        convolution_70: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_232, convert_element_type_234, convert_element_type_233, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  convert_element_type_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        convert_element_type_235: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_70, torch.float32)
        neg_46: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.neg.default(convert_element_type_235)
        exp_46: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.exp.default(neg_46);  neg_46 = None
        add_107: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_46, 1);  exp_46 = None
        div_46: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_235, add_107);  convert_element_type_235 = add_107 = None
        convert_element_type_236: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(div_46, torch.bfloat16);  div_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_153: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(primals_195, [1, 1536, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_194: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_196, 0.09125009274634042)
        view_154: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(mul_194, [-1]);  mul_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_51 = torch.ops.aten.var_mean.correction(view_153, [0, 2], correction = 0, keepdim = True)
        getitem_102: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = var_mean_51[0]
        getitem_103: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = var_mean_51[1];  var_mean_51 = None
        add_108: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_102, 1e-05);  getitem_102 = None
        rsqrt_51: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
        sub_51: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_153, getitem_103);  view_153 = None
        mul_195: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
        squeeze_102: "f32[1536][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2]);  getitem_103 = None
        squeeze_103: "f32[1536][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2]);  rsqrt_51 = None
        unsqueeze_51: "f32[1536, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_154, -1);  view_154 = None
        mul_196: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_195, unsqueeze_51);  mul_195 = unsqueeze_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_155: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_196, [1536, 384, 1, 1]);  mul_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_237: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_197, torch.bfloat16);  primals_197 = None
        convert_element_type_238: "bf16[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_155, torch.bfloat16);  view_155 = None
        convolution_71: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_236, convert_element_type_238, convert_element_type_237, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_10: "bf16[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convolution_71, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convert_element_type_239: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_199, torch.bfloat16);  primals_199 = None
        convert_element_type_240: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(primals_198, torch.bfloat16);  primals_198 = None
        convolution_72: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.convolution.default(mean_10, convert_element_type_240, convert_element_type_239, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_10: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.relu.default(convolution_72);  convolution_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convert_element_type_241: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_201, torch.bfloat16);  primals_201 = None
        convert_element_type_242: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_200, torch.bfloat16);  primals_200 = None
        convolution_73: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.convolution.default(relu_10, convert_element_type_242, convert_element_type_241, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_10: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sigmoid.default(convolution_73)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_197: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convolution_71, sigmoid_10);  sigmoid_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_198: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_197, 2.0);  mul_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_199: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_198, 0.2);  mul_198 = None
        add_109: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_199, add_100);  mul_199 = add_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        convert_element_type_243: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(add_109, torch.float32)
        neg_47: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.neg.default(convert_element_type_243)
        exp_47: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.exp.default(neg_47);  neg_47 = None
        add_110: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.add.Tensor(exp_47, 1);  exp_47 = None
        div_47: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_243, add_110);  convert_element_type_243 = add_110 = None
        convert_element_type_244: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(div_47, torch.bfloat16);  div_47 = None
        mul_200: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_244, 0.9622504486493761);  convert_element_type_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_156: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(primals_202, [1, 384, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_201: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_203, 0.04562504637317021)
        view_157: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_201, [-1]);  mul_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_52 = torch.ops.aten.var_mean.correction(view_156, [0, 2], correction = 0, keepdim = True)
        getitem_104: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_52[0]
        getitem_105: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_52[1];  var_mean_52 = None
        add_111: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
        rsqrt_52: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
        sub_52: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_156, getitem_105);  view_156 = None
        mul_202: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
        squeeze_104: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_105, [0, 2]);  getitem_105 = None
        squeeze_105: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2]);  rsqrt_52 = None
        unsqueeze_52: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_157, -1);  view_157 = None
        mul_203: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_202, unsqueeze_52);  mul_202 = unsqueeze_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_158: "f32[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_203, [384, 1536, 1, 1]);  mul_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_245: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_204, torch.bfloat16);  primals_204 = None
        convert_element_type_246: "bf16[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_158, torch.bfloat16);  view_158 = None
        convolution_74: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.convolution.default(mul_200, convert_element_type_246, convert_element_type_245, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        convert_element_type_247: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_74, torch.float32)
        neg_48: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.neg.default(convert_element_type_247)
        exp_48: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.exp.default(neg_48);  neg_48 = None
        add_112: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_48, 1);  exp_48 = None
        div_48: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_247, add_112);  convert_element_type_247 = add_112 = None
        convert_element_type_248: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(div_48, torch.bfloat16);  div_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_52: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_205, memory_format = torch.contiguous_format)
        view_159: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_52, [1, 384, 576]);  clone_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_204: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_206, 0.07450538873672485)
        view_160: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_204, [-1]);  mul_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_53 = torch.ops.aten.var_mean.correction(view_159, [0, 2], correction = 0, keepdim = True)
        getitem_106: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_53[0]
        getitem_107: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_53[1];  var_mean_53 = None
        add_113: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_106, 1e-05);  getitem_106 = None
        rsqrt_53: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
        sub_53: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_159, getitem_107);  view_159 = None
        mul_205: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = None
        squeeze_106: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2]);  getitem_107 = None
        squeeze_107: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2]);  rsqrt_53 = None
        unsqueeze_53: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_160, -1);  view_160 = None
        mul_206: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_205, unsqueeze_53);  mul_205 = unsqueeze_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_161: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_206, [384, 64, 3, 3]);  mul_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_249: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_207, torch.bfloat16);  primals_207 = None
        convert_element_type_250: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_161, torch.bfloat16);  view_161 = None
        convolution_75: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_248, convert_element_type_250, convert_element_type_249, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  convert_element_type_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        convert_element_type_251: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_75, torch.float32)
        neg_49: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.neg.default(convert_element_type_251)
        exp_49: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.exp.default(neg_49);  neg_49 = None
        add_114: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_49, 1);  exp_49 = None
        div_49: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_251, add_114);  convert_element_type_251 = add_114 = None
        convert_element_type_252: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(div_49, torch.bfloat16);  div_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_54: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_208, memory_format = torch.contiguous_format)
        view_162: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_54, [1, 384, 576]);  clone_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_207: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_209, 0.07450538873672485)
        view_163: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_207, [-1]);  mul_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_54 = torch.ops.aten.var_mean.correction(view_162, [0, 2], correction = 0, keepdim = True)
        getitem_108: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_54[0]
        getitem_109: "f32[1, 384, 1][384, 1, 1]cuda:0" = var_mean_54[1];  var_mean_54 = None
        add_115: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
        rsqrt_54: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
        sub_54: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_162, getitem_109);  view_162 = None
        mul_208: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = None
        squeeze_108: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_109, [0, 2]);  getitem_109 = None
        squeeze_109: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2]);  rsqrt_54 = None
        unsqueeze_54: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_163, -1);  view_163 = None
        mul_209: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_208, unsqueeze_54);  mul_208 = unsqueeze_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_164: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_209, [384, 64, 3, 3]);  mul_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_253: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_210, torch.bfloat16);  primals_210 = None
        convert_element_type_254: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_164, torch.bfloat16);  view_164 = None
        convolution_76: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_252, convert_element_type_254, convert_element_type_253, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  convert_element_type_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        convert_element_type_255: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_76, torch.float32)
        neg_50: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.neg.default(convert_element_type_255)
        exp_50: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.exp.default(neg_50);  neg_50 = None
        add_116: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_50, 1);  exp_50 = None
        div_50: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_255, add_116);  convert_element_type_255 = add_116 = None
        convert_element_type_256: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(div_50, torch.bfloat16);  div_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_165: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(primals_211, [1, 1536, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_210: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_212, 0.09125009274634042)
        view_166: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(mul_210, [-1]);  mul_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_55 = torch.ops.aten.var_mean.correction(view_165, [0, 2], correction = 0, keepdim = True)
        getitem_110: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = var_mean_55[0]
        getitem_111: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = var_mean_55[1];  var_mean_55 = None
        add_117: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
        rsqrt_55: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
        sub_55: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_165, getitem_111);  view_165 = None
        mul_211: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = None
        squeeze_110: "f32[1536][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_111, [0, 2]);  getitem_111 = None
        squeeze_111: "f32[1536][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2]);  rsqrt_55 = None
        unsqueeze_55: "f32[1536, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_166, -1);  view_166 = None
        mul_212: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_211, unsqueeze_55);  mul_211 = unsqueeze_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_167: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_212, [1536, 384, 1, 1]);  mul_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_257: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_213, torch.bfloat16);  primals_213 = None
        convert_element_type_258: "bf16[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_167, torch.bfloat16);  view_167 = None
        convolution_77: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_256, convert_element_type_258, convert_element_type_257, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_11: "bf16[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convolution_77, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convert_element_type_259: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_215, torch.bfloat16);  primals_215 = None
        convert_element_type_260: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(primals_214, torch.bfloat16);  primals_214 = None
        convolution_78: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.convolution.default(mean_11, convert_element_type_260, convert_element_type_259, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_11: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.relu.default(convolution_78);  convolution_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convert_element_type_261: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_217, torch.bfloat16);  primals_217 = None
        convert_element_type_262: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_216, torch.bfloat16);  primals_216 = None
        convolution_79: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.convolution.default(relu_11, convert_element_type_262, convert_element_type_261, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_11: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sigmoid.default(convolution_79)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_213: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convolution_77, sigmoid_11);  sigmoid_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_214: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_213, 2.0);  mul_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_215: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_214, 0.2);  mul_214 = None
        add_118: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_215, add_109);  mul_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_168: "f32[1, 2304, 1536][3538944, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(primals_218, [1, 2304, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_216: "f32[2304, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_219, 0.04562504637317021)
        view_169: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(mul_216, [-1]);  mul_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_56 = torch.ops.aten.var_mean.correction(view_168, [0, 2], correction = 0, keepdim = True)
        getitem_112: "f32[1, 2304, 1][2304, 1, 1]cuda:0" = var_mean_56[0]
        getitem_113: "f32[1, 2304, 1][2304, 1, 1]cuda:0" = var_mean_56[1];  var_mean_56 = None
        add_119: "f32[1, 2304, 1][2304, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
        rsqrt_56: "f32[1, 2304, 1][2304, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
        sub_56: "f32[1, 2304, 1536][3538944, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_168, getitem_113);  view_168 = None
        mul_217: "f32[1, 2304, 1536][3538944, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = None
        squeeze_112: "f32[2304][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2]);  getitem_113 = None
        squeeze_113: "f32[2304][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_56, [0, 2]);  rsqrt_56 = None
        unsqueeze_56: "f32[2304, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_169, -1);  view_169 = None
        mul_218: "f32[1, 2304, 1536][3538944, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_56);  mul_217 = unsqueeze_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_170: "f32[2304, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_218, [2304, 1536, 1, 1]);  mul_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type_263: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_220, torch.bfloat16);  primals_220 = None
        convert_element_type_264: "bf16[2304, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_170, torch.bfloat16);  view_170 = None
        convolution_80: "bf16[128, 2304, 7, 7][112896, 1, 16128, 2304]cuda:0" = torch.ops.aten.convolution.default(add_118, convert_element_type_264, convert_element_type_263, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:569 in forward_features, code: x = self.final_act(x)
        convert_element_type_265: "f32[128, 2304, 7, 7][112896, 1, 16128, 2304]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_80, torch.float32)
        neg_51: "f32[128, 2304, 7, 7][112896, 1, 16128, 2304]cuda:0" = torch.ops.aten.neg.default(convert_element_type_265)
        exp_51: "f32[128, 2304, 7, 7][112896, 1, 16128, 2304]cuda:0" = torch.ops.aten.exp.default(neg_51);  neg_51 = None
        add_120: "f32[128, 2304, 7, 7][112896, 1, 16128, 2304]cuda:0" = torch.ops.aten.add.Tensor(exp_51, 1);  exp_51 = None
        div_51: "f32[128, 2304, 7, 7][112896, 1, 16128, 2304]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_265, add_120);  convert_element_type_265 = add_120 = None
        convert_element_type_266: "bf16[128, 2304, 7, 7][112896, 1, 16128, 2304]cuda:0" = torch.ops.prims.convert_element_type.default(div_51, torch.bfloat16);  div_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_12: "bf16[128, 2304, 1, 1][2304, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_266, [-1, -2], True);  convert_element_type_266 = None
        as_strided: "bf16[128, 2304, 1, 1][2304, 1, 2304, 2304]cuda:0" = torch.ops.aten.as_strided.default(mean_12, [128, 2304, 1, 1], [2304, 1, 2304, 2304]);  mean_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_171: "bf16[128, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(as_strided, [128, 2304]);  as_strided = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:141 in forward, code: x = self.fc(x)
        convert_element_type_267: "bf16[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_222, torch.bfloat16);  primals_222 = None
        convert_element_type_268: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_221, torch.bfloat16);  primals_221 = None
        permute: "bf16[2304, 1000][1, 2304]cuda:0" = torch.ops.aten.permute.default(convert_element_type_268, [1, 0]);  convert_element_type_268 = None
        addmm: "bf16[128, 1000][1000, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_267, view_171, permute);  convert_element_type_267 = None
        permute_1: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_57: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_112, 0);  squeeze_112 = None
        unsqueeze_58: "f32[1, 2304, 1][2304, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_57, 2);  unsqueeze_57 = None
        unsqueeze_65: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_110, 0);  squeeze_110 = None
        unsqueeze_66: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_65, 2);  unsqueeze_65 = None
        unsqueeze_73: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
        unsqueeze_74: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_73, 2);  unsqueeze_73 = None
        unsqueeze_81: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_106, 0);  squeeze_106 = None
        unsqueeze_82: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_81, 2);  unsqueeze_81 = None
        unsqueeze_89: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_104, 0);  squeeze_104 = None
        unsqueeze_90: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_89, 2);  unsqueeze_89 = None
        unsqueeze_97: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
        unsqueeze_98: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_97, 2);  unsqueeze_97 = None
        unsqueeze_105: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_100, 0);  squeeze_100 = None
        unsqueeze_106: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_105, 2);  unsqueeze_105 = None
        unsqueeze_113: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_98, 0);  squeeze_98 = None
        unsqueeze_114: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_113, 2);  unsqueeze_113 = None
        unsqueeze_121: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
        unsqueeze_122: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_121, 2);  unsqueeze_121 = None
        unsqueeze_129: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_94, 0);  squeeze_94 = None
        unsqueeze_130: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_129, 2);  unsqueeze_129 = None
        unsqueeze_137: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_92, 0);  squeeze_92 = None
        unsqueeze_138: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_137, 2);  unsqueeze_137 = None
        unsqueeze_145: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
        unsqueeze_146: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_145, 2);  unsqueeze_145 = None
        unsqueeze_153: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_88, 0);  squeeze_88 = None
        unsqueeze_154: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_153, 2);  unsqueeze_153 = None
        unsqueeze_161: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_86, 0);  squeeze_86 = None
        unsqueeze_162: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_161, 2);  unsqueeze_161 = None
        unsqueeze_169: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
        unsqueeze_170: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_169, 2);  unsqueeze_169 = None
        unsqueeze_177: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_82, 0);  squeeze_82 = None
        unsqueeze_178: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_177, 2);  unsqueeze_177 = None
        unsqueeze_185: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_80, 0);  squeeze_80 = None
        unsqueeze_186: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 2);  unsqueeze_185 = None
        unsqueeze_193: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
        unsqueeze_194: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_193, 2);  unsqueeze_193 = None
        unsqueeze_201: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_76, 0);  squeeze_76 = None
        unsqueeze_202: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_201, 2);  unsqueeze_201 = None
        unsqueeze_209: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_74, 0);  squeeze_74 = None
        unsqueeze_210: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 2);  unsqueeze_209 = None
        unsqueeze_217: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
        unsqueeze_218: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 2);  unsqueeze_217 = None
        unsqueeze_225: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_70, 0);  squeeze_70 = None
        unsqueeze_226: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_225, 2);  unsqueeze_225 = None
        unsqueeze_233: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_68, 0);  squeeze_68 = None
        unsqueeze_234: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 2);  unsqueeze_233 = None
        unsqueeze_241: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
        unsqueeze_242: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 2);  unsqueeze_241 = None
        unsqueeze_249: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_64, 0);  squeeze_64 = None
        unsqueeze_250: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_249, 2);  unsqueeze_249 = None
        unsqueeze_257: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_62, 0);  squeeze_62 = None
        unsqueeze_258: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 2);  unsqueeze_257 = None
        unsqueeze_265: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
        unsqueeze_266: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 2);  unsqueeze_265 = None
        unsqueeze_273: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_58, 0);  squeeze_58 = None
        unsqueeze_274: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 2);  unsqueeze_273 = None
        unsqueeze_281: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_56, 0);  squeeze_56 = None
        unsqueeze_282: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 2);  unsqueeze_281 = None
        unsqueeze_289: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
        unsqueeze_290: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 2);  unsqueeze_289 = None
        unsqueeze_297: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_52, 0);  squeeze_52 = None
        unsqueeze_298: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 2);  unsqueeze_297 = None
        unsqueeze_305: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_50, 0);  squeeze_50 = None
        unsqueeze_306: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 2);  unsqueeze_305 = None
        unsqueeze_313: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
        unsqueeze_314: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 2);  unsqueeze_313 = None
        unsqueeze_321: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_46, 0);  squeeze_46 = None
        unsqueeze_322: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 2);  unsqueeze_321 = None
        unsqueeze_329: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_44, 0);  squeeze_44 = None
        unsqueeze_330: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 2);  unsqueeze_329 = None
        unsqueeze_337: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
        unsqueeze_338: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 2);  unsqueeze_337 = None
        unsqueeze_345: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_40, 0);  squeeze_40 = None
        unsqueeze_346: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 2);  unsqueeze_345 = None
        unsqueeze_353: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_38, 0);  squeeze_38 = None
        unsqueeze_354: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 2);  unsqueeze_353 = None
        unsqueeze_361: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
        unsqueeze_362: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
        unsqueeze_369: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_34, 0);  squeeze_34 = None
        unsqueeze_370: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 2);  unsqueeze_369 = None
        unsqueeze_377: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_32, 0);  squeeze_32 = None
        unsqueeze_378: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 2);  unsqueeze_377 = None
        unsqueeze_385: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
        unsqueeze_386: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
        unsqueeze_393: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_28, 0);  squeeze_28 = None
        unsqueeze_394: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 2);  unsqueeze_393 = None
        unsqueeze_401: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_26, 0);  squeeze_26 = None
        unsqueeze_402: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 2);  unsqueeze_401 = None
        unsqueeze_409: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
        unsqueeze_410: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 2);  unsqueeze_409 = None
        unsqueeze_417: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_22, 0);  squeeze_22 = None
        unsqueeze_418: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 2);  unsqueeze_417 = None
        unsqueeze_425: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_20, 0);  squeeze_20 = None
        unsqueeze_426: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 2);  unsqueeze_425 = None
        unsqueeze_433: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
        unsqueeze_434: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 2);  unsqueeze_433 = None
        unsqueeze_441: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_16, 0);  squeeze_16 = None
        unsqueeze_442: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 2);  unsqueeze_441 = None
        unsqueeze_449: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_14, 0);  squeeze_14 = None
        unsqueeze_450: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 2);  unsqueeze_449 = None
        unsqueeze_457: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
        unsqueeze_458: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 2);  unsqueeze_457 = None
        unsqueeze_465: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_10, 0);  squeeze_10 = None
        unsqueeze_466: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 2);  unsqueeze_465 = None
        unsqueeze_473: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_8, 0);  squeeze_8 = None
        unsqueeze_474: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 2);  unsqueeze_473 = None
        unsqueeze_481: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
        unsqueeze_482: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 2);  unsqueeze_481 = None
        unsqueeze_489: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_4, 0);  squeeze_4 = None
        unsqueeze_490: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 2);  unsqueeze_489 = None
        unsqueeze_497: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_2, 0);  squeeze_2 = None
        unsqueeze_498: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 2);  unsqueeze_497 = None
        unsqueeze_505: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_506: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 2);  unsqueeze_505 = None
        return (addmm, primals_1, primals_2, primals_5, primals_6, primals_8, primals_9, primals_11, primals_12, primals_14, primals_15, primals_17, primals_18, primals_20, primals_21, primals_23, primals_24, primals_26, primals_27, primals_33, primals_34, primals_36, primals_37, primals_39, primals_40, primals_42, primals_43, primals_45, primals_46, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_68, primals_69, primals_71, primals_72, primals_74, primals_75, primals_77, primals_78, primals_80, primals_81, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_119, primals_120, primals_122, primals_123, primals_125, primals_126, primals_128, primals_129, primals_135, primals_136, primals_138, primals_139, primals_141, primals_142, primals_144, primals_145, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_161, primals_167, primals_168, primals_170, primals_171, primals_173, primals_174, primals_176, primals_177, primals_179, primals_180, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_195, primals_196, primals_202, primals_203, primals_205, primals_206, primals_208, primals_209, primals_211, primals_212, primals_218, primals_219, squeeze_1, convert_element_type_1, convert_element_type_2, convolution, convert_element_type_4, squeeze_3, convert_element_type_6, convolution_1, convert_element_type_8, squeeze_5, convert_element_type_10, convolution_2, convert_element_type_12, squeeze_7, convert_element_type_14, convolution_3, mul_12, squeeze_9, convert_element_type_18, convolution_4, squeeze_11, convert_element_type_20, convolution_5, convert_element_type_22, squeeze_13, convert_element_type_24, convolution_6, convert_element_type_26, squeeze_15, convert_element_type_28, convolution_7, convert_element_type_30, squeeze_17, convert_element_type_32, convolution_8, mean, convert_element_type_34, relu, convert_element_type_36, convolution_10, mul_31, avg_pool2d, squeeze_19, convert_element_type_40, convolution_11, squeeze_21, convert_element_type_42, convolution_12, convert_element_type_44, squeeze_23, convert_element_type_46, convolution_13, convert_element_type_48, squeeze_25, convert_element_type_50, convolution_14, convert_element_type_52, squeeze_27, convert_element_type_54, convolution_15, mean_1, convert_element_type_56, relu_1, convert_element_type_58, convolution_17, mul_50, squeeze_29, convert_element_type_62, convolution_18, convert_element_type_64, squeeze_31, convert_element_type_66, convolution_19, convert_element_type_68, squeeze_33, convert_element_type_70, convolution_20, convert_element_type_72, squeeze_35, convert_element_type_74, convolution_21, mean_2, convert_element_type_76, relu_2, convert_element_type_78, convolution_23, add_35, mul_66, avg_pool2d_1, squeeze_37, convert_element_type_82, convolution_24, squeeze_39, convert_element_type_84, convolution_25, convert_element_type_86, squeeze_41, convert_element_type_88, convolution_26, convert_element_type_90, squeeze_43, convert_element_type_92, convolution_27, convert_element_type_94, squeeze_45, convert_element_type_96, convolution_28, mean_3, convert_element_type_98, relu_3, convert_element_type_100, convolution_30, mul_85, squeeze_47, convert_element_type_104, convolution_31, convert_element_type_106, squeeze_49, convert_element_type_108, convolution_32, convert_element_type_110, squeeze_51, convert_element_type_112, convolution_33, convert_element_type_114, squeeze_53, convert_element_type_116, convolution_34, mean_4, convert_element_type_118, relu_4, convert_element_type_120, convolution_36, add_54, mul_101, squeeze_55, convert_element_type_124, convolution_37, convert_element_type_126, squeeze_57, convert_element_type_128, convolution_38, convert_element_type_130, squeeze_59, convert_element_type_132, convolution_39, convert_element_type_134, squeeze_61, convert_element_type_136, convolution_40, mean_5, convert_element_type_138, relu_5, convert_element_type_140, convolution_42, add_63, mul_117, squeeze_63, convert_element_type_144, convolution_43, convert_element_type_146, squeeze_65, convert_element_type_148, convolution_44, convert_element_type_150, squeeze_67, convert_element_type_152, convolution_45, convert_element_type_154, squeeze_69, convert_element_type_156, convolution_46, mean_6, convert_element_type_158, relu_6, convert_element_type_160, convolution_48, add_72, mul_133, squeeze_71, convert_element_type_164, convolution_49, convert_element_type_166, squeeze_73, convert_element_type_168, convolution_50, convert_element_type_170, squeeze_75, convert_element_type_172, convolution_51, convert_element_type_174, squeeze_77, convert_element_type_176, convolution_52, mean_7, convert_element_type_178, relu_7, convert_element_type_180, convolution_54, add_81, mul_149, squeeze_79, convert_element_type_184, convolution_55, convert_element_type_186, squeeze_81, convert_element_type_188, convolution_56, convert_element_type_190, squeeze_83, convert_element_type_192, convolution_57, convert_element_type_194, squeeze_85, convert_element_type_196, convolution_58, mean_8, convert_element_type_198, relu_8, convert_element_type_200, convolution_60, add_90, mul_165, avg_pool2d_2, squeeze_87, convert_element_type_204, convolution_61, squeeze_89, convert_element_type_206, convolution_62, convert_element_type_208, squeeze_91, convert_element_type_210, convolution_63, convert_element_type_212, squeeze_93, convert_element_type_214, convolution_64, convert_element_type_216, squeeze_95, convert_element_type_218, convolution_65, mean_9, convert_element_type_220, relu_9, convert_element_type_222, convolution_67, mul_184, squeeze_97, convert_element_type_226, convolution_68, convert_element_type_228, squeeze_99, convert_element_type_230, convolution_69, convert_element_type_232, squeeze_101, convert_element_type_234, convolution_70, convert_element_type_236, squeeze_103, convert_element_type_238, convolution_71, mean_10, convert_element_type_240, relu_10, convert_element_type_242, convolution_73, add_109, mul_200, squeeze_105, convert_element_type_246, convolution_74, convert_element_type_248, squeeze_107, convert_element_type_250, convolution_75, convert_element_type_252, squeeze_109, convert_element_type_254, convolution_76, convert_element_type_256, squeeze_111, convert_element_type_258, convolution_77, mean_11, convert_element_type_260, relu_11, convert_element_type_262, convolution_79, add_118, squeeze_113, convert_element_type_264, convolution_80, view_171, permute_1, unsqueeze_58, unsqueeze_66, unsqueeze_74, unsqueeze_82, unsqueeze_90, unsqueeze_98, unsqueeze_106, unsqueeze_114, unsqueeze_122, unsqueeze_130, unsqueeze_138, unsqueeze_146, unsqueeze_154, unsqueeze_162, unsqueeze_170, unsqueeze_178, unsqueeze_186, unsqueeze_194, unsqueeze_202, unsqueeze_210, unsqueeze_218, unsqueeze_226, unsqueeze_234, unsqueeze_242, unsqueeze_250, unsqueeze_258, unsqueeze_266, unsqueeze_274, unsqueeze_282, unsqueeze_290, unsqueeze_298, unsqueeze_306, unsqueeze_314, unsqueeze_322, unsqueeze_330, unsqueeze_338, unsqueeze_346, unsqueeze_354, unsqueeze_362, unsqueeze_370, unsqueeze_378, unsqueeze_386, unsqueeze_394, unsqueeze_402, unsqueeze_410, unsqueeze_418, unsqueeze_426, unsqueeze_434, unsqueeze_442, unsqueeze_450, unsqueeze_458, unsqueeze_466, unsqueeze_474, unsqueeze_482, unsqueeze_490, unsqueeze_498, unsqueeze_506)
