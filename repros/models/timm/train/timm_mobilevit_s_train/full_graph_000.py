class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[16, 3, 3, 3]", primals_2: "f32[128, 3, 256, 256]", primals_3: "i64[]", primals_4: "f32[16]", primals_5: "f32[16]", primals_6: "f32[16]", primals_7: "f32[16]", primals_8: "f32[64, 16, 1, 1]", primals_9: "i64[]", primals_10: "f32[64]", primals_11: "f32[64]", primals_12: "f32[64]", primals_13: "f32[64]", primals_14: "f32[64, 1, 3, 3]", primals_15: "i64[]", primals_16: "f32[64]", primals_17: "f32[64]", primals_18: "f32[64]", primals_19: "f32[64]", primals_20: "f32[32, 64, 1, 1]", primals_21: "i64[]", primals_22: "f32[32]", primals_23: "f32[32]", primals_24: "f32[32]", primals_25: "f32[32]", primals_26: "f32[128, 32, 1, 1]", primals_27: "i64[]", primals_28: "f32[128]", primals_29: "f32[128]", primals_30: "f32[128]", primals_31: "f32[128]", primals_32: "f32[128, 1, 3, 3]", primals_33: "i64[]", primals_34: "f32[128]", primals_35: "f32[128]", primals_36: "f32[128]", primals_37: "f32[128]", primals_38: "f32[64, 128, 1, 1]", primals_39: "i64[]", primals_40: "f32[64]", primals_41: "f32[64]", primals_42: "f32[64]", primals_43: "f32[64]", primals_44: "f32[256, 64, 1, 1]", primals_45: "i64[]", primals_46: "f32[256]", primals_47: "f32[256]", primals_48: "f32[256]", primals_49: "f32[256]", primals_50: "f32[256, 1, 3, 3]", primals_51: "i64[]", primals_52: "f32[256]", primals_53: "f32[256]", primals_54: "f32[256]", primals_55: "f32[256]", primals_56: "f32[64, 256, 1, 1]", primals_57: "i64[]", primals_58: "f32[64]", primals_59: "f32[64]", primals_60: "f32[64]", primals_61: "f32[64]", primals_62: "f32[256, 64, 1, 1]", primals_63: "i64[]", primals_64: "f32[256]", primals_65: "f32[256]", primals_66: "f32[256]", primals_67: "f32[256]", primals_68: "f32[256, 1, 3, 3]", primals_69: "i64[]", primals_70: "f32[256]", primals_71: "f32[256]", primals_72: "f32[256]", primals_73: "f32[256]", primals_74: "f32[64, 256, 1, 1]", primals_75: "i64[]", primals_76: "f32[64]", primals_77: "f32[64]", primals_78: "f32[64]", primals_79: "f32[64]", primals_80: "f32[256, 64, 1, 1]", primals_81: "i64[]", primals_82: "f32[256]", primals_83: "f32[256]", primals_84: "f32[256]", primals_85: "f32[256]", primals_86: "f32[256, 1, 3, 3]", primals_87: "i64[]", primals_88: "f32[256]", primals_89: "f32[256]", primals_90: "f32[256]", primals_91: "f32[256]", primals_92: "f32[96, 256, 1, 1]", primals_93: "i64[]", primals_94: "f32[96]", primals_95: "f32[96]", primals_96: "f32[96]", primals_97: "f32[96]", primals_98: "f32[96, 96, 3, 3]", primals_99: "i64[]", primals_100: "f32[96]", primals_101: "f32[96]", primals_102: "f32[96]", primals_103: "f32[96]", primals_104: "f32[144, 96, 1, 1]", primals_105: "f32[144]", primals_106: "f32[144]", primals_107: "f32[432, 144]", primals_108: "f32[432]", primals_109: "f32[144, 144]", primals_110: "f32[144]", primals_111: "f32[144]", primals_112: "f32[144]", primals_113: "f32[288, 144]", primals_114: "f32[288]", primals_115: "f32[144, 288]", primals_116: "f32[144]", primals_117: "f32[144]", primals_118: "f32[144]", primals_119: "f32[432, 144]", primals_120: "f32[432]", primals_121: "f32[144, 144]", primals_122: "f32[144]", primals_123: "f32[144]", primals_124: "f32[144]", primals_125: "f32[288, 144]", primals_126: "f32[288]", primals_127: "f32[144, 288]", primals_128: "f32[144]", primals_129: "f32[144]", primals_130: "f32[144]", primals_131: "f32[96, 144, 1, 1]", primals_132: "i64[]", primals_133: "f32[96]", primals_134: "f32[96]", primals_135: "f32[96]", primals_136: "f32[96]", primals_137: "f32[96, 192, 3, 3]", primals_138: "i64[]", primals_139: "f32[96]", primals_140: "f32[96]", primals_141: "f32[96]", primals_142: "f32[96]", primals_143: "f32[384, 96, 1, 1]", primals_144: "i64[]", primals_145: "f32[384]", primals_146: "f32[384]", primals_147: "f32[384]", primals_148: "f32[384]", primals_149: "f32[384, 1, 3, 3]", primals_150: "i64[]", primals_151: "f32[384]", primals_152: "f32[384]", primals_153: "f32[384]", primals_154: "f32[384]", primals_155: "f32[128, 384, 1, 1]", primals_156: "i64[]", primals_157: "f32[128]", primals_158: "f32[128]", primals_159: "f32[128]", primals_160: "f32[128]", primals_161: "f32[128, 128, 3, 3]", primals_162: "i64[]", primals_163: "f32[128]", primals_164: "f32[128]", primals_165: "f32[128]", primals_166: "f32[128]", primals_167: "f32[192, 128, 1, 1]", primals_168: "f32[192]", primals_169: "f32[192]", primals_170: "f32[576, 192]", primals_171: "f32[576]", primals_172: "f32[192, 192]", primals_173: "f32[192]", primals_174: "f32[192]", primals_175: "f32[192]", primals_176: "f32[384, 192]", primals_177: "f32[384]", primals_178: "f32[192, 384]", primals_179: "f32[192]", primals_180: "f32[192]", primals_181: "f32[192]", primals_182: "f32[576, 192]", primals_183: "f32[576]", primals_184: "f32[192, 192]", primals_185: "f32[192]", primals_186: "f32[192]", primals_187: "f32[192]", primals_188: "f32[384, 192]", primals_189: "f32[384]", primals_190: "f32[192, 384]", primals_191: "f32[192]", primals_192: "f32[192]", primals_193: "f32[192]", primals_194: "f32[576, 192]", primals_195: "f32[576]", primals_196: "f32[192, 192]", primals_197: "f32[192]", primals_198: "f32[192]", primals_199: "f32[192]", primals_200: "f32[384, 192]", primals_201: "f32[384]", primals_202: "f32[192, 384]", primals_203: "f32[192]", primals_204: "f32[192]", primals_205: "f32[192]", primals_206: "f32[576, 192]", primals_207: "f32[576]", primals_208: "f32[192, 192]", primals_209: "f32[192]", primals_210: "f32[192]", primals_211: "f32[192]", primals_212: "f32[384, 192]", primals_213: "f32[384]", primals_214: "f32[192, 384]", primals_215: "f32[192]", primals_216: "f32[192]", primals_217: "f32[192]", primals_218: "f32[128, 192, 1, 1]", primals_219: "i64[]", primals_220: "f32[128]", primals_221: "f32[128]", primals_222: "f32[128]", primals_223: "f32[128]", primals_224: "f32[128, 256, 3, 3]", primals_225: "i64[]", primals_226: "f32[128]", primals_227: "f32[128]", primals_228: "f32[128]", primals_229: "f32[128]", primals_230: "f32[512, 128, 1, 1]", primals_231: "i64[]", primals_232: "f32[512]", primals_233: "f32[512]", primals_234: "f32[512]", primals_235: "f32[512]", primals_236: "f32[512, 1, 3, 3]", primals_237: "i64[]", primals_238: "f32[512]", primals_239: "f32[512]", primals_240: "f32[512]", primals_241: "f32[512]", primals_242: "f32[160, 512, 1, 1]", primals_243: "i64[]", primals_244: "f32[160]", primals_245: "f32[160]", primals_246: "f32[160]", primals_247: "f32[160]", primals_248: "f32[160, 160, 3, 3]", primals_249: "i64[]", primals_250: "f32[160]", primals_251: "f32[160]", primals_252: "f32[160]", primals_253: "f32[160]", primals_254: "f32[240, 160, 1, 1]", primals_255: "f32[240]", primals_256: "f32[240]", primals_257: "f32[720, 240]", primals_258: "f32[720]", primals_259: "f32[240, 240]", primals_260: "f32[240]", primals_261: "f32[240]", primals_262: "f32[240]", primals_263: "f32[480, 240]", primals_264: "f32[480]", primals_265: "f32[240, 480]", primals_266: "f32[240]", primals_267: "f32[240]", primals_268: "f32[240]", primals_269: "f32[720, 240]", primals_270: "f32[720]", primals_271: "f32[240, 240]", primals_272: "f32[240]", primals_273: "f32[240]", primals_274: "f32[240]", primals_275: "f32[480, 240]", primals_276: "f32[480]", primals_277: "f32[240, 480]", primals_278: "f32[240]", primals_279: "f32[240]", primals_280: "f32[240]", primals_281: "f32[720, 240]", primals_282: "f32[720]", primals_283: "f32[240, 240]", primals_284: "f32[240]", primals_285: "f32[240]", primals_286: "f32[240]", primals_287: "f32[480, 240]", primals_288: "f32[480]", primals_289: "f32[240, 480]", primals_290: "f32[240]", primals_291: "f32[240]", primals_292: "f32[240]", primals_293: "f32[160, 240, 1, 1]", primals_294: "i64[]", primals_295: "f32[160]", primals_296: "f32[160]", primals_297: "f32[160]", primals_298: "f32[160]", primals_299: "f32[160, 320, 3, 3]", primals_300: "i64[]", primals_301: "f32[160]", primals_302: "f32[160]", primals_303: "f32[160]", primals_304: "f32[160]", primals_305: "f32[640, 160, 1, 1]", primals_306: "i64[]", primals_307: "f32[640]", primals_308: "f32[640]", primals_309: "f32[640]", primals_310: "f32[640]", primals_311: "f32[1000, 640]", primals_312: "f32[1000]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution: "f32[128, 16, 128, 128]" = torch.ops.aten.convolution.default(primals_2, primals_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add: "i64[]" = torch.ops.aten.add.Tensor(primals_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 16, 1, 1]" = var_mean[0]
        getitem_1: "f32[1, 16, 1, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[128, 16, 128, 128]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul: "f32[128, 16, 128, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3])
        mul_1: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze, 0.1);  squeeze = None
        mul_2: "f32[16]" = torch.ops.aten.mul.Tensor(primals_4, 0.9)
        add_2: "f32[16]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_2: "f32[16]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_3: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.0000004768373856);  squeeze_2 = None
        mul_4: "f32[16]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
        mul_5: "f32[16]" = torch.ops.aten.mul.Tensor(primals_5, 0.9)
        add_3: "f32[16]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1)
        unsqueeze_1: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[128, 16, 128, 128]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1)
        unsqueeze_3: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_4: "f32[128, 16, 128, 128]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg: "f32[128, 16, 128, 128]" = torch.ops.aten.neg.default(add_4)
        exp: "f32[128, 16, 128, 128]" = torch.ops.aten.exp.default(neg);  neg = None
        add_5: "f32[128, 16, 128, 128]" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[128, 16, 128, 128]" = torch.ops.aten.div.Tensor(add_4, add_5);  add_4 = add_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_1: "f32[128, 64, 128, 128]" = torch.ops.aten.convolution.default(div, primals_8, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_6: "i64[]" = torch.ops.aten.add.Tensor(primals_9, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[1, 64, 1, 1]" = var_mean_1[0]
        getitem_3: "f32[1, 64, 1, 1]" = var_mean_1[1];  var_mean_1 = None
        add_7: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
        rsqrt_1: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        sub_1: "f32[128, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
        mul_7: "f32[128, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        squeeze_3: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3])
        mul_8: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1);  squeeze_3 = None
        mul_9: "f32[64]" = torch.ops.aten.mul.Tensor(primals_10, 0.9)
        add_8: "f32[64]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        squeeze_5: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_10: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000004768373856);  squeeze_5 = None
        mul_11: "f32[64]" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
        mul_12: "f32[64]" = torch.ops.aten.mul.Tensor(primals_11, 0.9)
        add_9: "f32[64]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        unsqueeze_4: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1)
        unsqueeze_5: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_13: "f32[128, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
        unsqueeze_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1)
        unsqueeze_7: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_10: "f32[128, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_1: "f32[128, 64, 128, 128]" = torch.ops.aten.neg.default(add_10)
        exp_1: "f32[128, 64, 128, 128]" = torch.ops.aten.exp.default(neg_1);  neg_1 = None
        add_11: "f32[128, 64, 128, 128]" = torch.ops.aten.add.Tensor(exp_1, 1);  exp_1 = None
        div_1: "f32[128, 64, 128, 128]" = torch.ops.aten.div.Tensor(add_10, add_11);  add_10 = add_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_2: "f32[128, 64, 128, 128]" = torch.ops.aten.convolution.default(div_1, primals_14, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_12: "i64[]" = torch.ops.aten.add.Tensor(primals_15, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
        getitem_4: "f32[1, 64, 1, 1]" = var_mean_2[0]
        getitem_5: "f32[1, 64, 1, 1]" = var_mean_2[1];  var_mean_2 = None
        add_13: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
        rsqrt_2: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_2: "f32[128, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
        mul_14: "f32[128, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        squeeze_6: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3])
        mul_15: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1);  squeeze_6 = None
        mul_16: "f32[64]" = torch.ops.aten.mul.Tensor(primals_16, 0.9)
        add_14: "f32[64]" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
        squeeze_8: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_17: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0000004768373856);  squeeze_8 = None
        mul_18: "f32[64]" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
        mul_19: "f32[64]" = torch.ops.aten.mul.Tensor(primals_17, 0.9)
        add_15: "f32[64]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
        unsqueeze_8: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1)
        unsqueeze_9: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        mul_20: "f32[128, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
        unsqueeze_10: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1)
        unsqueeze_11: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        add_16: "f32[128, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_2: "f32[128, 64, 128, 128]" = torch.ops.aten.neg.default(add_16)
        exp_2: "f32[128, 64, 128, 128]" = torch.ops.aten.exp.default(neg_2);  neg_2 = None
        add_17: "f32[128, 64, 128, 128]" = torch.ops.aten.add.Tensor(exp_2, 1);  exp_2 = None
        div_2: "f32[128, 64, 128, 128]" = torch.ops.aten.div.Tensor(add_16, add_17);  add_16 = add_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_3: "f32[128, 32, 128, 128]" = torch.ops.aten.convolution.default(div_2, primals_20, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_18: "i64[]" = torch.ops.aten.add.Tensor(primals_21, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
        getitem_6: "f32[1, 32, 1, 1]" = var_mean_3[0]
        getitem_7: "f32[1, 32, 1, 1]" = var_mean_3[1];  var_mean_3 = None
        add_19: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
        rsqrt_3: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        sub_3: "f32[128, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_7)
        mul_21: "f32[128, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        squeeze_9: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        squeeze_10: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_22: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
        mul_23: "f32[32]" = torch.ops.aten.mul.Tensor(primals_22, 0.9)
        add_20: "f32[32]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
        squeeze_11: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_24: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0000004768373856);  squeeze_11 = None
        mul_25: "f32[32]" = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
        mul_26: "f32[32]" = torch.ops.aten.mul.Tensor(primals_23, 0.9)
        add_21: "f32[32]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
        unsqueeze_12: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1)
        unsqueeze_13: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_27: "f32[128, 32, 128, 128]" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
        unsqueeze_14: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1);  primals_25 = None
        unsqueeze_15: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_22: "f32[128, 32, 128, 128]" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_4: "f32[128, 128, 128, 128]" = torch.ops.aten.convolution.default(add_22, primals_26, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_23: "i64[]" = torch.ops.aten.add.Tensor(primals_27, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
        getitem_8: "f32[1, 128, 1, 1]" = var_mean_4[0]
        getitem_9: "f32[1, 128, 1, 1]" = var_mean_4[1];  var_mean_4 = None
        add_24: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
        rsqrt_4: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        sub_4: "f32[128, 128, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_9)
        mul_28: "f32[128, 128, 128, 128]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        squeeze_12: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3])
        mul_29: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1);  squeeze_12 = None
        mul_30: "f32[128]" = torch.ops.aten.mul.Tensor(primals_28, 0.9)
        add_25: "f32[128]" = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
        squeeze_14: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_31: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000004768373856);  squeeze_14 = None
        mul_32: "f32[128]" = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
        mul_33: "f32[128]" = torch.ops.aten.mul.Tensor(primals_29, 0.9)
        add_26: "f32[128]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
        unsqueeze_16: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1)
        unsqueeze_17: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        mul_34: "f32[128, 128, 128, 128]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
        unsqueeze_18: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1)
        unsqueeze_19: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        add_27: "f32[128, 128, 128, 128]" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_3: "f32[128, 128, 128, 128]" = torch.ops.aten.neg.default(add_27)
        exp_3: "f32[128, 128, 128, 128]" = torch.ops.aten.exp.default(neg_3);  neg_3 = None
        add_28: "f32[128, 128, 128, 128]" = torch.ops.aten.add.Tensor(exp_3, 1);  exp_3 = None
        div_3: "f32[128, 128, 128, 128]" = torch.ops.aten.div.Tensor(add_27, add_28);  add_27 = add_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_5: "f32[128, 128, 64, 64]" = torch.ops.aten.convolution.default(div_3, primals_32, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 128)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_29: "i64[]" = torch.ops.aten.add.Tensor(primals_33, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_5 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
        getitem_10: "f32[1, 128, 1, 1]" = var_mean_5[0]
        getitem_11: "f32[1, 128, 1, 1]" = var_mean_5[1];  var_mean_5 = None
        add_30: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
        rsqrt_5: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        sub_5: "f32[128, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_11)
        mul_35: "f32[128, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        squeeze_15: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3])
        mul_36: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1);  squeeze_15 = None
        mul_37: "f32[128]" = torch.ops.aten.mul.Tensor(primals_34, 0.9)
        add_31: "f32[128]" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
        squeeze_17: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_38: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0000019073522708);  squeeze_17 = None
        mul_39: "f32[128]" = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
        mul_40: "f32[128]" = torch.ops.aten.mul.Tensor(primals_35, 0.9)
        add_32: "f32[128]" = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
        unsqueeze_20: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1)
        unsqueeze_21: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_41: "f32[128, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
        unsqueeze_22: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1)
        unsqueeze_23: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_33: "f32[128, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_4: "f32[128, 128, 64, 64]" = torch.ops.aten.neg.default(add_33)
        exp_4: "f32[128, 128, 64, 64]" = torch.ops.aten.exp.default(neg_4);  neg_4 = None
        add_34: "f32[128, 128, 64, 64]" = torch.ops.aten.add.Tensor(exp_4, 1);  exp_4 = None
        div_4: "f32[128, 128, 64, 64]" = torch.ops.aten.div.Tensor(add_33, add_34);  add_33 = add_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_6: "f32[128, 64, 64, 64]" = torch.ops.aten.convolution.default(div_4, primals_38, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_35: "i64[]" = torch.ops.aten.add.Tensor(primals_39, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_6 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
        getitem_12: "f32[1, 64, 1, 1]" = var_mean_6[0]
        getitem_13: "f32[1, 64, 1, 1]" = var_mean_6[1];  var_mean_6 = None
        add_36: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
        rsqrt_6: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_6: "f32[128, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_13)
        mul_42: "f32[128, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        squeeze_18: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        squeeze_19: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_43: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
        mul_44: "f32[64]" = torch.ops.aten.mul.Tensor(primals_40, 0.9)
        add_37: "f32[64]" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
        squeeze_20: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
        mul_45: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0000019073522708);  squeeze_20 = None
        mul_46: "f32[64]" = torch.ops.aten.mul.Tensor(mul_45, 0.1);  mul_45 = None
        mul_47: "f32[64]" = torch.ops.aten.mul.Tensor(primals_41, 0.9)
        add_38: "f32[64]" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
        unsqueeze_24: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1)
        unsqueeze_25: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        mul_48: "f32[128, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_25);  mul_42 = unsqueeze_25 = None
        unsqueeze_26: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1);  primals_43 = None
        unsqueeze_27: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        add_39: "f32[128, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_48, unsqueeze_27);  mul_48 = unsqueeze_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_7: "f32[128, 256, 64, 64]" = torch.ops.aten.convolution.default(add_39, primals_44, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_40: "i64[]" = torch.ops.aten.add.Tensor(primals_45, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_7 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
        getitem_14: "f32[1, 256, 1, 1]" = var_mean_7[0]
        getitem_15: "f32[1, 256, 1, 1]" = var_mean_7[1];  var_mean_7 = None
        add_41: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
        rsqrt_7: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
        sub_7: "f32[128, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_15)
        mul_49: "f32[128, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        squeeze_21: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3])
        mul_50: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1);  squeeze_21 = None
        mul_51: "f32[256]" = torch.ops.aten.mul.Tensor(primals_46, 0.9)
        add_42: "f32[256]" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
        squeeze_23: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
        mul_52: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0000019073522708);  squeeze_23 = None
        mul_53: "f32[256]" = torch.ops.aten.mul.Tensor(mul_52, 0.1);  mul_52 = None
        mul_54: "f32[256]" = torch.ops.aten.mul.Tensor(primals_47, 0.9)
        add_43: "f32[256]" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
        unsqueeze_28: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1)
        unsqueeze_29: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_55: "f32[128, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
        unsqueeze_30: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1)
        unsqueeze_31: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_44: "f32[128, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_5: "f32[128, 256, 64, 64]" = torch.ops.aten.neg.default(add_44)
        exp_5: "f32[128, 256, 64, 64]" = torch.ops.aten.exp.default(neg_5);  neg_5 = None
        add_45: "f32[128, 256, 64, 64]" = torch.ops.aten.add.Tensor(exp_5, 1);  exp_5 = None
        div_5: "f32[128, 256, 64, 64]" = torch.ops.aten.div.Tensor(add_44, add_45);  add_44 = add_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_8: "f32[128, 256, 64, 64]" = torch.ops.aten.convolution.default(div_5, primals_50, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 256)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_46: "i64[]" = torch.ops.aten.add.Tensor(primals_51, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_8 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
        getitem_16: "f32[1, 256, 1, 1]" = var_mean_8[0]
        getitem_17: "f32[1, 256, 1, 1]" = var_mean_8[1];  var_mean_8 = None
        add_47: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
        rsqrt_8: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        sub_8: "f32[128, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_17)
        mul_56: "f32[128, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        squeeze_24: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3])
        mul_57: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1);  squeeze_24 = None
        mul_58: "f32[256]" = torch.ops.aten.mul.Tensor(primals_52, 0.9)
        add_48: "f32[256]" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
        squeeze_26: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
        mul_59: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0000019073522708);  squeeze_26 = None
        mul_60: "f32[256]" = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
        mul_61: "f32[256]" = torch.ops.aten.mul.Tensor(primals_53, 0.9)
        add_49: "f32[256]" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
        unsqueeze_32: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1)
        unsqueeze_33: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        mul_62: "f32[128, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
        unsqueeze_34: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1)
        unsqueeze_35: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        add_50: "f32[128, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_6: "f32[128, 256, 64, 64]" = torch.ops.aten.neg.default(add_50)
        exp_6: "f32[128, 256, 64, 64]" = torch.ops.aten.exp.default(neg_6);  neg_6 = None
        add_51: "f32[128, 256, 64, 64]" = torch.ops.aten.add.Tensor(exp_6, 1);  exp_6 = None
        div_6: "f32[128, 256, 64, 64]" = torch.ops.aten.div.Tensor(add_50, add_51);  add_50 = add_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_9: "f32[128, 64, 64, 64]" = torch.ops.aten.convolution.default(div_6, primals_56, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_52: "i64[]" = torch.ops.aten.add.Tensor(primals_57, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_9 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
        getitem_18: "f32[1, 64, 1, 1]" = var_mean_9[0]
        getitem_19: "f32[1, 64, 1, 1]" = var_mean_9[1];  var_mean_9 = None
        add_53: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
        rsqrt_9: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_9: "f32[128, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_19)
        mul_63: "f32[128, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        squeeze_27: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        squeeze_28: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_64: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
        mul_65: "f32[64]" = torch.ops.aten.mul.Tensor(primals_58, 0.9)
        add_54: "f32[64]" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
        squeeze_29: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
        mul_66: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0000019073522708);  squeeze_29 = None
        mul_67: "f32[64]" = torch.ops.aten.mul.Tensor(mul_66, 0.1);  mul_66 = None
        mul_68: "f32[64]" = torch.ops.aten.mul.Tensor(primals_59, 0.9)
        add_55: "f32[64]" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
        unsqueeze_36: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1)
        unsqueeze_37: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_69: "f32[128, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_37);  mul_63 = unsqueeze_37 = None
        unsqueeze_38: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_61, -1);  primals_61 = None
        unsqueeze_39: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_56: "f32[128, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_39);  mul_69 = unsqueeze_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:501 in forward, code: x = x + self.shortcut(shortcut)
        add_57: "f32[128, 64, 64, 64]" = torch.ops.aten.add.Tensor(add_56, add_39);  add_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_10: "f32[128, 256, 64, 64]" = torch.ops.aten.convolution.default(add_57, primals_62, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_58: "i64[]" = torch.ops.aten.add.Tensor(primals_63, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_10 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
        getitem_20: "f32[1, 256, 1, 1]" = var_mean_10[0]
        getitem_21: "f32[1, 256, 1, 1]" = var_mean_10[1];  var_mean_10 = None
        add_59: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
        rsqrt_10: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
        sub_10: "f32[128, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_21)
        mul_70: "f32[128, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        squeeze_30: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3])
        mul_71: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1);  squeeze_30 = None
        mul_72: "f32[256]" = torch.ops.aten.mul.Tensor(primals_64, 0.9)
        add_60: "f32[256]" = torch.ops.aten.add.Tensor(mul_71, mul_72);  mul_71 = mul_72 = None
        squeeze_32: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        mul_73: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0000019073522708);  squeeze_32 = None
        mul_74: "f32[256]" = torch.ops.aten.mul.Tensor(mul_73, 0.1);  mul_73 = None
        mul_75: "f32[256]" = torch.ops.aten.mul.Tensor(primals_65, 0.9)
        add_61: "f32[256]" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
        unsqueeze_40: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1)
        unsqueeze_41: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        mul_76: "f32[128, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_41);  mul_70 = unsqueeze_41 = None
        unsqueeze_42: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_67, -1)
        unsqueeze_43: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        add_62: "f32[128, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_76, unsqueeze_43);  mul_76 = unsqueeze_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_7: "f32[128, 256, 64, 64]" = torch.ops.aten.neg.default(add_62)
        exp_7: "f32[128, 256, 64, 64]" = torch.ops.aten.exp.default(neg_7);  neg_7 = None
        add_63: "f32[128, 256, 64, 64]" = torch.ops.aten.add.Tensor(exp_7, 1);  exp_7 = None
        div_7: "f32[128, 256, 64, 64]" = torch.ops.aten.div.Tensor(add_62, add_63);  add_62 = add_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_11: "f32[128, 256, 64, 64]" = torch.ops.aten.convolution.default(div_7, primals_68, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 256)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_64: "i64[]" = torch.ops.aten.add.Tensor(primals_69, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_11 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
        getitem_22: "f32[1, 256, 1, 1]" = var_mean_11[0]
        getitem_23: "f32[1, 256, 1, 1]" = var_mean_11[1];  var_mean_11 = None
        add_65: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
        rsqrt_11: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        sub_11: "f32[128, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_23)
        mul_77: "f32[128, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        squeeze_33: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3])
        mul_78: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1);  squeeze_33 = None
        mul_79: "f32[256]" = torch.ops.aten.mul.Tensor(primals_70, 0.9)
        add_66: "f32[256]" = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
        squeeze_35: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
        mul_80: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0000019073522708);  squeeze_35 = None
        mul_81: "f32[256]" = torch.ops.aten.mul.Tensor(mul_80, 0.1);  mul_80 = None
        mul_82: "f32[256]" = torch.ops.aten.mul.Tensor(primals_71, 0.9)
        add_67: "f32[256]" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
        unsqueeze_44: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1)
        unsqueeze_45: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_83: "f32[128, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_45);  mul_77 = unsqueeze_45 = None
        unsqueeze_46: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_73, -1)
        unsqueeze_47: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_68: "f32[128, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_47);  mul_83 = unsqueeze_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_8: "f32[128, 256, 64, 64]" = torch.ops.aten.neg.default(add_68)
        exp_8: "f32[128, 256, 64, 64]" = torch.ops.aten.exp.default(neg_8);  neg_8 = None
        add_69: "f32[128, 256, 64, 64]" = torch.ops.aten.add.Tensor(exp_8, 1);  exp_8 = None
        div_8: "f32[128, 256, 64, 64]" = torch.ops.aten.div.Tensor(add_68, add_69);  add_68 = add_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_12: "f32[128, 64, 64, 64]" = torch.ops.aten.convolution.default(div_8, primals_74, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_70: "i64[]" = torch.ops.aten.add.Tensor(primals_75, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_12 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
        getitem_24: "f32[1, 64, 1, 1]" = var_mean_12[0]
        getitem_25: "f32[1, 64, 1, 1]" = var_mean_12[1];  var_mean_12 = None
        add_71: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
        rsqrt_12: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        sub_12: "f32[128, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_12, getitem_25)
        mul_84: "f32[128, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        squeeze_36: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        squeeze_37: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_85: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
        mul_86: "f32[64]" = torch.ops.aten.mul.Tensor(primals_76, 0.9)
        add_72: "f32[64]" = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
        squeeze_38: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
        mul_87: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0000019073522708);  squeeze_38 = None
        mul_88: "f32[64]" = torch.ops.aten.mul.Tensor(mul_87, 0.1);  mul_87 = None
        mul_89: "f32[64]" = torch.ops.aten.mul.Tensor(primals_77, 0.9)
        add_73: "f32[64]" = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
        unsqueeze_48: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1)
        unsqueeze_49: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        mul_90: "f32[128, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_49);  mul_84 = unsqueeze_49 = None
        unsqueeze_50: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_79, -1);  primals_79 = None
        unsqueeze_51: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        add_74: "f32[128, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_51);  mul_90 = unsqueeze_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:501 in forward, code: x = x + self.shortcut(shortcut)
        add_75: "f32[128, 64, 64, 64]" = torch.ops.aten.add.Tensor(add_74, add_57);  add_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_13: "f32[128, 256, 64, 64]" = torch.ops.aten.convolution.default(add_75, primals_80, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_76: "i64[]" = torch.ops.aten.add.Tensor(primals_81, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_13 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
        getitem_26: "f32[1, 256, 1, 1]" = var_mean_13[0]
        getitem_27: "f32[1, 256, 1, 1]" = var_mean_13[1];  var_mean_13 = None
        add_77: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
        rsqrt_13: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        sub_13: "f32[128, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_27)
        mul_91: "f32[128, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        squeeze_39: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3])
        mul_92: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1);  squeeze_39 = None
        mul_93: "f32[256]" = torch.ops.aten.mul.Tensor(primals_82, 0.9)
        add_78: "f32[256]" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
        squeeze_41: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        mul_94: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0000019073522708);  squeeze_41 = None
        mul_95: "f32[256]" = torch.ops.aten.mul.Tensor(mul_94, 0.1);  mul_94 = None
        mul_96: "f32[256]" = torch.ops.aten.mul.Tensor(primals_83, 0.9)
        add_79: "f32[256]" = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None
        unsqueeze_52: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1)
        unsqueeze_53: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_97: "f32[128, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_53);  mul_91 = unsqueeze_53 = None
        unsqueeze_54: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_85, -1)
        unsqueeze_55: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_80: "f32[128, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_55);  mul_97 = unsqueeze_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_9: "f32[128, 256, 64, 64]" = torch.ops.aten.neg.default(add_80)
        exp_9: "f32[128, 256, 64, 64]" = torch.ops.aten.exp.default(neg_9);  neg_9 = None
        add_81: "f32[128, 256, 64, 64]" = torch.ops.aten.add.Tensor(exp_9, 1);  exp_9 = None
        div_9: "f32[128, 256, 64, 64]" = torch.ops.aten.div.Tensor(add_80, add_81);  add_80 = add_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_14: "f32[128, 256, 32, 32]" = torch.ops.aten.convolution.default(div_9, primals_86, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 256)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_82: "i64[]" = torch.ops.aten.add.Tensor(primals_87, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_14 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
        getitem_28: "f32[1, 256, 1, 1]" = var_mean_14[0]
        getitem_29: "f32[1, 256, 1, 1]" = var_mean_14[1];  var_mean_14 = None
        add_83: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
        rsqrt_14: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
        sub_14: "f32[128, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_29)
        mul_98: "f32[128, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        squeeze_42: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3])
        mul_99: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1);  squeeze_42 = None
        mul_100: "f32[256]" = torch.ops.aten.mul.Tensor(primals_88, 0.9)
        add_84: "f32[256]" = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
        squeeze_44: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        mul_101: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0000076294527394);  squeeze_44 = None
        mul_102: "f32[256]" = torch.ops.aten.mul.Tensor(mul_101, 0.1);  mul_101 = None
        mul_103: "f32[256]" = torch.ops.aten.mul.Tensor(primals_89, 0.9)
        add_85: "f32[256]" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
        unsqueeze_56: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1)
        unsqueeze_57: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        mul_104: "f32[128, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_57);  mul_98 = unsqueeze_57 = None
        unsqueeze_58: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_91, -1)
        unsqueeze_59: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        add_86: "f32[128, 256, 32, 32]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_59);  mul_104 = unsqueeze_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_10: "f32[128, 256, 32, 32]" = torch.ops.aten.neg.default(add_86)
        exp_10: "f32[128, 256, 32, 32]" = torch.ops.aten.exp.default(neg_10);  neg_10 = None
        add_87: "f32[128, 256, 32, 32]" = torch.ops.aten.add.Tensor(exp_10, 1);  exp_10 = None
        div_10: "f32[128, 256, 32, 32]" = torch.ops.aten.div.Tensor(add_86, add_87);  add_86 = add_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_15: "f32[128, 96, 32, 32]" = torch.ops.aten.convolution.default(div_10, primals_92, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_88: "i64[]" = torch.ops.aten.add.Tensor(primals_93, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_15 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
        getitem_30: "f32[1, 96, 1, 1]" = var_mean_15[0]
        getitem_31: "f32[1, 96, 1, 1]" = var_mean_15[1];  var_mean_15 = None
        add_89: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
        rsqrt_15: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
        sub_15: "f32[128, 96, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_31)
        mul_105: "f32[128, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        squeeze_45: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
        squeeze_46: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_106: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
        mul_107: "f32[96]" = torch.ops.aten.mul.Tensor(primals_94, 0.9)
        add_90: "f32[96]" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
        squeeze_47: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
        mul_108: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0000076294527394);  squeeze_47 = None
        mul_109: "f32[96]" = torch.ops.aten.mul.Tensor(mul_108, 0.1);  mul_108 = None
        mul_110: "f32[96]" = torch.ops.aten.mul.Tensor(primals_95, 0.9)
        add_91: "f32[96]" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
        unsqueeze_60: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1)
        unsqueeze_61: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_111: "f32[128, 96, 32, 32]" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_61);  mul_105 = unsqueeze_61 = None
        unsqueeze_62: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_97, -1);  primals_97 = None
        unsqueeze_63: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_92: "f32[128, 96, 32, 32]" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_63);  mul_111 = unsqueeze_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_16: "f32[128, 96, 32, 32]" = torch.ops.aten.convolution.default(add_92, primals_98, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_93: "i64[]" = torch.ops.aten.add.Tensor(primals_99, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_16 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
        getitem_32: "f32[1, 96, 1, 1]" = var_mean_16[0]
        getitem_33: "f32[1, 96, 1, 1]" = var_mean_16[1];  var_mean_16 = None
        add_94: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
        rsqrt_16: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        sub_16: "f32[128, 96, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_33)
        mul_112: "f32[128, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        squeeze_48: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3])
        mul_113: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1);  squeeze_48 = None
        mul_114: "f32[96]" = torch.ops.aten.mul.Tensor(primals_100, 0.9)
        add_95: "f32[96]" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
        squeeze_50: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
        mul_115: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0000076294527394);  squeeze_50 = None
        mul_116: "f32[96]" = torch.ops.aten.mul.Tensor(mul_115, 0.1);  mul_115 = None
        mul_117: "f32[96]" = torch.ops.aten.mul.Tensor(primals_101, 0.9)
        add_96: "f32[96]" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
        unsqueeze_64: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_102, -1)
        unsqueeze_65: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        mul_118: "f32[128, 96, 32, 32]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_65);  mul_112 = unsqueeze_65 = None
        unsqueeze_66: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_103, -1)
        unsqueeze_67: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        add_97: "f32[128, 96, 32, 32]" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_67);  mul_118 = unsqueeze_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_11: "f32[128, 96, 32, 32]" = torch.ops.aten.neg.default(add_97)
        exp_11: "f32[128, 96, 32, 32]" = torch.ops.aten.exp.default(neg_11);  neg_11 = None
        add_98: "f32[128, 96, 32, 32]" = torch.ops.aten.add.Tensor(exp_11, 1);  exp_11 = None
        div_11: "f32[128, 96, 32, 32]" = torch.ops.aten.div.Tensor(add_97, add_98);  add_97 = add_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:243 in forward, code: x = self.conv_1x1(x)
        convolution_17: "f32[128, 144, 32, 32]" = torch.ops.aten.convolution.default(div_11, primals_104, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:258 in forward, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
        clone_12: "f32[128, 144, 32, 32]" = torch.ops.aten.clone.default(convolution_17, memory_format = torch.contiguous_format);  convolution_17 = None
        view: "f32[294912, 2, 16, 2]" = torch.ops.aten.reshape.default(clone_12, [294912, 2, 16, 2]);  clone_12 = None
        permute: "f32[294912, 16, 2, 2]" = torch.ops.aten.permute.default(view, [0, 2, 1, 3]);  view = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:260 in forward, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
        clone_13: "f32[294912, 16, 2, 2]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_1: "f32[128, 144, 256, 4]" = torch.ops.aten.reshape.default(clone_13, [128, 144, 256, 4]);  clone_13 = None
        permute_1: "f32[128, 4, 256, 144]" = torch.ops.aten.permute.default(view_1, [0, 3, 2, 1]);  view_1 = None
        clone_14: "f32[128, 4, 256, 144]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_2: "f32[512, 256, 144]" = torch.ops.aten.reshape.default(clone_14, [512, 256, 144]);  clone_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        var_mean_17 = torch.ops.aten.var_mean.correction(view_2, [2], correction = 0, keepdim = True)
        getitem_34: "f32[512, 256, 1]" = var_mean_17[0]
        getitem_35: "f32[512, 256, 1]" = var_mean_17[1];  var_mean_17 = None
        add_99: "f32[512, 256, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_17: "f32[512, 256, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
        sub_17: "f32[512, 256, 144]" = torch.ops.aten.sub.Tensor(view_2, getitem_35);  getitem_35 = None
        mul_119: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        mul_120: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(mul_119, primals_105)
        add_100: "f32[512, 256, 144]" = torch.ops.aten.add.Tensor(mul_120, primals_106);  mul_120 = primals_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_3: "f32[131072, 144]" = torch.ops.aten.reshape.default(add_100, [131072, 144]);  add_100 = None
        permute_2: "f32[144, 432]" = torch.ops.aten.permute.default(primals_107, [1, 0])
        addmm: "f32[131072, 432]" = torch.ops.aten.addmm.default(primals_108, view_3, permute_2);  primals_108 = permute_2 = None
        view_4: "f32[512, 256, 432]" = torch.ops.aten.reshape.default(addmm, [512, 256, 432]);  addmm = None
        view_5: "f32[512, 256, 3, 4, 36]" = torch.ops.aten.reshape.default(view_4, [512, 256, 3, 4, 36]);  view_4 = None
        permute_3: "f32[3, 512, 4, 256, 36]" = torch.ops.aten.permute.default(view_5, [2, 0, 3, 1, 4]);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind = torch.ops.aten.unbind.int(permute_3);  permute_3 = None
        getitem_36: "f32[512, 4, 256, 36]" = unbind[0]
        getitem_37: "f32[512, 4, 256, 36]" = unbind[1]
        getitem_38: "f32[512, 4, 256, 36]" = unbind[2];  unbind = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_36, getitem_37, getitem_38, None, True)
        getitem_39: "f32[512, 4, 256, 36]" = _scaled_dot_product_efficient_attention[0]
        getitem_40: "f32[512, 4, 256]" = _scaled_dot_product_efficient_attention[1]
        getitem_41: "i64[]" = _scaled_dot_product_efficient_attention[2]
        getitem_42: "i64[]" = _scaled_dot_product_efficient_attention[3];  _scaled_dot_product_efficient_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_4: "f32[512, 256, 4, 36]" = torch.ops.aten.permute.default(getitem_39, [0, 2, 1, 3])
        view_6: "f32[512, 256, 144]" = torch.ops.aten.reshape.default(permute_4, [512, 256, 144]);  permute_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_7: "f32[131072, 144]" = torch.ops.aten.reshape.default(view_6, [131072, 144]);  view_6 = None
        permute_5: "f32[144, 144]" = torch.ops.aten.permute.default(primals_109, [1, 0])
        addmm_1: "f32[131072, 144]" = torch.ops.aten.addmm.default(primals_110, view_7, permute_5);  primals_110 = view_7 = permute_5 = None
        view_8: "f32[512, 256, 144]" = torch.ops.aten.reshape.default(addmm_1, [512, 256, 144]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_101: "f32[512, 256, 144]" = torch.ops.aten.add.Tensor(view_2, view_8);  view_2 = view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_18 = torch.ops.aten.var_mean.correction(add_101, [2], correction = 0, keepdim = True)
        getitem_43: "f32[512, 256, 1]" = var_mean_18[0]
        getitem_44: "f32[512, 256, 1]" = var_mean_18[1];  var_mean_18 = None
        add_102: "f32[512, 256, 1]" = torch.ops.aten.add.Tensor(getitem_43, 1e-05);  getitem_43 = None
        rsqrt_18: "f32[512, 256, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        sub_18: "f32[512, 256, 144]" = torch.ops.aten.sub.Tensor(add_101, getitem_44);  getitem_44 = None
        mul_121: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        mul_122: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(mul_121, primals_111)
        add_103: "f32[512, 256, 144]" = torch.ops.aten.add.Tensor(mul_122, primals_112);  mul_122 = primals_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_9: "f32[131072, 144]" = torch.ops.aten.reshape.default(add_103, [131072, 144]);  add_103 = None
        permute_6: "f32[144, 288]" = torch.ops.aten.permute.default(primals_113, [1, 0])
        addmm_2: "f32[131072, 288]" = torch.ops.aten.addmm.default(primals_114, view_9, permute_6);  primals_114 = permute_6 = None
        view_10: "f32[512, 256, 288]" = torch.ops.aten.reshape.default(addmm_2, [512, 256, 288])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        neg_12: "f32[512, 256, 288]" = torch.ops.aten.neg.default(view_10)
        exp_12: "f32[512, 256, 288]" = torch.ops.aten.exp.default(neg_12);  neg_12 = None
        add_104: "f32[512, 256, 288]" = torch.ops.aten.add.Tensor(exp_12, 1);  exp_12 = None
        div_12: "f32[512, 256, 288]" = torch.ops.aten.div.Tensor(view_10, add_104);  view_10 = add_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_11: "f32[131072, 288]" = torch.ops.aten.reshape.default(div_12, [131072, 288]);  div_12 = None
        permute_7: "f32[288, 144]" = torch.ops.aten.permute.default(primals_115, [1, 0])
        addmm_3: "f32[131072, 144]" = torch.ops.aten.addmm.default(primals_116, view_11, permute_7);  primals_116 = permute_7 = None
        view_12: "f32[512, 256, 144]" = torch.ops.aten.reshape.default(addmm_3, [512, 256, 144]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_105: "f32[512, 256, 144]" = torch.ops.aten.add.Tensor(add_101, view_12);  add_101 = view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        var_mean_19 = torch.ops.aten.var_mean.correction(add_105, [2], correction = 0, keepdim = True)
        getitem_45: "f32[512, 256, 1]" = var_mean_19[0]
        getitem_46: "f32[512, 256, 1]" = var_mean_19[1];  var_mean_19 = None
        add_106: "f32[512, 256, 1]" = torch.ops.aten.add.Tensor(getitem_45, 1e-05);  getitem_45 = None
        rsqrt_19: "f32[512, 256, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        sub_19: "f32[512, 256, 144]" = torch.ops.aten.sub.Tensor(add_105, getitem_46);  getitem_46 = None
        mul_123: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        mul_124: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(mul_123, primals_117)
        add_107: "f32[512, 256, 144]" = torch.ops.aten.add.Tensor(mul_124, primals_118);  mul_124 = primals_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_13: "f32[131072, 144]" = torch.ops.aten.reshape.default(add_107, [131072, 144]);  add_107 = None
        permute_8: "f32[144, 432]" = torch.ops.aten.permute.default(primals_119, [1, 0])
        addmm_4: "f32[131072, 432]" = torch.ops.aten.addmm.default(primals_120, view_13, permute_8);  primals_120 = permute_8 = None
        view_14: "f32[512, 256, 432]" = torch.ops.aten.reshape.default(addmm_4, [512, 256, 432]);  addmm_4 = None
        view_15: "f32[512, 256, 3, 4, 36]" = torch.ops.aten.reshape.default(view_14, [512, 256, 3, 4, 36]);  view_14 = None
        permute_9: "f32[3, 512, 4, 256, 36]" = torch.ops.aten.permute.default(view_15, [2, 0, 3, 1, 4]);  view_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_1 = torch.ops.aten.unbind.int(permute_9);  permute_9 = None
        getitem_47: "f32[512, 4, 256, 36]" = unbind_1[0]
        getitem_48: "f32[512, 4, 256, 36]" = unbind_1[1]
        getitem_49: "f32[512, 4, 256, 36]" = unbind_1[2];  unbind_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_47, getitem_48, getitem_49, None, True)
        getitem_50: "f32[512, 4, 256, 36]" = _scaled_dot_product_efficient_attention_1[0]
        getitem_51: "f32[512, 4, 256]" = _scaled_dot_product_efficient_attention_1[1]
        getitem_52: "i64[]" = _scaled_dot_product_efficient_attention_1[2]
        getitem_53: "i64[]" = _scaled_dot_product_efficient_attention_1[3];  _scaled_dot_product_efficient_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_10: "f32[512, 256, 4, 36]" = torch.ops.aten.permute.default(getitem_50, [0, 2, 1, 3])
        view_16: "f32[512, 256, 144]" = torch.ops.aten.reshape.default(permute_10, [512, 256, 144]);  permute_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_17: "f32[131072, 144]" = torch.ops.aten.reshape.default(view_16, [131072, 144]);  view_16 = None
        permute_11: "f32[144, 144]" = torch.ops.aten.permute.default(primals_121, [1, 0])
        addmm_5: "f32[131072, 144]" = torch.ops.aten.addmm.default(primals_122, view_17, permute_11);  primals_122 = view_17 = permute_11 = None
        view_18: "f32[512, 256, 144]" = torch.ops.aten.reshape.default(addmm_5, [512, 256, 144]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_108: "f32[512, 256, 144]" = torch.ops.aten.add.Tensor(add_105, view_18);  add_105 = view_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_20 = torch.ops.aten.var_mean.correction(add_108, [2], correction = 0, keepdim = True)
        getitem_54: "f32[512, 256, 1]" = var_mean_20[0]
        getitem_55: "f32[512, 256, 1]" = var_mean_20[1];  var_mean_20 = None
        add_109: "f32[512, 256, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
        rsqrt_20: "f32[512, 256, 1]" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
        sub_20: "f32[512, 256, 144]" = torch.ops.aten.sub.Tensor(add_108, getitem_55);  getitem_55 = None
        mul_125: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
        mul_126: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(mul_125, primals_123)
        add_110: "f32[512, 256, 144]" = torch.ops.aten.add.Tensor(mul_126, primals_124);  mul_126 = primals_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_19: "f32[131072, 144]" = torch.ops.aten.reshape.default(add_110, [131072, 144]);  add_110 = None
        permute_12: "f32[144, 288]" = torch.ops.aten.permute.default(primals_125, [1, 0])
        addmm_6: "f32[131072, 288]" = torch.ops.aten.addmm.default(primals_126, view_19, permute_12);  primals_126 = permute_12 = None
        view_20: "f32[512, 256, 288]" = torch.ops.aten.reshape.default(addmm_6, [512, 256, 288])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        neg_13: "f32[512, 256, 288]" = torch.ops.aten.neg.default(view_20)
        exp_13: "f32[512, 256, 288]" = torch.ops.aten.exp.default(neg_13);  neg_13 = None
        add_111: "f32[512, 256, 288]" = torch.ops.aten.add.Tensor(exp_13, 1);  exp_13 = None
        div_13: "f32[512, 256, 288]" = torch.ops.aten.div.Tensor(view_20, add_111);  view_20 = add_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_21: "f32[131072, 288]" = torch.ops.aten.reshape.default(div_13, [131072, 288]);  div_13 = None
        permute_13: "f32[288, 144]" = torch.ops.aten.permute.default(primals_127, [1, 0])
        addmm_7: "f32[131072, 144]" = torch.ops.aten.addmm.default(primals_128, view_21, permute_13);  primals_128 = permute_13 = None
        view_22: "f32[512, 256, 144]" = torch.ops.aten.reshape.default(addmm_7, [512, 256, 144]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_112: "f32[512, 256, 144]" = torch.ops.aten.add.Tensor(add_108, view_22);  add_108 = view_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:264 in forward, code: x = self.norm(x)
        var_mean_21 = torch.ops.aten.var_mean.correction(add_112, [2], correction = 0, keepdim = True)
        getitem_56: "f32[512, 256, 1]" = var_mean_21[0]
        getitem_57: "f32[512, 256, 1]" = var_mean_21[1];  var_mean_21 = None
        add_113: "f32[512, 256, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
        rsqrt_21: "f32[512, 256, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
        sub_21: "f32[512, 256, 144]" = torch.ops.aten.sub.Tensor(add_112, getitem_57);  add_112 = getitem_57 = None
        mul_127: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
        mul_128: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(mul_127, primals_129)
        add_114: "f32[512, 256, 144]" = torch.ops.aten.add.Tensor(mul_128, primals_130);  mul_128 = primals_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:268 in forward, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
        view_23: "f32[128, 4, 256, 144]" = torch.ops.aten.reshape.default(add_114, [128, 4, 256, -1]);  add_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:269 in forward, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
        permute_14: "f32[128, 144, 256, 4]" = torch.ops.aten.permute.default(view_23, [0, 3, 2, 1]);  view_23 = None
        clone_21: "f32[128, 144, 256, 4]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
        view_24: "f32[294912, 16, 2, 2]" = torch.ops.aten.reshape.default(clone_21, [294912, 16, 2, 2]);  clone_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:271 in forward, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
        permute_15: "f32[294912, 2, 16, 2]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        clone_22: "f32[294912, 2, 16, 2]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
        view_25: "f32[128, 144, 32, 32]" = torch.ops.aten.reshape.default(clone_22, [128, 144, 32, 32]);  clone_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_18: "f32[128, 96, 32, 32]" = torch.ops.aten.convolution.default(view_25, primals_131, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_115: "i64[]" = torch.ops.aten.add.Tensor(primals_132, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_22 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
        getitem_58: "f32[1, 96, 1, 1]" = var_mean_22[0]
        getitem_59: "f32[1, 96, 1, 1]" = var_mean_22[1];  var_mean_22 = None
        add_116: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05)
        rsqrt_22: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        sub_22: "f32[128, 96, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_59)
        mul_129: "f32[128, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        squeeze_51: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3])
        mul_130: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1);  squeeze_51 = None
        mul_131: "f32[96]" = torch.ops.aten.mul.Tensor(primals_133, 0.9)
        add_117: "f32[96]" = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
        squeeze_53: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
        mul_132: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0000076294527394);  squeeze_53 = None
        mul_133: "f32[96]" = torch.ops.aten.mul.Tensor(mul_132, 0.1);  mul_132 = None
        mul_134: "f32[96]" = torch.ops.aten.mul.Tensor(primals_134, 0.9)
        add_118: "f32[96]" = torch.ops.aten.add.Tensor(mul_133, mul_134);  mul_133 = mul_134 = None
        unsqueeze_68: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_135, -1)
        unsqueeze_69: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_135: "f32[128, 96, 32, 32]" = torch.ops.aten.mul.Tensor(mul_129, unsqueeze_69);  mul_129 = unsqueeze_69 = None
        unsqueeze_70: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_136, -1)
        unsqueeze_71: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_119: "f32[128, 96, 32, 32]" = torch.ops.aten.add.Tensor(mul_135, unsqueeze_71);  mul_135 = unsqueeze_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_14: "f32[128, 96, 32, 32]" = torch.ops.aten.neg.default(add_119)
        exp_14: "f32[128, 96, 32, 32]" = torch.ops.aten.exp.default(neg_14);  neg_14 = None
        add_120: "f32[128, 96, 32, 32]" = torch.ops.aten.add.Tensor(exp_14, 1);  exp_14 = None
        div_14: "f32[128, 96, 32, 32]" = torch.ops.aten.div.Tensor(add_119, add_120);  add_119 = add_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:277 in forward, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
        cat: "f32[128, 192, 32, 32]" = torch.ops.aten.cat.default([add_92, div_14], 1);  div_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_19: "f32[128, 96, 32, 32]" = torch.ops.aten.convolution.default(cat, primals_137, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_121: "i64[]" = torch.ops.aten.add.Tensor(primals_138, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_23 = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
        getitem_60: "f32[1, 96, 1, 1]" = var_mean_23[0]
        getitem_61: "f32[1, 96, 1, 1]" = var_mean_23[1];  var_mean_23 = None
        add_122: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05)
        rsqrt_23: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        sub_23: "f32[128, 96, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_61)
        mul_136: "f32[128, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
        squeeze_54: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3])
        mul_137: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1);  squeeze_54 = None
        mul_138: "f32[96]" = torch.ops.aten.mul.Tensor(primals_139, 0.9)
        add_123: "f32[96]" = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
        squeeze_56: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
        mul_139: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0000076294527394);  squeeze_56 = None
        mul_140: "f32[96]" = torch.ops.aten.mul.Tensor(mul_139, 0.1);  mul_139 = None
        mul_141: "f32[96]" = torch.ops.aten.mul.Tensor(primals_140, 0.9)
        add_124: "f32[96]" = torch.ops.aten.add.Tensor(mul_140, mul_141);  mul_140 = mul_141 = None
        unsqueeze_72: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_141, -1)
        unsqueeze_73: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        mul_142: "f32[128, 96, 32, 32]" = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_73);  mul_136 = unsqueeze_73 = None
        unsqueeze_74: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_142, -1)
        unsqueeze_75: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        add_125: "f32[128, 96, 32, 32]" = torch.ops.aten.add.Tensor(mul_142, unsqueeze_75);  mul_142 = unsqueeze_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_15: "f32[128, 96, 32, 32]" = torch.ops.aten.neg.default(add_125)
        exp_15: "f32[128, 96, 32, 32]" = torch.ops.aten.exp.default(neg_15);  neg_15 = None
        add_126: "f32[128, 96, 32, 32]" = torch.ops.aten.add.Tensor(exp_15, 1);  exp_15 = None
        div_15: "f32[128, 96, 32, 32]" = torch.ops.aten.div.Tensor(add_125, add_126);  add_125 = add_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_20: "f32[128, 384, 32, 32]" = torch.ops.aten.convolution.default(div_15, primals_143, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_127: "i64[]" = torch.ops.aten.add.Tensor(primals_144, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_24 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
        getitem_62: "f32[1, 384, 1, 1]" = var_mean_24[0]
        getitem_63: "f32[1, 384, 1, 1]" = var_mean_24[1];  var_mean_24 = None
        add_128: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
        rsqrt_24: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
        sub_24: "f32[128, 384, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_63)
        mul_143: "f32[128, 384, 32, 32]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        squeeze_57: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3])
        mul_144: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1);  squeeze_57 = None
        mul_145: "f32[384]" = torch.ops.aten.mul.Tensor(primals_145, 0.9)
        add_129: "f32[384]" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
        squeeze_59: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
        mul_146: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0000076294527394);  squeeze_59 = None
        mul_147: "f32[384]" = torch.ops.aten.mul.Tensor(mul_146, 0.1);  mul_146 = None
        mul_148: "f32[384]" = torch.ops.aten.mul.Tensor(primals_146, 0.9)
        add_130: "f32[384]" = torch.ops.aten.add.Tensor(mul_147, mul_148);  mul_147 = mul_148 = None
        unsqueeze_76: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_147, -1)
        unsqueeze_77: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_149: "f32[128, 384, 32, 32]" = torch.ops.aten.mul.Tensor(mul_143, unsqueeze_77);  mul_143 = unsqueeze_77 = None
        unsqueeze_78: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_148, -1)
        unsqueeze_79: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_131: "f32[128, 384, 32, 32]" = torch.ops.aten.add.Tensor(mul_149, unsqueeze_79);  mul_149 = unsqueeze_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_16: "f32[128, 384, 32, 32]" = torch.ops.aten.neg.default(add_131)
        exp_16: "f32[128, 384, 32, 32]" = torch.ops.aten.exp.default(neg_16);  neg_16 = None
        add_132: "f32[128, 384, 32, 32]" = torch.ops.aten.add.Tensor(exp_16, 1);  exp_16 = None
        div_16: "f32[128, 384, 32, 32]" = torch.ops.aten.div.Tensor(add_131, add_132);  add_131 = add_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_21: "f32[128, 384, 16, 16]" = torch.ops.aten.convolution.default(div_16, primals_149, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 384)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_133: "i64[]" = torch.ops.aten.add.Tensor(primals_150, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_25 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
        getitem_64: "f32[1, 384, 1, 1]" = var_mean_25[0]
        getitem_65: "f32[1, 384, 1, 1]" = var_mean_25[1];  var_mean_25 = None
        add_134: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
        rsqrt_25: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        sub_25: "f32[128, 384, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_65)
        mul_150: "f32[128, 384, 16, 16]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        squeeze_60: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3])
        mul_151: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1);  squeeze_60 = None
        mul_152: "f32[384]" = torch.ops.aten.mul.Tensor(primals_151, 0.9)
        add_135: "f32[384]" = torch.ops.aten.add.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
        squeeze_62: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
        mul_153: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.000030518509476);  squeeze_62 = None
        mul_154: "f32[384]" = torch.ops.aten.mul.Tensor(mul_153, 0.1);  mul_153 = None
        mul_155: "f32[384]" = torch.ops.aten.mul.Tensor(primals_152, 0.9)
        add_136: "f32[384]" = torch.ops.aten.add.Tensor(mul_154, mul_155);  mul_154 = mul_155 = None
        unsqueeze_80: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_153, -1)
        unsqueeze_81: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        mul_156: "f32[128, 384, 16, 16]" = torch.ops.aten.mul.Tensor(mul_150, unsqueeze_81);  mul_150 = unsqueeze_81 = None
        unsqueeze_82: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_154, -1)
        unsqueeze_83: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        add_137: "f32[128, 384, 16, 16]" = torch.ops.aten.add.Tensor(mul_156, unsqueeze_83);  mul_156 = unsqueeze_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_17: "f32[128, 384, 16, 16]" = torch.ops.aten.neg.default(add_137)
        exp_17: "f32[128, 384, 16, 16]" = torch.ops.aten.exp.default(neg_17);  neg_17 = None
        add_138: "f32[128, 384, 16, 16]" = torch.ops.aten.add.Tensor(exp_17, 1);  exp_17 = None
        div_17: "f32[128, 384, 16, 16]" = torch.ops.aten.div.Tensor(add_137, add_138);  add_137 = add_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_22: "f32[128, 128, 16, 16]" = torch.ops.aten.convolution.default(div_17, primals_155, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_139: "i64[]" = torch.ops.aten.add.Tensor(primals_156, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_26 = torch.ops.aten.var_mean.correction(convolution_22, [0, 2, 3], correction = 0, keepdim = True)
        getitem_66: "f32[1, 128, 1, 1]" = var_mean_26[0]
        getitem_67: "f32[1, 128, 1, 1]" = var_mean_26[1];  var_mean_26 = None
        add_140: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05)
        rsqrt_26: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
        sub_26: "f32[128, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_22, getitem_67)
        mul_157: "f32[128, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
        squeeze_63: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
        squeeze_64: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
        mul_158: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
        mul_159: "f32[128]" = torch.ops.aten.mul.Tensor(primals_157, 0.9)
        add_141: "f32[128]" = torch.ops.aten.add.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None
        squeeze_65: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
        mul_160: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.000030518509476);  squeeze_65 = None
        mul_161: "f32[128]" = torch.ops.aten.mul.Tensor(mul_160, 0.1);  mul_160 = None
        mul_162: "f32[128]" = torch.ops.aten.mul.Tensor(primals_158, 0.9)
        add_142: "f32[128]" = torch.ops.aten.add.Tensor(mul_161, mul_162);  mul_161 = mul_162 = None
        unsqueeze_84: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_159, -1)
        unsqueeze_85: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_163: "f32[128, 128, 16, 16]" = torch.ops.aten.mul.Tensor(mul_157, unsqueeze_85);  mul_157 = unsqueeze_85 = None
        unsqueeze_86: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_160, -1);  primals_160 = None
        unsqueeze_87: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_143: "f32[128, 128, 16, 16]" = torch.ops.aten.add.Tensor(mul_163, unsqueeze_87);  mul_163 = unsqueeze_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_23: "f32[128, 128, 16, 16]" = torch.ops.aten.convolution.default(add_143, primals_161, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_144: "i64[]" = torch.ops.aten.add.Tensor(primals_162, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_27 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
        getitem_68: "f32[1, 128, 1, 1]" = var_mean_27[0]
        getitem_69: "f32[1, 128, 1, 1]" = var_mean_27[1];  var_mean_27 = None
        add_145: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05)
        rsqrt_27: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
        sub_27: "f32[128, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_23, getitem_69)
        mul_164: "f32[128, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
        squeeze_66: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3])
        mul_165: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1);  squeeze_66 = None
        mul_166: "f32[128]" = torch.ops.aten.mul.Tensor(primals_163, 0.9)
        add_146: "f32[128]" = torch.ops.aten.add.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
        squeeze_68: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
        mul_167: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.000030518509476);  squeeze_68 = None
        mul_168: "f32[128]" = torch.ops.aten.mul.Tensor(mul_167, 0.1);  mul_167 = None
        mul_169: "f32[128]" = torch.ops.aten.mul.Tensor(primals_164, 0.9)
        add_147: "f32[128]" = torch.ops.aten.add.Tensor(mul_168, mul_169);  mul_168 = mul_169 = None
        unsqueeze_88: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_165, -1)
        unsqueeze_89: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        mul_170: "f32[128, 128, 16, 16]" = torch.ops.aten.mul.Tensor(mul_164, unsqueeze_89);  mul_164 = unsqueeze_89 = None
        unsqueeze_90: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_166, -1)
        unsqueeze_91: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        add_148: "f32[128, 128, 16, 16]" = torch.ops.aten.add.Tensor(mul_170, unsqueeze_91);  mul_170 = unsqueeze_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_18: "f32[128, 128, 16, 16]" = torch.ops.aten.neg.default(add_148)
        exp_18: "f32[128, 128, 16, 16]" = torch.ops.aten.exp.default(neg_18);  neg_18 = None
        add_149: "f32[128, 128, 16, 16]" = torch.ops.aten.add.Tensor(exp_18, 1);  exp_18 = None
        div_18: "f32[128, 128, 16, 16]" = torch.ops.aten.div.Tensor(add_148, add_149);  add_148 = add_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:243 in forward, code: x = self.conv_1x1(x)
        convolution_24: "f32[128, 192, 16, 16]" = torch.ops.aten.convolution.default(div_18, primals_167, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:258 in forward, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
        clone_28: "f32[128, 192, 16, 16]" = torch.ops.aten.clone.default(convolution_24, memory_format = torch.contiguous_format);  convolution_24 = None
        view_26: "f32[196608, 2, 8, 2]" = torch.ops.aten.reshape.default(clone_28, [196608, 2, 8, 2]);  clone_28 = None
        permute_16: "f32[196608, 8, 2, 2]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:260 in forward, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
        clone_29: "f32[196608, 8, 2, 2]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        view_27: "f32[128, 192, 64, 4]" = torch.ops.aten.reshape.default(clone_29, [128, 192, 64, 4]);  clone_29 = None
        permute_17: "f32[128, 4, 64, 192]" = torch.ops.aten.permute.default(view_27, [0, 3, 2, 1]);  view_27 = None
        clone_30: "f32[128, 4, 64, 192]" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
        view_28: "f32[512, 64, 192]" = torch.ops.aten.reshape.default(clone_30, [512, 64, 192]);  clone_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        var_mean_28 = torch.ops.aten.var_mean.correction(view_28, [2], correction = 0, keepdim = True)
        getitem_70: "f32[512, 64, 1]" = var_mean_28[0]
        getitem_71: "f32[512, 64, 1]" = var_mean_28[1];  var_mean_28 = None
        add_150: "f32[512, 64, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
        rsqrt_28: "f32[512, 64, 1]" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
        sub_28: "f32[512, 64, 192]" = torch.ops.aten.sub.Tensor(view_28, getitem_71);  getitem_71 = None
        mul_171: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
        mul_172: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(mul_171, primals_168)
        add_151: "f32[512, 64, 192]" = torch.ops.aten.add.Tensor(mul_172, primals_169);  mul_172 = primals_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_29: "f32[32768, 192]" = torch.ops.aten.reshape.default(add_151, [32768, 192]);  add_151 = None
        permute_18: "f32[192, 576]" = torch.ops.aten.permute.default(primals_170, [1, 0])
        addmm_8: "f32[32768, 576]" = torch.ops.aten.addmm.default(primals_171, view_29, permute_18);  primals_171 = permute_18 = None
        view_30: "f32[512, 64, 576]" = torch.ops.aten.reshape.default(addmm_8, [512, 64, 576]);  addmm_8 = None
        view_31: "f32[512, 64, 3, 4, 48]" = torch.ops.aten.reshape.default(view_30, [512, 64, 3, 4, 48]);  view_30 = None
        permute_19: "f32[3, 512, 4, 64, 48]" = torch.ops.aten.permute.default(view_31, [2, 0, 3, 1, 4]);  view_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_2 = torch.ops.aten.unbind.int(permute_19);  permute_19 = None
        getitem_72: "f32[512, 4, 64, 48]" = unbind_2[0]
        getitem_73: "f32[512, 4, 64, 48]" = unbind_2[1]
        getitem_74: "f32[512, 4, 64, 48]" = unbind_2[2];  unbind_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_72, getitem_73, getitem_74, None, True)
        getitem_75: "f32[512, 4, 64, 48]" = _scaled_dot_product_efficient_attention_2[0]
        getitem_76: "f32[512, 4, 64]" = _scaled_dot_product_efficient_attention_2[1]
        getitem_77: "i64[]" = _scaled_dot_product_efficient_attention_2[2]
        getitem_78: "i64[]" = _scaled_dot_product_efficient_attention_2[3];  _scaled_dot_product_efficient_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_20: "f32[512, 64, 4, 48]" = torch.ops.aten.permute.default(getitem_75, [0, 2, 1, 3])
        view_32: "f32[512, 64, 192]" = torch.ops.aten.reshape.default(permute_20, [512, 64, 192]);  permute_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_33: "f32[32768, 192]" = torch.ops.aten.reshape.default(view_32, [32768, 192]);  view_32 = None
        permute_21: "f32[192, 192]" = torch.ops.aten.permute.default(primals_172, [1, 0])
        addmm_9: "f32[32768, 192]" = torch.ops.aten.addmm.default(primals_173, view_33, permute_21);  primals_173 = view_33 = permute_21 = None
        view_34: "f32[512, 64, 192]" = torch.ops.aten.reshape.default(addmm_9, [512, 64, 192]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_152: "f32[512, 64, 192]" = torch.ops.aten.add.Tensor(view_28, view_34);  view_28 = view_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_29 = torch.ops.aten.var_mean.correction(add_152, [2], correction = 0, keepdim = True)
        getitem_79: "f32[512, 64, 1]" = var_mean_29[0]
        getitem_80: "f32[512, 64, 1]" = var_mean_29[1];  var_mean_29 = None
        add_153: "f32[512, 64, 1]" = torch.ops.aten.add.Tensor(getitem_79, 1e-05);  getitem_79 = None
        rsqrt_29: "f32[512, 64, 1]" = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
        sub_29: "f32[512, 64, 192]" = torch.ops.aten.sub.Tensor(add_152, getitem_80);  getitem_80 = None
        mul_173: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
        mul_174: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(mul_173, primals_174)
        add_154: "f32[512, 64, 192]" = torch.ops.aten.add.Tensor(mul_174, primals_175);  mul_174 = primals_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_35: "f32[32768, 192]" = torch.ops.aten.reshape.default(add_154, [32768, 192]);  add_154 = None
        permute_22: "f32[192, 384]" = torch.ops.aten.permute.default(primals_176, [1, 0])
        addmm_10: "f32[32768, 384]" = torch.ops.aten.addmm.default(primals_177, view_35, permute_22);  primals_177 = permute_22 = None
        view_36: "f32[512, 64, 384]" = torch.ops.aten.reshape.default(addmm_10, [512, 64, 384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        neg_19: "f32[512, 64, 384]" = torch.ops.aten.neg.default(view_36)
        exp_19: "f32[512, 64, 384]" = torch.ops.aten.exp.default(neg_19);  neg_19 = None
        add_155: "f32[512, 64, 384]" = torch.ops.aten.add.Tensor(exp_19, 1);  exp_19 = None
        div_19: "f32[512, 64, 384]" = torch.ops.aten.div.Tensor(view_36, add_155);  view_36 = add_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_37: "f32[32768, 384]" = torch.ops.aten.reshape.default(div_19, [32768, 384]);  div_19 = None
        permute_23: "f32[384, 192]" = torch.ops.aten.permute.default(primals_178, [1, 0])
        addmm_11: "f32[32768, 192]" = torch.ops.aten.addmm.default(primals_179, view_37, permute_23);  primals_179 = permute_23 = None
        view_38: "f32[512, 64, 192]" = torch.ops.aten.reshape.default(addmm_11, [512, 64, 192]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_156: "f32[512, 64, 192]" = torch.ops.aten.add.Tensor(add_152, view_38);  add_152 = view_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        var_mean_30 = torch.ops.aten.var_mean.correction(add_156, [2], correction = 0, keepdim = True)
        getitem_81: "f32[512, 64, 1]" = var_mean_30[0]
        getitem_82: "f32[512, 64, 1]" = var_mean_30[1];  var_mean_30 = None
        add_157: "f32[512, 64, 1]" = torch.ops.aten.add.Tensor(getitem_81, 1e-05);  getitem_81 = None
        rsqrt_30: "f32[512, 64, 1]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
        sub_30: "f32[512, 64, 192]" = torch.ops.aten.sub.Tensor(add_156, getitem_82);  getitem_82 = None
        mul_175: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
        mul_176: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(mul_175, primals_180)
        add_158: "f32[512, 64, 192]" = torch.ops.aten.add.Tensor(mul_176, primals_181);  mul_176 = primals_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_39: "f32[32768, 192]" = torch.ops.aten.reshape.default(add_158, [32768, 192]);  add_158 = None
        permute_24: "f32[192, 576]" = torch.ops.aten.permute.default(primals_182, [1, 0])
        addmm_12: "f32[32768, 576]" = torch.ops.aten.addmm.default(primals_183, view_39, permute_24);  primals_183 = permute_24 = None
        view_40: "f32[512, 64, 576]" = torch.ops.aten.reshape.default(addmm_12, [512, 64, 576]);  addmm_12 = None
        view_41: "f32[512, 64, 3, 4, 48]" = torch.ops.aten.reshape.default(view_40, [512, 64, 3, 4, 48]);  view_40 = None
        permute_25: "f32[3, 512, 4, 64, 48]" = torch.ops.aten.permute.default(view_41, [2, 0, 3, 1, 4]);  view_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_3 = torch.ops.aten.unbind.int(permute_25);  permute_25 = None
        getitem_83: "f32[512, 4, 64, 48]" = unbind_3[0]
        getitem_84: "f32[512, 4, 64, 48]" = unbind_3[1]
        getitem_85: "f32[512, 4, 64, 48]" = unbind_3[2];  unbind_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_83, getitem_84, getitem_85, None, True)
        getitem_86: "f32[512, 4, 64, 48]" = _scaled_dot_product_efficient_attention_3[0]
        getitem_87: "f32[512, 4, 64]" = _scaled_dot_product_efficient_attention_3[1]
        getitem_88: "i64[]" = _scaled_dot_product_efficient_attention_3[2]
        getitem_89: "i64[]" = _scaled_dot_product_efficient_attention_3[3];  _scaled_dot_product_efficient_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_26: "f32[512, 64, 4, 48]" = torch.ops.aten.permute.default(getitem_86, [0, 2, 1, 3])
        view_42: "f32[512, 64, 192]" = torch.ops.aten.reshape.default(permute_26, [512, 64, 192]);  permute_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_43: "f32[32768, 192]" = torch.ops.aten.reshape.default(view_42, [32768, 192]);  view_42 = None
        permute_27: "f32[192, 192]" = torch.ops.aten.permute.default(primals_184, [1, 0])
        addmm_13: "f32[32768, 192]" = torch.ops.aten.addmm.default(primals_185, view_43, permute_27);  primals_185 = view_43 = permute_27 = None
        view_44: "f32[512, 64, 192]" = torch.ops.aten.reshape.default(addmm_13, [512, 64, 192]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_159: "f32[512, 64, 192]" = torch.ops.aten.add.Tensor(add_156, view_44);  add_156 = view_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_31 = torch.ops.aten.var_mean.correction(add_159, [2], correction = 0, keepdim = True)
        getitem_90: "f32[512, 64, 1]" = var_mean_31[0]
        getitem_91: "f32[512, 64, 1]" = var_mean_31[1];  var_mean_31 = None
        add_160: "f32[512, 64, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
        rsqrt_31: "f32[512, 64, 1]" = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
        sub_31: "f32[512, 64, 192]" = torch.ops.aten.sub.Tensor(add_159, getitem_91);  getitem_91 = None
        mul_177: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
        mul_178: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(mul_177, primals_186)
        add_161: "f32[512, 64, 192]" = torch.ops.aten.add.Tensor(mul_178, primals_187);  mul_178 = primals_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_45: "f32[32768, 192]" = torch.ops.aten.reshape.default(add_161, [32768, 192]);  add_161 = None
        permute_28: "f32[192, 384]" = torch.ops.aten.permute.default(primals_188, [1, 0])
        addmm_14: "f32[32768, 384]" = torch.ops.aten.addmm.default(primals_189, view_45, permute_28);  primals_189 = permute_28 = None
        view_46: "f32[512, 64, 384]" = torch.ops.aten.reshape.default(addmm_14, [512, 64, 384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        neg_20: "f32[512, 64, 384]" = torch.ops.aten.neg.default(view_46)
        exp_20: "f32[512, 64, 384]" = torch.ops.aten.exp.default(neg_20);  neg_20 = None
        add_162: "f32[512, 64, 384]" = torch.ops.aten.add.Tensor(exp_20, 1);  exp_20 = None
        div_20: "f32[512, 64, 384]" = torch.ops.aten.div.Tensor(view_46, add_162);  view_46 = add_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_47: "f32[32768, 384]" = torch.ops.aten.reshape.default(div_20, [32768, 384]);  div_20 = None
        permute_29: "f32[384, 192]" = torch.ops.aten.permute.default(primals_190, [1, 0])
        addmm_15: "f32[32768, 192]" = torch.ops.aten.addmm.default(primals_191, view_47, permute_29);  primals_191 = permute_29 = None
        view_48: "f32[512, 64, 192]" = torch.ops.aten.reshape.default(addmm_15, [512, 64, 192]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_163: "f32[512, 64, 192]" = torch.ops.aten.add.Tensor(add_159, view_48);  add_159 = view_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        var_mean_32 = torch.ops.aten.var_mean.correction(add_163, [2], correction = 0, keepdim = True)
        getitem_92: "f32[512, 64, 1]" = var_mean_32[0]
        getitem_93: "f32[512, 64, 1]" = var_mean_32[1];  var_mean_32 = None
        add_164: "f32[512, 64, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
        rsqrt_32: "f32[512, 64, 1]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
        sub_32: "f32[512, 64, 192]" = torch.ops.aten.sub.Tensor(add_163, getitem_93);  getitem_93 = None
        mul_179: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
        mul_180: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(mul_179, primals_192)
        add_165: "f32[512, 64, 192]" = torch.ops.aten.add.Tensor(mul_180, primals_193);  mul_180 = primals_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_49: "f32[32768, 192]" = torch.ops.aten.reshape.default(add_165, [32768, 192]);  add_165 = None
        permute_30: "f32[192, 576]" = torch.ops.aten.permute.default(primals_194, [1, 0])
        addmm_16: "f32[32768, 576]" = torch.ops.aten.addmm.default(primals_195, view_49, permute_30);  primals_195 = permute_30 = None
        view_50: "f32[512, 64, 576]" = torch.ops.aten.reshape.default(addmm_16, [512, 64, 576]);  addmm_16 = None
        view_51: "f32[512, 64, 3, 4, 48]" = torch.ops.aten.reshape.default(view_50, [512, 64, 3, 4, 48]);  view_50 = None
        permute_31: "f32[3, 512, 4, 64, 48]" = torch.ops.aten.permute.default(view_51, [2, 0, 3, 1, 4]);  view_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_4 = torch.ops.aten.unbind.int(permute_31);  permute_31 = None
        getitem_94: "f32[512, 4, 64, 48]" = unbind_4[0]
        getitem_95: "f32[512, 4, 64, 48]" = unbind_4[1]
        getitem_96: "f32[512, 4, 64, 48]" = unbind_4[2];  unbind_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_94, getitem_95, getitem_96, None, True)
        getitem_97: "f32[512, 4, 64, 48]" = _scaled_dot_product_efficient_attention_4[0]
        getitem_98: "f32[512, 4, 64]" = _scaled_dot_product_efficient_attention_4[1]
        getitem_99: "i64[]" = _scaled_dot_product_efficient_attention_4[2]
        getitem_100: "i64[]" = _scaled_dot_product_efficient_attention_4[3];  _scaled_dot_product_efficient_attention_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_32: "f32[512, 64, 4, 48]" = torch.ops.aten.permute.default(getitem_97, [0, 2, 1, 3])
        view_52: "f32[512, 64, 192]" = torch.ops.aten.reshape.default(permute_32, [512, 64, 192]);  permute_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_53: "f32[32768, 192]" = torch.ops.aten.reshape.default(view_52, [32768, 192]);  view_52 = None
        permute_33: "f32[192, 192]" = torch.ops.aten.permute.default(primals_196, [1, 0])
        addmm_17: "f32[32768, 192]" = torch.ops.aten.addmm.default(primals_197, view_53, permute_33);  primals_197 = view_53 = permute_33 = None
        view_54: "f32[512, 64, 192]" = torch.ops.aten.reshape.default(addmm_17, [512, 64, 192]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_166: "f32[512, 64, 192]" = torch.ops.aten.add.Tensor(add_163, view_54);  add_163 = view_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_33 = torch.ops.aten.var_mean.correction(add_166, [2], correction = 0, keepdim = True)
        getitem_101: "f32[512, 64, 1]" = var_mean_33[0]
        getitem_102: "f32[512, 64, 1]" = var_mean_33[1];  var_mean_33 = None
        add_167: "f32[512, 64, 1]" = torch.ops.aten.add.Tensor(getitem_101, 1e-05);  getitem_101 = None
        rsqrt_33: "f32[512, 64, 1]" = torch.ops.aten.rsqrt.default(add_167);  add_167 = None
        sub_33: "f32[512, 64, 192]" = torch.ops.aten.sub.Tensor(add_166, getitem_102);  getitem_102 = None
        mul_181: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
        mul_182: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(mul_181, primals_198)
        add_168: "f32[512, 64, 192]" = torch.ops.aten.add.Tensor(mul_182, primals_199);  mul_182 = primals_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_55: "f32[32768, 192]" = torch.ops.aten.reshape.default(add_168, [32768, 192]);  add_168 = None
        permute_34: "f32[192, 384]" = torch.ops.aten.permute.default(primals_200, [1, 0])
        addmm_18: "f32[32768, 384]" = torch.ops.aten.addmm.default(primals_201, view_55, permute_34);  primals_201 = permute_34 = None
        view_56: "f32[512, 64, 384]" = torch.ops.aten.reshape.default(addmm_18, [512, 64, 384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        neg_21: "f32[512, 64, 384]" = torch.ops.aten.neg.default(view_56)
        exp_21: "f32[512, 64, 384]" = torch.ops.aten.exp.default(neg_21);  neg_21 = None
        add_169: "f32[512, 64, 384]" = torch.ops.aten.add.Tensor(exp_21, 1);  exp_21 = None
        div_21: "f32[512, 64, 384]" = torch.ops.aten.div.Tensor(view_56, add_169);  view_56 = add_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_57: "f32[32768, 384]" = torch.ops.aten.reshape.default(div_21, [32768, 384]);  div_21 = None
        permute_35: "f32[384, 192]" = torch.ops.aten.permute.default(primals_202, [1, 0])
        addmm_19: "f32[32768, 192]" = torch.ops.aten.addmm.default(primals_203, view_57, permute_35);  primals_203 = permute_35 = None
        view_58: "f32[512, 64, 192]" = torch.ops.aten.reshape.default(addmm_19, [512, 64, 192]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_170: "f32[512, 64, 192]" = torch.ops.aten.add.Tensor(add_166, view_58);  add_166 = view_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        var_mean_34 = torch.ops.aten.var_mean.correction(add_170, [2], correction = 0, keepdim = True)
        getitem_103: "f32[512, 64, 1]" = var_mean_34[0]
        getitem_104: "f32[512, 64, 1]" = var_mean_34[1];  var_mean_34 = None
        add_171: "f32[512, 64, 1]" = torch.ops.aten.add.Tensor(getitem_103, 1e-05);  getitem_103 = None
        rsqrt_34: "f32[512, 64, 1]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
        sub_34: "f32[512, 64, 192]" = torch.ops.aten.sub.Tensor(add_170, getitem_104);  getitem_104 = None
        mul_183: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
        mul_184: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(mul_183, primals_204)
        add_172: "f32[512, 64, 192]" = torch.ops.aten.add.Tensor(mul_184, primals_205);  mul_184 = primals_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_59: "f32[32768, 192]" = torch.ops.aten.reshape.default(add_172, [32768, 192]);  add_172 = None
        permute_36: "f32[192, 576]" = torch.ops.aten.permute.default(primals_206, [1, 0])
        addmm_20: "f32[32768, 576]" = torch.ops.aten.addmm.default(primals_207, view_59, permute_36);  primals_207 = permute_36 = None
        view_60: "f32[512, 64, 576]" = torch.ops.aten.reshape.default(addmm_20, [512, 64, 576]);  addmm_20 = None
        view_61: "f32[512, 64, 3, 4, 48]" = torch.ops.aten.reshape.default(view_60, [512, 64, 3, 4, 48]);  view_60 = None
        permute_37: "f32[3, 512, 4, 64, 48]" = torch.ops.aten.permute.default(view_61, [2, 0, 3, 1, 4]);  view_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_5 = torch.ops.aten.unbind.int(permute_37);  permute_37 = None
        getitem_105: "f32[512, 4, 64, 48]" = unbind_5[0]
        getitem_106: "f32[512, 4, 64, 48]" = unbind_5[1]
        getitem_107: "f32[512, 4, 64, 48]" = unbind_5[2];  unbind_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_105, getitem_106, getitem_107, None, True)
        getitem_108: "f32[512, 4, 64, 48]" = _scaled_dot_product_efficient_attention_5[0]
        getitem_109: "f32[512, 4, 64]" = _scaled_dot_product_efficient_attention_5[1]
        getitem_110: "i64[]" = _scaled_dot_product_efficient_attention_5[2]
        getitem_111: "i64[]" = _scaled_dot_product_efficient_attention_5[3];  _scaled_dot_product_efficient_attention_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_38: "f32[512, 64, 4, 48]" = torch.ops.aten.permute.default(getitem_108, [0, 2, 1, 3])
        view_62: "f32[512, 64, 192]" = torch.ops.aten.reshape.default(permute_38, [512, 64, 192]);  permute_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_63: "f32[32768, 192]" = torch.ops.aten.reshape.default(view_62, [32768, 192]);  view_62 = None
        permute_39: "f32[192, 192]" = torch.ops.aten.permute.default(primals_208, [1, 0])
        addmm_21: "f32[32768, 192]" = torch.ops.aten.addmm.default(primals_209, view_63, permute_39);  primals_209 = view_63 = permute_39 = None
        view_64: "f32[512, 64, 192]" = torch.ops.aten.reshape.default(addmm_21, [512, 64, 192]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_173: "f32[512, 64, 192]" = torch.ops.aten.add.Tensor(add_170, view_64);  add_170 = view_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_35 = torch.ops.aten.var_mean.correction(add_173, [2], correction = 0, keepdim = True)
        getitem_112: "f32[512, 64, 1]" = var_mean_35[0]
        getitem_113: "f32[512, 64, 1]" = var_mean_35[1];  var_mean_35 = None
        add_174: "f32[512, 64, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
        rsqrt_35: "f32[512, 64, 1]" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        sub_35: "f32[512, 64, 192]" = torch.ops.aten.sub.Tensor(add_173, getitem_113);  getitem_113 = None
        mul_185: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
        mul_186: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(mul_185, primals_210)
        add_175: "f32[512, 64, 192]" = torch.ops.aten.add.Tensor(mul_186, primals_211);  mul_186 = primals_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_65: "f32[32768, 192]" = torch.ops.aten.reshape.default(add_175, [32768, 192]);  add_175 = None
        permute_40: "f32[192, 384]" = torch.ops.aten.permute.default(primals_212, [1, 0])
        addmm_22: "f32[32768, 384]" = torch.ops.aten.addmm.default(primals_213, view_65, permute_40);  primals_213 = permute_40 = None
        view_66: "f32[512, 64, 384]" = torch.ops.aten.reshape.default(addmm_22, [512, 64, 384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        neg_22: "f32[512, 64, 384]" = torch.ops.aten.neg.default(view_66)
        exp_22: "f32[512, 64, 384]" = torch.ops.aten.exp.default(neg_22);  neg_22 = None
        add_176: "f32[512, 64, 384]" = torch.ops.aten.add.Tensor(exp_22, 1);  exp_22 = None
        div_22: "f32[512, 64, 384]" = torch.ops.aten.div.Tensor(view_66, add_176);  view_66 = add_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_67: "f32[32768, 384]" = torch.ops.aten.reshape.default(div_22, [32768, 384]);  div_22 = None
        permute_41: "f32[384, 192]" = torch.ops.aten.permute.default(primals_214, [1, 0])
        addmm_23: "f32[32768, 192]" = torch.ops.aten.addmm.default(primals_215, view_67, permute_41);  primals_215 = permute_41 = None
        view_68: "f32[512, 64, 192]" = torch.ops.aten.reshape.default(addmm_23, [512, 64, 192]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_177: "f32[512, 64, 192]" = torch.ops.aten.add.Tensor(add_173, view_68);  add_173 = view_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:264 in forward, code: x = self.norm(x)
        var_mean_36 = torch.ops.aten.var_mean.correction(add_177, [2], correction = 0, keepdim = True)
        getitem_114: "f32[512, 64, 1]" = var_mean_36[0]
        getitem_115: "f32[512, 64, 1]" = var_mean_36[1];  var_mean_36 = None
        add_178: "f32[512, 64, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-05);  getitem_114 = None
        rsqrt_36: "f32[512, 64, 1]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
        sub_36: "f32[512, 64, 192]" = torch.ops.aten.sub.Tensor(add_177, getitem_115);  add_177 = getitem_115 = None
        mul_187: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
        mul_188: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(mul_187, primals_216)
        add_179: "f32[512, 64, 192]" = torch.ops.aten.add.Tensor(mul_188, primals_217);  mul_188 = primals_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:268 in forward, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
        view_69: "f32[128, 4, 64, 192]" = torch.ops.aten.reshape.default(add_179, [128, 4, 64, -1]);  add_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:269 in forward, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
        permute_42: "f32[128, 192, 64, 4]" = torch.ops.aten.permute.default(view_69, [0, 3, 2, 1]);  view_69 = None
        clone_43: "f32[128, 192, 64, 4]" = torch.ops.aten.clone.default(permute_42, memory_format = torch.contiguous_format);  permute_42 = None
        view_70: "f32[196608, 8, 2, 2]" = torch.ops.aten.reshape.default(clone_43, [196608, 8, 2, 2]);  clone_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:271 in forward, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
        permute_43: "f32[196608, 2, 8, 2]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
        clone_44: "f32[196608, 2, 8, 2]" = torch.ops.aten.clone.default(permute_43, memory_format = torch.contiguous_format);  permute_43 = None
        view_71: "f32[128, 192, 16, 16]" = torch.ops.aten.reshape.default(clone_44, [128, 192, 16, 16]);  clone_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_25: "f32[128, 128, 16, 16]" = torch.ops.aten.convolution.default(view_71, primals_218, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_180: "i64[]" = torch.ops.aten.add.Tensor(primals_219, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_37 = torch.ops.aten.var_mean.correction(convolution_25, [0, 2, 3], correction = 0, keepdim = True)
        getitem_116: "f32[1, 128, 1, 1]" = var_mean_37[0]
        getitem_117: "f32[1, 128, 1, 1]" = var_mean_37[1];  var_mean_37 = None
        add_181: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-05)
        rsqrt_37: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
        sub_37: "f32[128, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_117)
        mul_189: "f32[128, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
        squeeze_69: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_117, [0, 2, 3])
        mul_190: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1);  squeeze_69 = None
        mul_191: "f32[128]" = torch.ops.aten.mul.Tensor(primals_220, 0.9)
        add_182: "f32[128]" = torch.ops.aten.add.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
        squeeze_71: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_116, [0, 2, 3]);  getitem_116 = None
        mul_192: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.000030518509476);  squeeze_71 = None
        mul_193: "f32[128]" = torch.ops.aten.mul.Tensor(mul_192, 0.1);  mul_192 = None
        mul_194: "f32[128]" = torch.ops.aten.mul.Tensor(primals_221, 0.9)
        add_183: "f32[128]" = torch.ops.aten.add.Tensor(mul_193, mul_194);  mul_193 = mul_194 = None
        unsqueeze_92: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_222, -1)
        unsqueeze_93: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        mul_195: "f32[128, 128, 16, 16]" = torch.ops.aten.mul.Tensor(mul_189, unsqueeze_93);  mul_189 = unsqueeze_93 = None
        unsqueeze_94: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_223, -1)
        unsqueeze_95: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        add_184: "f32[128, 128, 16, 16]" = torch.ops.aten.add.Tensor(mul_195, unsqueeze_95);  mul_195 = unsqueeze_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_23: "f32[128, 128, 16, 16]" = torch.ops.aten.neg.default(add_184)
        exp_23: "f32[128, 128, 16, 16]" = torch.ops.aten.exp.default(neg_23);  neg_23 = None
        add_185: "f32[128, 128, 16, 16]" = torch.ops.aten.add.Tensor(exp_23, 1);  exp_23 = None
        div_23: "f32[128, 128, 16, 16]" = torch.ops.aten.div.Tensor(add_184, add_185);  add_184 = add_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:277 in forward, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
        cat_1: "f32[128, 256, 16, 16]" = torch.ops.aten.cat.default([add_143, div_23], 1);  div_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_26: "f32[128, 128, 16, 16]" = torch.ops.aten.convolution.default(cat_1, primals_224, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_186: "i64[]" = torch.ops.aten.add.Tensor(primals_225, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_38 = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
        getitem_118: "f32[1, 128, 1, 1]" = var_mean_38[0]
        getitem_119: "f32[1, 128, 1, 1]" = var_mean_38[1];  var_mean_38 = None
        add_187: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05)
        rsqrt_38: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_187);  add_187 = None
        sub_38: "f32[128, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_119)
        mul_196: "f32[128, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
        squeeze_72: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3])
        mul_197: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1);  squeeze_72 = None
        mul_198: "f32[128]" = torch.ops.aten.mul.Tensor(primals_226, 0.9)
        add_188: "f32[128]" = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
        squeeze_74: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_118, [0, 2, 3]);  getitem_118 = None
        mul_199: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.000030518509476);  squeeze_74 = None
        mul_200: "f32[128]" = torch.ops.aten.mul.Tensor(mul_199, 0.1);  mul_199 = None
        mul_201: "f32[128]" = torch.ops.aten.mul.Tensor(primals_227, 0.9)
        add_189: "f32[128]" = torch.ops.aten.add.Tensor(mul_200, mul_201);  mul_200 = mul_201 = None
        unsqueeze_96: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_228, -1)
        unsqueeze_97: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        mul_202: "f32[128, 128, 16, 16]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_97);  mul_196 = unsqueeze_97 = None
        unsqueeze_98: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_229, -1)
        unsqueeze_99: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        add_190: "f32[128, 128, 16, 16]" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_99);  mul_202 = unsqueeze_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_24: "f32[128, 128, 16, 16]" = torch.ops.aten.neg.default(add_190)
        exp_24: "f32[128, 128, 16, 16]" = torch.ops.aten.exp.default(neg_24);  neg_24 = None
        add_191: "f32[128, 128, 16, 16]" = torch.ops.aten.add.Tensor(exp_24, 1);  exp_24 = None
        div_24: "f32[128, 128, 16, 16]" = torch.ops.aten.div.Tensor(add_190, add_191);  add_190 = add_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_27: "f32[128, 512, 16, 16]" = torch.ops.aten.convolution.default(div_24, primals_230, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_192: "i64[]" = torch.ops.aten.add.Tensor(primals_231, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_39 = torch.ops.aten.var_mean.correction(convolution_27, [0, 2, 3], correction = 0, keepdim = True)
        getitem_120: "f32[1, 512, 1, 1]" = var_mean_39[0]
        getitem_121: "f32[1, 512, 1, 1]" = var_mean_39[1];  var_mean_39 = None
        add_193: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-05)
        rsqrt_39: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_193);  add_193 = None
        sub_39: "f32[128, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_27, getitem_121)
        mul_203: "f32[128, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
        squeeze_75: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3])
        mul_204: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1);  squeeze_75 = None
        mul_205: "f32[512]" = torch.ops.aten.mul.Tensor(primals_232, 0.9)
        add_194: "f32[512]" = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None
        squeeze_77: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_120, [0, 2, 3]);  getitem_120 = None
        mul_206: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.000030518509476);  squeeze_77 = None
        mul_207: "f32[512]" = torch.ops.aten.mul.Tensor(mul_206, 0.1);  mul_206 = None
        mul_208: "f32[512]" = torch.ops.aten.mul.Tensor(primals_233, 0.9)
        add_195: "f32[512]" = torch.ops.aten.add.Tensor(mul_207, mul_208);  mul_207 = mul_208 = None
        unsqueeze_100: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_234, -1)
        unsqueeze_101: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_209: "f32[128, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_101);  mul_203 = unsqueeze_101 = None
        unsqueeze_102: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_235, -1)
        unsqueeze_103: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_196: "f32[128, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_103);  mul_209 = unsqueeze_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_25: "f32[128, 512, 16, 16]" = torch.ops.aten.neg.default(add_196)
        exp_25: "f32[128, 512, 16, 16]" = torch.ops.aten.exp.default(neg_25);  neg_25 = None
        add_197: "f32[128, 512, 16, 16]" = torch.ops.aten.add.Tensor(exp_25, 1);  exp_25 = None
        div_25: "f32[128, 512, 16, 16]" = torch.ops.aten.div.Tensor(add_196, add_197);  add_196 = add_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_28: "f32[128, 512, 8, 8]" = torch.ops.aten.convolution.default(div_25, primals_236, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 512)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_198: "i64[]" = torch.ops.aten.add.Tensor(primals_237, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_40 = torch.ops.aten.var_mean.correction(convolution_28, [0, 2, 3], correction = 0, keepdim = True)
        getitem_122: "f32[1, 512, 1, 1]" = var_mean_40[0]
        getitem_123: "f32[1, 512, 1, 1]" = var_mean_40[1];  var_mean_40 = None
        add_199: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-05)
        rsqrt_40: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
        sub_40: "f32[128, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_28, getitem_123)
        mul_210: "f32[128, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
        squeeze_78: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_123, [0, 2, 3])
        mul_211: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1);  squeeze_78 = None
        mul_212: "f32[512]" = torch.ops.aten.mul.Tensor(primals_238, 0.9)
        add_200: "f32[512]" = torch.ops.aten.add.Tensor(mul_211, mul_212);  mul_211 = mul_212 = None
        squeeze_80: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_122, [0, 2, 3]);  getitem_122 = None
        mul_213: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0001220852154804);  squeeze_80 = None
        mul_214: "f32[512]" = torch.ops.aten.mul.Tensor(mul_213, 0.1);  mul_213 = None
        mul_215: "f32[512]" = torch.ops.aten.mul.Tensor(primals_239, 0.9)
        add_201: "f32[512]" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
        unsqueeze_104: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_240, -1)
        unsqueeze_105: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        mul_216: "f32[128, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_105);  mul_210 = unsqueeze_105 = None
        unsqueeze_106: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_241, -1)
        unsqueeze_107: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
        add_202: "f32[128, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_216, unsqueeze_107);  mul_216 = unsqueeze_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_26: "f32[128, 512, 8, 8]" = torch.ops.aten.neg.default(add_202)
        exp_26: "f32[128, 512, 8, 8]" = torch.ops.aten.exp.default(neg_26);  neg_26 = None
        add_203: "f32[128, 512, 8, 8]" = torch.ops.aten.add.Tensor(exp_26, 1);  exp_26 = None
        div_26: "f32[128, 512, 8, 8]" = torch.ops.aten.div.Tensor(add_202, add_203);  add_202 = add_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_29: "f32[128, 160, 8, 8]" = torch.ops.aten.convolution.default(div_26, primals_242, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_204: "i64[]" = torch.ops.aten.add.Tensor(primals_243, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_41 = torch.ops.aten.var_mean.correction(convolution_29, [0, 2, 3], correction = 0, keepdim = True)
        getitem_124: "f32[1, 160, 1, 1]" = var_mean_41[0]
        getitem_125: "f32[1, 160, 1, 1]" = var_mean_41[1];  var_mean_41 = None
        add_205: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-05)
        rsqrt_41: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_205);  add_205 = None
        sub_41: "f32[128, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_29, getitem_125)
        mul_217: "f32[128, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
        squeeze_81: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_125, [0, 2, 3]);  getitem_125 = None
        squeeze_82: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
        mul_218: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
        mul_219: "f32[160]" = torch.ops.aten.mul.Tensor(primals_244, 0.9)
        add_206: "f32[160]" = torch.ops.aten.add.Tensor(mul_218, mul_219);  mul_218 = mul_219 = None
        squeeze_83: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_124, [0, 2, 3]);  getitem_124 = None
        mul_220: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0001220852154804);  squeeze_83 = None
        mul_221: "f32[160]" = torch.ops.aten.mul.Tensor(mul_220, 0.1);  mul_220 = None
        mul_222: "f32[160]" = torch.ops.aten.mul.Tensor(primals_245, 0.9)
        add_207: "f32[160]" = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
        unsqueeze_108: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_246, -1)
        unsqueeze_109: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
        mul_223: "f32[128, 160, 8, 8]" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_109);  mul_217 = unsqueeze_109 = None
        unsqueeze_110: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_247, -1);  primals_247 = None
        unsqueeze_111: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
        add_208: "f32[128, 160, 8, 8]" = torch.ops.aten.add.Tensor(mul_223, unsqueeze_111);  mul_223 = unsqueeze_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_30: "f32[128, 160, 8, 8]" = torch.ops.aten.convolution.default(add_208, primals_248, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_209: "i64[]" = torch.ops.aten.add.Tensor(primals_249, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_42 = torch.ops.aten.var_mean.correction(convolution_30, [0, 2, 3], correction = 0, keepdim = True)
        getitem_126: "f32[1, 160, 1, 1]" = var_mean_42[0]
        getitem_127: "f32[1, 160, 1, 1]" = var_mean_42[1];  var_mean_42 = None
        add_210: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-05)
        rsqrt_42: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_210);  add_210 = None
        sub_42: "f32[128, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_30, getitem_127)
        mul_224: "f32[128, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
        squeeze_84: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_127, [0, 2, 3])
        mul_225: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1);  squeeze_84 = None
        mul_226: "f32[160]" = torch.ops.aten.mul.Tensor(primals_250, 0.9)
        add_211: "f32[160]" = torch.ops.aten.add.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
        squeeze_86: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_126, [0, 2, 3]);  getitem_126 = None
        mul_227: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0001220852154804);  squeeze_86 = None
        mul_228: "f32[160]" = torch.ops.aten.mul.Tensor(mul_227, 0.1);  mul_227 = None
        mul_229: "f32[160]" = torch.ops.aten.mul.Tensor(primals_251, 0.9)
        add_212: "f32[160]" = torch.ops.aten.add.Tensor(mul_228, mul_229);  mul_228 = mul_229 = None
        unsqueeze_112: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_252, -1)
        unsqueeze_113: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        mul_230: "f32[128, 160, 8, 8]" = torch.ops.aten.mul.Tensor(mul_224, unsqueeze_113);  mul_224 = unsqueeze_113 = None
        unsqueeze_114: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_253, -1)
        unsqueeze_115: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        add_213: "f32[128, 160, 8, 8]" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_115);  mul_230 = unsqueeze_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_27: "f32[128, 160, 8, 8]" = torch.ops.aten.neg.default(add_213)
        exp_27: "f32[128, 160, 8, 8]" = torch.ops.aten.exp.default(neg_27);  neg_27 = None
        add_214: "f32[128, 160, 8, 8]" = torch.ops.aten.add.Tensor(exp_27, 1);  exp_27 = None
        div_27: "f32[128, 160, 8, 8]" = torch.ops.aten.div.Tensor(add_213, add_214);  add_213 = add_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:243 in forward, code: x = self.conv_1x1(x)
        convolution_31: "f32[128, 240, 8, 8]" = torch.ops.aten.convolution.default(div_27, primals_254, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:258 in forward, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
        clone_50: "f32[128, 240, 8, 8]" = torch.ops.aten.clone.default(convolution_31, memory_format = torch.contiguous_format);  convolution_31 = None
        view_72: "f32[122880, 2, 4, 2]" = torch.ops.aten.reshape.default(clone_50, [122880, 2, 4, 2]);  clone_50 = None
        permute_44: "f32[122880, 4, 2, 2]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:260 in forward, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
        clone_51: "f32[122880, 4, 2, 2]" = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
        view_73: "f32[128, 240, 16, 4]" = torch.ops.aten.reshape.default(clone_51, [128, 240, 16, 4]);  clone_51 = None
        permute_45: "f32[128, 4, 16, 240]" = torch.ops.aten.permute.default(view_73, [0, 3, 2, 1]);  view_73 = None
        clone_52: "f32[128, 4, 16, 240]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
        view_74: "f32[512, 16, 240]" = torch.ops.aten.reshape.default(clone_52, [512, 16, 240]);  clone_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        var_mean_43 = torch.ops.aten.var_mean.correction(view_74, [2], correction = 0, keepdim = True)
        getitem_128: "f32[512, 16, 1]" = var_mean_43[0]
        getitem_129: "f32[512, 16, 1]" = var_mean_43[1];  var_mean_43 = None
        add_215: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-05);  getitem_128 = None
        rsqrt_43: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_215);  add_215 = None
        sub_43: "f32[512, 16, 240]" = torch.ops.aten.sub.Tensor(view_74, getitem_129);  getitem_129 = None
        mul_231: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
        mul_232: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(mul_231, primals_255)
        add_216: "f32[512, 16, 240]" = torch.ops.aten.add.Tensor(mul_232, primals_256);  mul_232 = primals_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_75: "f32[8192, 240]" = torch.ops.aten.reshape.default(add_216, [8192, 240]);  add_216 = None
        permute_46: "f32[240, 720]" = torch.ops.aten.permute.default(primals_257, [1, 0])
        addmm_24: "f32[8192, 720]" = torch.ops.aten.addmm.default(primals_258, view_75, permute_46);  primals_258 = permute_46 = None
        view_76: "f32[512, 16, 720]" = torch.ops.aten.reshape.default(addmm_24, [512, 16, 720]);  addmm_24 = None
        view_77: "f32[512, 16, 3, 4, 60]" = torch.ops.aten.reshape.default(view_76, [512, 16, 3, 4, 60]);  view_76 = None
        permute_47: "f32[3, 512, 4, 16, 60]" = torch.ops.aten.permute.default(view_77, [2, 0, 3, 1, 4]);  view_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_6 = torch.ops.aten.unbind.int(permute_47);  permute_47 = None
        getitem_130: "f32[512, 4, 16, 60]" = unbind_6[0]
        getitem_131: "f32[512, 4, 16, 60]" = unbind_6[1]
        getitem_132: "f32[512, 4, 16, 60]" = unbind_6[2];  unbind_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_130, getitem_131, getitem_132, None, True)
        getitem_133: "f32[512, 4, 16, 60]" = _scaled_dot_product_efficient_attention_6[0]
        getitem_134: "f32[512, 4, 32]" = _scaled_dot_product_efficient_attention_6[1]
        getitem_135: "i64[]" = _scaled_dot_product_efficient_attention_6[2]
        getitem_136: "i64[]" = _scaled_dot_product_efficient_attention_6[3];  _scaled_dot_product_efficient_attention_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_48: "f32[512, 16, 4, 60]" = torch.ops.aten.permute.default(getitem_133, [0, 2, 1, 3])
        view_78: "f32[512, 16, 240]" = torch.ops.aten.reshape.default(permute_48, [512, 16, 240]);  permute_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_79: "f32[8192, 240]" = torch.ops.aten.reshape.default(view_78, [8192, 240]);  view_78 = None
        permute_49: "f32[240, 240]" = torch.ops.aten.permute.default(primals_259, [1, 0])
        addmm_25: "f32[8192, 240]" = torch.ops.aten.addmm.default(primals_260, view_79, permute_49);  primals_260 = view_79 = permute_49 = None
        view_80: "f32[512, 16, 240]" = torch.ops.aten.reshape.default(addmm_25, [512, 16, 240]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_217: "f32[512, 16, 240]" = torch.ops.aten.add.Tensor(view_74, view_80);  view_74 = view_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_44 = torch.ops.aten.var_mean.correction(add_217, [2], correction = 0, keepdim = True)
        getitem_137: "f32[512, 16, 1]" = var_mean_44[0]
        getitem_138: "f32[512, 16, 1]" = var_mean_44[1];  var_mean_44 = None
        add_218: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_137, 1e-05);  getitem_137 = None
        rsqrt_44: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
        sub_44: "f32[512, 16, 240]" = torch.ops.aten.sub.Tensor(add_217, getitem_138);  getitem_138 = None
        mul_233: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
        mul_234: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(mul_233, primals_261)
        add_219: "f32[512, 16, 240]" = torch.ops.aten.add.Tensor(mul_234, primals_262);  mul_234 = primals_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_81: "f32[8192, 240]" = torch.ops.aten.reshape.default(add_219, [8192, 240]);  add_219 = None
        permute_50: "f32[240, 480]" = torch.ops.aten.permute.default(primals_263, [1, 0])
        addmm_26: "f32[8192, 480]" = torch.ops.aten.addmm.default(primals_264, view_81, permute_50);  primals_264 = permute_50 = None
        view_82: "f32[512, 16, 480]" = torch.ops.aten.reshape.default(addmm_26, [512, 16, 480])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        neg_28: "f32[512, 16, 480]" = torch.ops.aten.neg.default(view_82)
        exp_28: "f32[512, 16, 480]" = torch.ops.aten.exp.default(neg_28);  neg_28 = None
        add_220: "f32[512, 16, 480]" = torch.ops.aten.add.Tensor(exp_28, 1);  exp_28 = None
        div_28: "f32[512, 16, 480]" = torch.ops.aten.div.Tensor(view_82, add_220);  view_82 = add_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_83: "f32[8192, 480]" = torch.ops.aten.reshape.default(div_28, [8192, 480]);  div_28 = None
        permute_51: "f32[480, 240]" = torch.ops.aten.permute.default(primals_265, [1, 0])
        addmm_27: "f32[8192, 240]" = torch.ops.aten.addmm.default(primals_266, view_83, permute_51);  primals_266 = permute_51 = None
        view_84: "f32[512, 16, 240]" = torch.ops.aten.reshape.default(addmm_27, [512, 16, 240]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_221: "f32[512, 16, 240]" = torch.ops.aten.add.Tensor(add_217, view_84);  add_217 = view_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        var_mean_45 = torch.ops.aten.var_mean.correction(add_221, [2], correction = 0, keepdim = True)
        getitem_139: "f32[512, 16, 1]" = var_mean_45[0]
        getitem_140: "f32[512, 16, 1]" = var_mean_45[1];  var_mean_45 = None
        add_222: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_139, 1e-05);  getitem_139 = None
        rsqrt_45: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_222);  add_222 = None
        sub_45: "f32[512, 16, 240]" = torch.ops.aten.sub.Tensor(add_221, getitem_140);  getitem_140 = None
        mul_235: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
        mul_236: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(mul_235, primals_267)
        add_223: "f32[512, 16, 240]" = torch.ops.aten.add.Tensor(mul_236, primals_268);  mul_236 = primals_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_85: "f32[8192, 240]" = torch.ops.aten.reshape.default(add_223, [8192, 240]);  add_223 = None
        permute_52: "f32[240, 720]" = torch.ops.aten.permute.default(primals_269, [1, 0])
        addmm_28: "f32[8192, 720]" = torch.ops.aten.addmm.default(primals_270, view_85, permute_52);  primals_270 = permute_52 = None
        view_86: "f32[512, 16, 720]" = torch.ops.aten.reshape.default(addmm_28, [512, 16, 720]);  addmm_28 = None
        view_87: "f32[512, 16, 3, 4, 60]" = torch.ops.aten.reshape.default(view_86, [512, 16, 3, 4, 60]);  view_86 = None
        permute_53: "f32[3, 512, 4, 16, 60]" = torch.ops.aten.permute.default(view_87, [2, 0, 3, 1, 4]);  view_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_7 = torch.ops.aten.unbind.int(permute_53);  permute_53 = None
        getitem_141: "f32[512, 4, 16, 60]" = unbind_7[0]
        getitem_142: "f32[512, 4, 16, 60]" = unbind_7[1]
        getitem_143: "f32[512, 4, 16, 60]" = unbind_7[2];  unbind_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_141, getitem_142, getitem_143, None, True)
        getitem_144: "f32[512, 4, 16, 60]" = _scaled_dot_product_efficient_attention_7[0]
        getitem_145: "f32[512, 4, 32]" = _scaled_dot_product_efficient_attention_7[1]
        getitem_146: "i64[]" = _scaled_dot_product_efficient_attention_7[2]
        getitem_147: "i64[]" = _scaled_dot_product_efficient_attention_7[3];  _scaled_dot_product_efficient_attention_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_54: "f32[512, 16, 4, 60]" = torch.ops.aten.permute.default(getitem_144, [0, 2, 1, 3])
        view_88: "f32[512, 16, 240]" = torch.ops.aten.reshape.default(permute_54, [512, 16, 240]);  permute_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_89: "f32[8192, 240]" = torch.ops.aten.reshape.default(view_88, [8192, 240]);  view_88 = None
        permute_55: "f32[240, 240]" = torch.ops.aten.permute.default(primals_271, [1, 0])
        addmm_29: "f32[8192, 240]" = torch.ops.aten.addmm.default(primals_272, view_89, permute_55);  primals_272 = view_89 = permute_55 = None
        view_90: "f32[512, 16, 240]" = torch.ops.aten.reshape.default(addmm_29, [512, 16, 240]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_224: "f32[512, 16, 240]" = torch.ops.aten.add.Tensor(add_221, view_90);  add_221 = view_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_46 = torch.ops.aten.var_mean.correction(add_224, [2], correction = 0, keepdim = True)
        getitem_148: "f32[512, 16, 1]" = var_mean_46[0]
        getitem_149: "f32[512, 16, 1]" = var_mean_46[1];  var_mean_46 = None
        add_225: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_148, 1e-05);  getitem_148 = None
        rsqrt_46: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_225);  add_225 = None
        sub_46: "f32[512, 16, 240]" = torch.ops.aten.sub.Tensor(add_224, getitem_149);  getitem_149 = None
        mul_237: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
        mul_238: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(mul_237, primals_273)
        add_226: "f32[512, 16, 240]" = torch.ops.aten.add.Tensor(mul_238, primals_274);  mul_238 = primals_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_91: "f32[8192, 240]" = torch.ops.aten.reshape.default(add_226, [8192, 240]);  add_226 = None
        permute_56: "f32[240, 480]" = torch.ops.aten.permute.default(primals_275, [1, 0])
        addmm_30: "f32[8192, 480]" = torch.ops.aten.addmm.default(primals_276, view_91, permute_56);  primals_276 = permute_56 = None
        view_92: "f32[512, 16, 480]" = torch.ops.aten.reshape.default(addmm_30, [512, 16, 480])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        neg_29: "f32[512, 16, 480]" = torch.ops.aten.neg.default(view_92)
        exp_29: "f32[512, 16, 480]" = torch.ops.aten.exp.default(neg_29);  neg_29 = None
        add_227: "f32[512, 16, 480]" = torch.ops.aten.add.Tensor(exp_29, 1);  exp_29 = None
        div_29: "f32[512, 16, 480]" = torch.ops.aten.div.Tensor(view_92, add_227);  view_92 = add_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_93: "f32[8192, 480]" = torch.ops.aten.reshape.default(div_29, [8192, 480]);  div_29 = None
        permute_57: "f32[480, 240]" = torch.ops.aten.permute.default(primals_277, [1, 0])
        addmm_31: "f32[8192, 240]" = torch.ops.aten.addmm.default(primals_278, view_93, permute_57);  primals_278 = permute_57 = None
        view_94: "f32[512, 16, 240]" = torch.ops.aten.reshape.default(addmm_31, [512, 16, 240]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_228: "f32[512, 16, 240]" = torch.ops.aten.add.Tensor(add_224, view_94);  add_224 = view_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        var_mean_47 = torch.ops.aten.var_mean.correction(add_228, [2], correction = 0, keepdim = True)
        getitem_150: "f32[512, 16, 1]" = var_mean_47[0]
        getitem_151: "f32[512, 16, 1]" = var_mean_47[1];  var_mean_47 = None
        add_229: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_150, 1e-05);  getitem_150 = None
        rsqrt_47: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
        sub_47: "f32[512, 16, 240]" = torch.ops.aten.sub.Tensor(add_228, getitem_151);  getitem_151 = None
        mul_239: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
        mul_240: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(mul_239, primals_279)
        add_230: "f32[512, 16, 240]" = torch.ops.aten.add.Tensor(mul_240, primals_280);  mul_240 = primals_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_95: "f32[8192, 240]" = torch.ops.aten.reshape.default(add_230, [8192, 240]);  add_230 = None
        permute_58: "f32[240, 720]" = torch.ops.aten.permute.default(primals_281, [1, 0])
        addmm_32: "f32[8192, 720]" = torch.ops.aten.addmm.default(primals_282, view_95, permute_58);  primals_282 = permute_58 = None
        view_96: "f32[512, 16, 720]" = torch.ops.aten.reshape.default(addmm_32, [512, 16, 720]);  addmm_32 = None
        view_97: "f32[512, 16, 3, 4, 60]" = torch.ops.aten.reshape.default(view_96, [512, 16, 3, 4, 60]);  view_96 = None
        permute_59: "f32[3, 512, 4, 16, 60]" = torch.ops.aten.permute.default(view_97, [2, 0, 3, 1, 4]);  view_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_8 = torch.ops.aten.unbind.int(permute_59);  permute_59 = None
        getitem_152: "f32[512, 4, 16, 60]" = unbind_8[0]
        getitem_153: "f32[512, 4, 16, 60]" = unbind_8[1]
        getitem_154: "f32[512, 4, 16, 60]" = unbind_8[2];  unbind_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_152, getitem_153, getitem_154, None, True)
        getitem_155: "f32[512, 4, 16, 60]" = _scaled_dot_product_efficient_attention_8[0]
        getitem_156: "f32[512, 4, 32]" = _scaled_dot_product_efficient_attention_8[1]
        getitem_157: "i64[]" = _scaled_dot_product_efficient_attention_8[2]
        getitem_158: "i64[]" = _scaled_dot_product_efficient_attention_8[3];  _scaled_dot_product_efficient_attention_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_60: "f32[512, 16, 4, 60]" = torch.ops.aten.permute.default(getitem_155, [0, 2, 1, 3])
        view_98: "f32[512, 16, 240]" = torch.ops.aten.reshape.default(permute_60, [512, 16, 240]);  permute_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_99: "f32[8192, 240]" = torch.ops.aten.reshape.default(view_98, [8192, 240]);  view_98 = None
        permute_61: "f32[240, 240]" = torch.ops.aten.permute.default(primals_283, [1, 0])
        addmm_33: "f32[8192, 240]" = torch.ops.aten.addmm.default(primals_284, view_99, permute_61);  primals_284 = view_99 = permute_61 = None
        view_100: "f32[512, 16, 240]" = torch.ops.aten.reshape.default(addmm_33, [512, 16, 240]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_231: "f32[512, 16, 240]" = torch.ops.aten.add.Tensor(add_228, view_100);  add_228 = view_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_48 = torch.ops.aten.var_mean.correction(add_231, [2], correction = 0, keepdim = True)
        getitem_159: "f32[512, 16, 1]" = var_mean_48[0]
        getitem_160: "f32[512, 16, 1]" = var_mean_48[1];  var_mean_48 = None
        add_232: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_159, 1e-05);  getitem_159 = None
        rsqrt_48: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_232);  add_232 = None
        sub_48: "f32[512, 16, 240]" = torch.ops.aten.sub.Tensor(add_231, getitem_160);  getitem_160 = None
        mul_241: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
        mul_242: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(mul_241, primals_285)
        add_233: "f32[512, 16, 240]" = torch.ops.aten.add.Tensor(mul_242, primals_286);  mul_242 = primals_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_101: "f32[8192, 240]" = torch.ops.aten.reshape.default(add_233, [8192, 240]);  add_233 = None
        permute_62: "f32[240, 480]" = torch.ops.aten.permute.default(primals_287, [1, 0])
        addmm_34: "f32[8192, 480]" = torch.ops.aten.addmm.default(primals_288, view_101, permute_62);  primals_288 = permute_62 = None
        view_102: "f32[512, 16, 480]" = torch.ops.aten.reshape.default(addmm_34, [512, 16, 480])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        neg_30: "f32[512, 16, 480]" = torch.ops.aten.neg.default(view_102)
        exp_30: "f32[512, 16, 480]" = torch.ops.aten.exp.default(neg_30);  neg_30 = None
        add_234: "f32[512, 16, 480]" = torch.ops.aten.add.Tensor(exp_30, 1);  exp_30 = None
        div_30: "f32[512, 16, 480]" = torch.ops.aten.div.Tensor(view_102, add_234);  view_102 = add_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_103: "f32[8192, 480]" = torch.ops.aten.reshape.default(div_30, [8192, 480]);  div_30 = None
        permute_63: "f32[480, 240]" = torch.ops.aten.permute.default(primals_289, [1, 0])
        addmm_35: "f32[8192, 240]" = torch.ops.aten.addmm.default(primals_290, view_103, permute_63);  primals_290 = permute_63 = None
        view_104: "f32[512, 16, 240]" = torch.ops.aten.reshape.default(addmm_35, [512, 16, 240]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_235: "f32[512, 16, 240]" = torch.ops.aten.add.Tensor(add_231, view_104);  add_231 = view_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:264 in forward, code: x = self.norm(x)
        var_mean_49 = torch.ops.aten.var_mean.correction(add_235, [2], correction = 0, keepdim = True)
        getitem_161: "f32[512, 16, 1]" = var_mean_49[0]
        getitem_162: "f32[512, 16, 1]" = var_mean_49[1];  var_mean_49 = None
        add_236: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_161, 1e-05);  getitem_161 = None
        rsqrt_49: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_236);  add_236 = None
        sub_49: "f32[512, 16, 240]" = torch.ops.aten.sub.Tensor(add_235, getitem_162);  add_235 = getitem_162 = None
        mul_243: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
        mul_244: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(mul_243, primals_291)
        add_237: "f32[512, 16, 240]" = torch.ops.aten.add.Tensor(mul_244, primals_292);  mul_244 = primals_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:268 in forward, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
        view_105: "f32[128, 4, 16, 240]" = torch.ops.aten.reshape.default(add_237, [128, 4, 16, -1]);  add_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:269 in forward, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
        permute_64: "f32[128, 240, 16, 4]" = torch.ops.aten.permute.default(view_105, [0, 3, 2, 1]);  view_105 = None
        clone_62: "f32[128, 240, 16, 4]" = torch.ops.aten.clone.default(permute_64, memory_format = torch.contiguous_format);  permute_64 = None
        view_106: "f32[122880, 4, 2, 2]" = torch.ops.aten.reshape.default(clone_62, [122880, 4, 2, 2]);  clone_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:271 in forward, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
        permute_65: "f32[122880, 2, 4, 2]" = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None
        clone_63: "f32[122880, 2, 4, 2]" = torch.ops.aten.clone.default(permute_65, memory_format = torch.contiguous_format);  permute_65 = None
        view_107: "f32[128, 240, 8, 8]" = torch.ops.aten.reshape.default(clone_63, [128, 240, 8, 8]);  clone_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_32: "f32[128, 160, 8, 8]" = torch.ops.aten.convolution.default(view_107, primals_293, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_238: "i64[]" = torch.ops.aten.add.Tensor(primals_294, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_50 = torch.ops.aten.var_mean.correction(convolution_32, [0, 2, 3], correction = 0, keepdim = True)
        getitem_163: "f32[1, 160, 1, 1]" = var_mean_50[0]
        getitem_164: "f32[1, 160, 1, 1]" = var_mean_50[1];  var_mean_50 = None
        add_239: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_163, 1e-05)
        rsqrt_50: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_239);  add_239 = None
        sub_50: "f32[128, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_32, getitem_164)
        mul_245: "f32[128, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
        squeeze_87: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_164, [0, 2, 3])
        mul_246: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1);  squeeze_87 = None
        mul_247: "f32[160]" = torch.ops.aten.mul.Tensor(primals_295, 0.9)
        add_240: "f32[160]" = torch.ops.aten.add.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
        squeeze_89: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_163, [0, 2, 3]);  getitem_163 = None
        mul_248: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0001220852154804);  squeeze_89 = None
        mul_249: "f32[160]" = torch.ops.aten.mul.Tensor(mul_248, 0.1);  mul_248 = None
        mul_250: "f32[160]" = torch.ops.aten.mul.Tensor(primals_296, 0.9)
        add_241: "f32[160]" = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
        unsqueeze_116: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_297, -1)
        unsqueeze_117: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        mul_251: "f32[128, 160, 8, 8]" = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_117);  mul_245 = unsqueeze_117 = None
        unsqueeze_118: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_298, -1)
        unsqueeze_119: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
        add_242: "f32[128, 160, 8, 8]" = torch.ops.aten.add.Tensor(mul_251, unsqueeze_119);  mul_251 = unsqueeze_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_31: "f32[128, 160, 8, 8]" = torch.ops.aten.neg.default(add_242)
        exp_31: "f32[128, 160, 8, 8]" = torch.ops.aten.exp.default(neg_31);  neg_31 = None
        add_243: "f32[128, 160, 8, 8]" = torch.ops.aten.add.Tensor(exp_31, 1);  exp_31 = None
        div_31: "f32[128, 160, 8, 8]" = torch.ops.aten.div.Tensor(add_242, add_243);  add_242 = add_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:277 in forward, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
        cat_2: "f32[128, 320, 8, 8]" = torch.ops.aten.cat.default([add_208, div_31], 1);  div_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_33: "f32[128, 160, 8, 8]" = torch.ops.aten.convolution.default(cat_2, primals_299, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_244: "i64[]" = torch.ops.aten.add.Tensor(primals_300, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_51 = torch.ops.aten.var_mean.correction(convolution_33, [0, 2, 3], correction = 0, keepdim = True)
        getitem_165: "f32[1, 160, 1, 1]" = var_mean_51[0]
        getitem_166: "f32[1, 160, 1, 1]" = var_mean_51[1];  var_mean_51 = None
        add_245: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_165, 1e-05)
        rsqrt_51: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_245);  add_245 = None
        sub_51: "f32[128, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_33, getitem_166)
        mul_252: "f32[128, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
        squeeze_90: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_166, [0, 2, 3])
        mul_253: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1);  squeeze_90 = None
        mul_254: "f32[160]" = torch.ops.aten.mul.Tensor(primals_301, 0.9)
        add_246: "f32[160]" = torch.ops.aten.add.Tensor(mul_253, mul_254);  mul_253 = mul_254 = None
        squeeze_92: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_165, [0, 2, 3]);  getitem_165 = None
        mul_255: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0001220852154804);  squeeze_92 = None
        mul_256: "f32[160]" = torch.ops.aten.mul.Tensor(mul_255, 0.1);  mul_255 = None
        mul_257: "f32[160]" = torch.ops.aten.mul.Tensor(primals_302, 0.9)
        add_247: "f32[160]" = torch.ops.aten.add.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
        unsqueeze_120: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_303, -1)
        unsqueeze_121: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        mul_258: "f32[128, 160, 8, 8]" = torch.ops.aten.mul.Tensor(mul_252, unsqueeze_121);  mul_252 = unsqueeze_121 = None
        unsqueeze_122: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_304, -1)
        unsqueeze_123: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        add_248: "f32[128, 160, 8, 8]" = torch.ops.aten.add.Tensor(mul_258, unsqueeze_123);  mul_258 = unsqueeze_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_32: "f32[128, 160, 8, 8]" = torch.ops.aten.neg.default(add_248)
        exp_32: "f32[128, 160, 8, 8]" = torch.ops.aten.exp.default(neg_32);  neg_32 = None
        add_249: "f32[128, 160, 8, 8]" = torch.ops.aten.add.Tensor(exp_32, 1);  exp_32 = None
        div_32: "f32[128, 160, 8, 8]" = torch.ops.aten.div.Tensor(add_248, add_249);  add_248 = add_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_34: "f32[128, 640, 8, 8]" = torch.ops.aten.convolution.default(div_32, primals_305, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_250: "i64[]" = torch.ops.aten.add.Tensor(primals_306, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_52 = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
        getitem_167: "f32[1, 640, 1, 1]" = var_mean_52[0]
        getitem_168: "f32[1, 640, 1, 1]" = var_mean_52[1];  var_mean_52 = None
        add_251: "f32[1, 640, 1, 1]" = torch.ops.aten.add.Tensor(getitem_167, 1e-05)
        rsqrt_52: "f32[1, 640, 1, 1]" = torch.ops.aten.rsqrt.default(add_251);  add_251 = None
        sub_52: "f32[128, 640, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_168)
        mul_259: "f32[128, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
        squeeze_93: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_168, [0, 2, 3])
        mul_260: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1);  squeeze_93 = None
        mul_261: "f32[640]" = torch.ops.aten.mul.Tensor(primals_307, 0.9)
        add_252: "f32[640]" = torch.ops.aten.add.Tensor(mul_260, mul_261);  mul_260 = mul_261 = None
        squeeze_95: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_167, [0, 2, 3]);  getitem_167 = None
        mul_262: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0001220852154804);  squeeze_95 = None
        mul_263: "f32[640]" = torch.ops.aten.mul.Tensor(mul_262, 0.1);  mul_262 = None
        mul_264: "f32[640]" = torch.ops.aten.mul.Tensor(primals_308, 0.9)
        add_253: "f32[640]" = torch.ops.aten.add.Tensor(mul_263, mul_264);  mul_263 = mul_264 = None
        unsqueeze_124: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_309, -1)
        unsqueeze_125: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        mul_265: "f32[128, 640, 8, 8]" = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_125);  mul_259 = unsqueeze_125 = None
        unsqueeze_126: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_310, -1)
        unsqueeze_127: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        add_254: "f32[128, 640, 8, 8]" = torch.ops.aten.add.Tensor(mul_265, unsqueeze_127);  mul_265 = unsqueeze_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_33: "f32[128, 640, 8, 8]" = torch.ops.aten.neg.default(add_254)
        exp_33: "f32[128, 640, 8, 8]" = torch.ops.aten.exp.default(neg_33);  neg_33 = None
        add_255: "f32[128, 640, 8, 8]" = torch.ops.aten.add.Tensor(exp_33, 1);  exp_33 = None
        div_33: "f32[128, 640, 8, 8]" = torch.ops.aten.div.Tensor(add_254, add_255);  add_254 = add_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean: "f32[128, 640, 1, 1]" = torch.ops.aten.mean.dim(div_33, [-1, -2], True);  div_33 = None
        as_strided: "f32[128, 640, 1, 1]" = torch.ops.aten.as_strided.default(mean, [128, 640, 1, 1], [640, 1, 640, 640]);  mean = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_108: "f32[128, 640]" = torch.ops.aten.reshape.default(as_strided, [128, 640]);  as_strided = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:141 in forward, code: x = self.fc(x)
        permute_66: "f32[640, 1000]" = torch.ops.aten.permute.default(primals_311, [1, 0])
        addmm_36: "f32[128, 1000]" = torch.ops.aten.addmm.default(primals_312, view_108, permute_66);  primals_312 = permute_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:264 in forward, code: x = self.norm(x)
        div_35: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_49, 240);  rsqrt_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        div_36: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_48, 240);  rsqrt_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        div_37: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_47, 240);  rsqrt_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        div_38: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 240);  rsqrt_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        div_39: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 240);  rsqrt_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        div_40: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 240);  rsqrt_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        div_41: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 240);  rsqrt_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_176: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
        unsqueeze_177: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, 2);  unsqueeze_176 = None
        unsqueeze_178: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_177, 3);  unsqueeze_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:264 in forward, code: x = self.norm(x)
        div_42: "f32[512, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 192);  rsqrt_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        div_43: "f32[512, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 192);  rsqrt_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        div_44: "f32[512, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 192);  rsqrt_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        div_45: "f32[512, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 192);  rsqrt_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        div_46: "f32[512, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 192);  rsqrt_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        div_47: "f32[512, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 192);  rsqrt_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        div_48: "f32[512, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 192);  rsqrt_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        div_49: "f32[512, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 192);  rsqrt_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        div_50: "f32[512, 64, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 192);  rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_248: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
        unsqueeze_249: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 2);  unsqueeze_248 = None
        unsqueeze_250: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_249, 3);  unsqueeze_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:264 in forward, code: x = self.norm(x)
        div_51: "f32[512, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 144);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        div_52: "f32[512, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 144);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        div_53: "f32[512, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 144);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        div_54: "f32[512, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 144);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        div_55: "f32[512, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 144);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_320: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
        unsqueeze_321: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 2);  unsqueeze_320 = None
        unsqueeze_322: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 3);  unsqueeze_321 = None
        unsqueeze_356: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
        unsqueeze_357: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 2);  unsqueeze_356 = None
        unsqueeze_358: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 3);  unsqueeze_357 = None
        unsqueeze_392: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
        unsqueeze_393: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 2);  unsqueeze_392 = None
        unsqueeze_394: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 3);  unsqueeze_393 = None
        unsqueeze_428: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
        unsqueeze_429: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 2);  unsqueeze_428 = None
        unsqueeze_430: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 3);  unsqueeze_429 = None
        unsqueeze_464: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
        unsqueeze_465: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 2);  unsqueeze_464 = None
        unsqueeze_466: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 3);  unsqueeze_465 = None

        # No stacktrace found for following nodes
        copy_: "i64[]" = torch.ops.aten.copy_.default(primals_3, add);  primals_3 = add = copy_ = None
        copy__1: "f32[16]" = torch.ops.aten.copy_.default(primals_4, add_2);  primals_4 = add_2 = copy__1 = None
        copy__2: "f32[16]" = torch.ops.aten.copy_.default(primals_5, add_3);  primals_5 = add_3 = copy__2 = None
        copy__3: "i64[]" = torch.ops.aten.copy_.default(primals_9, add_6);  primals_9 = add_6 = copy__3 = None
        copy__4: "f32[64]" = torch.ops.aten.copy_.default(primals_10, add_8);  primals_10 = add_8 = copy__4 = None
        copy__5: "f32[64]" = torch.ops.aten.copy_.default(primals_11, add_9);  primals_11 = add_9 = copy__5 = None
        copy__6: "i64[]" = torch.ops.aten.copy_.default(primals_15, add_12);  primals_15 = add_12 = copy__6 = None
        copy__7: "f32[64]" = torch.ops.aten.copy_.default(primals_16, add_14);  primals_16 = add_14 = copy__7 = None
        copy__8: "f32[64]" = torch.ops.aten.copy_.default(primals_17, add_15);  primals_17 = add_15 = copy__8 = None
        copy__9: "i64[]" = torch.ops.aten.copy_.default(primals_21, add_18);  primals_21 = add_18 = copy__9 = None
        copy__10: "f32[32]" = torch.ops.aten.copy_.default(primals_22, add_20);  primals_22 = add_20 = copy__10 = None
        copy__11: "f32[32]" = torch.ops.aten.copy_.default(primals_23, add_21);  primals_23 = add_21 = copy__11 = None
        copy__12: "i64[]" = torch.ops.aten.copy_.default(primals_27, add_23);  primals_27 = add_23 = copy__12 = None
        copy__13: "f32[128]" = torch.ops.aten.copy_.default(primals_28, add_25);  primals_28 = add_25 = copy__13 = None
        copy__14: "f32[128]" = torch.ops.aten.copy_.default(primals_29, add_26);  primals_29 = add_26 = copy__14 = None
        copy__15: "i64[]" = torch.ops.aten.copy_.default(primals_33, add_29);  primals_33 = add_29 = copy__15 = None
        copy__16: "f32[128]" = torch.ops.aten.copy_.default(primals_34, add_31);  primals_34 = add_31 = copy__16 = None
        copy__17: "f32[128]" = torch.ops.aten.copy_.default(primals_35, add_32);  primals_35 = add_32 = copy__17 = None
        copy__18: "i64[]" = torch.ops.aten.copy_.default(primals_39, add_35);  primals_39 = add_35 = copy__18 = None
        copy__19: "f32[64]" = torch.ops.aten.copy_.default(primals_40, add_37);  primals_40 = add_37 = copy__19 = None
        copy__20: "f32[64]" = torch.ops.aten.copy_.default(primals_41, add_38);  primals_41 = add_38 = copy__20 = None
        copy__21: "i64[]" = torch.ops.aten.copy_.default(primals_45, add_40);  primals_45 = add_40 = copy__21 = None
        copy__22: "f32[256]" = torch.ops.aten.copy_.default(primals_46, add_42);  primals_46 = add_42 = copy__22 = None
        copy__23: "f32[256]" = torch.ops.aten.copy_.default(primals_47, add_43);  primals_47 = add_43 = copy__23 = None
        copy__24: "i64[]" = torch.ops.aten.copy_.default(primals_51, add_46);  primals_51 = add_46 = copy__24 = None
        copy__25: "f32[256]" = torch.ops.aten.copy_.default(primals_52, add_48);  primals_52 = add_48 = copy__25 = None
        copy__26: "f32[256]" = torch.ops.aten.copy_.default(primals_53, add_49);  primals_53 = add_49 = copy__26 = None
        copy__27: "i64[]" = torch.ops.aten.copy_.default(primals_57, add_52);  primals_57 = add_52 = copy__27 = None
        copy__28: "f32[64]" = torch.ops.aten.copy_.default(primals_58, add_54);  primals_58 = add_54 = copy__28 = None
        copy__29: "f32[64]" = torch.ops.aten.copy_.default(primals_59, add_55);  primals_59 = add_55 = copy__29 = None
        copy__30: "i64[]" = torch.ops.aten.copy_.default(primals_63, add_58);  primals_63 = add_58 = copy__30 = None
        copy__31: "f32[256]" = torch.ops.aten.copy_.default(primals_64, add_60);  primals_64 = add_60 = copy__31 = None
        copy__32: "f32[256]" = torch.ops.aten.copy_.default(primals_65, add_61);  primals_65 = add_61 = copy__32 = None
        copy__33: "i64[]" = torch.ops.aten.copy_.default(primals_69, add_64);  primals_69 = add_64 = copy__33 = None
        copy__34: "f32[256]" = torch.ops.aten.copy_.default(primals_70, add_66);  primals_70 = add_66 = copy__34 = None
        copy__35: "f32[256]" = torch.ops.aten.copy_.default(primals_71, add_67);  primals_71 = add_67 = copy__35 = None
        copy__36: "i64[]" = torch.ops.aten.copy_.default(primals_75, add_70);  primals_75 = add_70 = copy__36 = None
        copy__37: "f32[64]" = torch.ops.aten.copy_.default(primals_76, add_72);  primals_76 = add_72 = copy__37 = None
        copy__38: "f32[64]" = torch.ops.aten.copy_.default(primals_77, add_73);  primals_77 = add_73 = copy__38 = None
        copy__39: "i64[]" = torch.ops.aten.copy_.default(primals_81, add_76);  primals_81 = add_76 = copy__39 = None
        copy__40: "f32[256]" = torch.ops.aten.copy_.default(primals_82, add_78);  primals_82 = add_78 = copy__40 = None
        copy__41: "f32[256]" = torch.ops.aten.copy_.default(primals_83, add_79);  primals_83 = add_79 = copy__41 = None
        copy__42: "i64[]" = torch.ops.aten.copy_.default(primals_87, add_82);  primals_87 = add_82 = copy__42 = None
        copy__43: "f32[256]" = torch.ops.aten.copy_.default(primals_88, add_84);  primals_88 = add_84 = copy__43 = None
        copy__44: "f32[256]" = torch.ops.aten.copy_.default(primals_89, add_85);  primals_89 = add_85 = copy__44 = None
        copy__45: "i64[]" = torch.ops.aten.copy_.default(primals_93, add_88);  primals_93 = add_88 = copy__45 = None
        copy__46: "f32[96]" = torch.ops.aten.copy_.default(primals_94, add_90);  primals_94 = add_90 = copy__46 = None
        copy__47: "f32[96]" = torch.ops.aten.copy_.default(primals_95, add_91);  primals_95 = add_91 = copy__47 = None
        copy__48: "i64[]" = torch.ops.aten.copy_.default(primals_99, add_93);  primals_99 = add_93 = copy__48 = None
        copy__49: "f32[96]" = torch.ops.aten.copy_.default(primals_100, add_95);  primals_100 = add_95 = copy__49 = None
        copy__50: "f32[96]" = torch.ops.aten.copy_.default(primals_101, add_96);  primals_101 = add_96 = copy__50 = None
        copy__51: "i64[]" = torch.ops.aten.copy_.default(primals_132, add_115);  primals_132 = add_115 = copy__51 = None
        copy__52: "f32[96]" = torch.ops.aten.copy_.default(primals_133, add_117);  primals_133 = add_117 = copy__52 = None
        copy__53: "f32[96]" = torch.ops.aten.copy_.default(primals_134, add_118);  primals_134 = add_118 = copy__53 = None
        copy__54: "i64[]" = torch.ops.aten.copy_.default(primals_138, add_121);  primals_138 = add_121 = copy__54 = None
        copy__55: "f32[96]" = torch.ops.aten.copy_.default(primals_139, add_123);  primals_139 = add_123 = copy__55 = None
        copy__56: "f32[96]" = torch.ops.aten.copy_.default(primals_140, add_124);  primals_140 = add_124 = copy__56 = None
        copy__57: "i64[]" = torch.ops.aten.copy_.default(primals_144, add_127);  primals_144 = add_127 = copy__57 = None
        copy__58: "f32[384]" = torch.ops.aten.copy_.default(primals_145, add_129);  primals_145 = add_129 = copy__58 = None
        copy__59: "f32[384]" = torch.ops.aten.copy_.default(primals_146, add_130);  primals_146 = add_130 = copy__59 = None
        copy__60: "i64[]" = torch.ops.aten.copy_.default(primals_150, add_133);  primals_150 = add_133 = copy__60 = None
        copy__61: "f32[384]" = torch.ops.aten.copy_.default(primals_151, add_135);  primals_151 = add_135 = copy__61 = None
        copy__62: "f32[384]" = torch.ops.aten.copy_.default(primals_152, add_136);  primals_152 = add_136 = copy__62 = None
        copy__63: "i64[]" = torch.ops.aten.copy_.default(primals_156, add_139);  primals_156 = add_139 = copy__63 = None
        copy__64: "f32[128]" = torch.ops.aten.copy_.default(primals_157, add_141);  primals_157 = add_141 = copy__64 = None
        copy__65: "f32[128]" = torch.ops.aten.copy_.default(primals_158, add_142);  primals_158 = add_142 = copy__65 = None
        copy__66: "i64[]" = torch.ops.aten.copy_.default(primals_162, add_144);  primals_162 = add_144 = copy__66 = None
        copy__67: "f32[128]" = torch.ops.aten.copy_.default(primals_163, add_146);  primals_163 = add_146 = copy__67 = None
        copy__68: "f32[128]" = torch.ops.aten.copy_.default(primals_164, add_147);  primals_164 = add_147 = copy__68 = None
        copy__69: "i64[]" = torch.ops.aten.copy_.default(primals_219, add_180);  primals_219 = add_180 = copy__69 = None
        copy__70: "f32[128]" = torch.ops.aten.copy_.default(primals_220, add_182);  primals_220 = add_182 = copy__70 = None
        copy__71: "f32[128]" = torch.ops.aten.copy_.default(primals_221, add_183);  primals_221 = add_183 = copy__71 = None
        copy__72: "i64[]" = torch.ops.aten.copy_.default(primals_225, add_186);  primals_225 = add_186 = copy__72 = None
        copy__73: "f32[128]" = torch.ops.aten.copy_.default(primals_226, add_188);  primals_226 = add_188 = copy__73 = None
        copy__74: "f32[128]" = torch.ops.aten.copy_.default(primals_227, add_189);  primals_227 = add_189 = copy__74 = None
        copy__75: "i64[]" = torch.ops.aten.copy_.default(primals_231, add_192);  primals_231 = add_192 = copy__75 = None
        copy__76: "f32[512]" = torch.ops.aten.copy_.default(primals_232, add_194);  primals_232 = add_194 = copy__76 = None
        copy__77: "f32[512]" = torch.ops.aten.copy_.default(primals_233, add_195);  primals_233 = add_195 = copy__77 = None
        copy__78: "i64[]" = torch.ops.aten.copy_.default(primals_237, add_198);  primals_237 = add_198 = copy__78 = None
        copy__79: "f32[512]" = torch.ops.aten.copy_.default(primals_238, add_200);  primals_238 = add_200 = copy__79 = None
        copy__80: "f32[512]" = torch.ops.aten.copy_.default(primals_239, add_201);  primals_239 = add_201 = copy__80 = None
        copy__81: "i64[]" = torch.ops.aten.copy_.default(primals_243, add_204);  primals_243 = add_204 = copy__81 = None
        copy__82: "f32[160]" = torch.ops.aten.copy_.default(primals_244, add_206);  primals_244 = add_206 = copy__82 = None
        copy__83: "f32[160]" = torch.ops.aten.copy_.default(primals_245, add_207);  primals_245 = add_207 = copy__83 = None
        copy__84: "i64[]" = torch.ops.aten.copy_.default(primals_249, add_209);  primals_249 = add_209 = copy__84 = None
        copy__85: "f32[160]" = torch.ops.aten.copy_.default(primals_250, add_211);  primals_250 = add_211 = copy__85 = None
        copy__86: "f32[160]" = torch.ops.aten.copy_.default(primals_251, add_212);  primals_251 = add_212 = copy__86 = None
        copy__87: "i64[]" = torch.ops.aten.copy_.default(primals_294, add_238);  primals_294 = add_238 = copy__87 = None
        copy__88: "f32[160]" = torch.ops.aten.copy_.default(primals_295, add_240);  primals_295 = add_240 = copy__88 = None
        copy__89: "f32[160]" = torch.ops.aten.copy_.default(primals_296, add_241);  primals_296 = add_241 = copy__89 = None
        copy__90: "i64[]" = torch.ops.aten.copy_.default(primals_300, add_244);  primals_300 = add_244 = copy__90 = None
        copy__91: "f32[160]" = torch.ops.aten.copy_.default(primals_301, add_246);  primals_301 = add_246 = copy__91 = None
        copy__92: "f32[160]" = torch.ops.aten.copy_.default(primals_302, add_247);  primals_302 = add_247 = copy__92 = None
        copy__93: "i64[]" = torch.ops.aten.copy_.default(primals_306, add_250);  primals_306 = add_250 = copy__93 = None
        copy__94: "f32[640]" = torch.ops.aten.copy_.default(primals_307, add_252);  primals_307 = add_252 = copy__94 = None
        copy__95: "f32[640]" = torch.ops.aten.copy_.default(primals_308, add_253);  primals_308 = add_253 = copy__95 = None
        return (addmm_36, primals_1, primals_2, primals_6, primals_7, primals_8, primals_12, primals_13, primals_14, primals_18, primals_19, primals_20, primals_24, primals_26, primals_30, primals_31, primals_32, primals_36, primals_37, primals_38, primals_42, primals_44, primals_48, primals_49, primals_50, primals_54, primals_55, primals_56, primals_60, primals_62, primals_66, primals_67, primals_68, primals_72, primals_73, primals_74, primals_78, primals_80, primals_84, primals_85, primals_86, primals_90, primals_91, primals_92, primals_96, primals_98, primals_102, primals_103, primals_104, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_135, primals_136, primals_137, primals_141, primals_142, primals_143, primals_147, primals_148, primals_149, primals_153, primals_154, primals_155, primals_159, primals_161, primals_165, primals_166, primals_167, primals_168, primals_170, primals_172, primals_174, primals_176, primals_178, primals_180, primals_182, primals_184, primals_186, primals_188, primals_190, primals_192, primals_194, primals_196, primals_198, primals_200, primals_202, primals_204, primals_206, primals_208, primals_210, primals_212, primals_214, primals_216, primals_218, primals_222, primals_223, primals_224, primals_228, primals_229, primals_230, primals_234, primals_235, primals_236, primals_240, primals_241, primals_242, primals_246, primals_248, primals_252, primals_253, primals_254, primals_255, primals_257, primals_259, primals_261, primals_263, primals_265, primals_267, primals_269, primals_271, primals_273, primals_275, primals_277, primals_279, primals_281, primals_283, primals_285, primals_287, primals_289, primals_291, primals_293, primals_297, primals_298, primals_299, primals_303, primals_304, primals_305, primals_309, primals_310, primals_311, convolution, getitem_1, rsqrt, div, convolution_1, getitem_3, rsqrt_1, div_1, convolution_2, getitem_5, rsqrt_2, div_2, convolution_3, squeeze_10, add_22, convolution_4, getitem_9, rsqrt_4, div_3, convolution_5, getitem_11, rsqrt_5, div_4, convolution_6, squeeze_19, add_39, convolution_7, getitem_15, rsqrt_7, div_5, convolution_8, getitem_17, rsqrt_8, div_6, convolution_9, squeeze_28, add_57, convolution_10, getitem_21, rsqrt_10, div_7, convolution_11, getitem_23, rsqrt_11, div_8, convolution_12, squeeze_37, add_75, convolution_13, getitem_27, rsqrt_13, div_9, convolution_14, getitem_29, rsqrt_14, div_10, convolution_15, squeeze_46, add_92, convolution_16, getitem_33, rsqrt_16, div_11, mul_119, view_3, getitem_36, getitem_37, getitem_38, getitem_39, getitem_40, getitem_41, getitem_42, mul_121, view_9, addmm_2, view_11, mul_123, view_13, getitem_47, getitem_48, getitem_49, getitem_50, getitem_51, getitem_52, getitem_53, mul_125, view_19, addmm_6, view_21, mul_127, view_25, convolution_18, getitem_59, rsqrt_22, cat, convolution_19, getitem_61, rsqrt_23, div_15, convolution_20, getitem_63, rsqrt_24, div_16, convolution_21, getitem_65, rsqrt_25, div_17, convolution_22, squeeze_64, add_143, convolution_23, getitem_69, rsqrt_27, div_18, mul_171, view_29, getitem_72, getitem_73, getitem_74, getitem_75, getitem_76, getitem_77, getitem_78, mul_173, view_35, addmm_10, view_37, mul_175, view_39, getitem_83, getitem_84, getitem_85, getitem_86, getitem_87, getitem_88, getitem_89, mul_177, view_45, addmm_14, view_47, mul_179, view_49, getitem_94, getitem_95, getitem_96, getitem_97, getitem_98, getitem_99, getitem_100, mul_181, view_55, addmm_18, view_57, mul_183, view_59, getitem_105, getitem_106, getitem_107, getitem_108, getitem_109, getitem_110, getitem_111, mul_185, view_65, addmm_22, view_67, mul_187, view_71, convolution_25, getitem_117, rsqrt_37, cat_1, convolution_26, getitem_119, rsqrt_38, div_24, convolution_27, getitem_121, rsqrt_39, div_25, convolution_28, getitem_123, rsqrt_40, div_26, convolution_29, squeeze_82, add_208, convolution_30, getitem_127, rsqrt_42, div_27, mul_231, view_75, getitem_130, getitem_131, getitem_132, getitem_133, getitem_134, getitem_135, getitem_136, mul_233, view_81, addmm_26, view_83, mul_235, view_85, getitem_141, getitem_142, getitem_143, getitem_144, getitem_145, getitem_146, getitem_147, mul_237, view_91, addmm_30, view_93, mul_239, view_95, getitem_152, getitem_153, getitem_154, getitem_155, getitem_156, getitem_157, getitem_158, mul_241, view_101, addmm_34, view_103, mul_243, view_107, convolution_32, getitem_164, rsqrt_50, cat_2, convolution_33, getitem_166, rsqrt_51, div_32, convolution_34, getitem_168, rsqrt_52, view_108, div_35, div_36, div_37, div_38, div_39, div_40, div_41, unsqueeze_178, div_42, div_43, div_44, div_45, div_46, div_47, div_48, div_49, div_50, unsqueeze_250, div_51, div_52, div_53, div_54, div_55, unsqueeze_322, unsqueeze_358, unsqueeze_394, unsqueeze_430, unsqueeze_466)
