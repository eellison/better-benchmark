class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[16, 3, 3, 3][27, 1, 9, 3]cuda:0", primals_2: "f32[512, 3, 224, 224][150528, 1, 672, 3]cuda:0", primals_3: "i64[][]cuda:0", primals_4: "f32[16][1]cuda:0", primals_5: "f32[16][1]cuda:0", primals_6: "f32[16][1]cuda:0", primals_7: "f32[16][1]cuda:0", primals_8: "f32[16, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_9: "i64[][]cuda:0", primals_10: "f32[16][1]cuda:0", primals_11: "f32[16][1]cuda:0", primals_12: "f32[16][1]cuda:0", primals_13: "f32[16][1]cuda:0", primals_14: "f32[16, 16, 1, 1][16, 1, 16, 16]cuda:0", primals_15: "i64[][]cuda:0", primals_16: "f32[16][1]cuda:0", primals_17: "f32[16][1]cuda:0", primals_18: "f32[16][1]cuda:0", primals_19: "f32[16][1]cuda:0", primals_20: "f32[64, 16, 1, 1][16, 1, 16, 16]cuda:0", primals_21: "i64[][]cuda:0", primals_22: "f32[64][1]cuda:0", primals_23: "f32[64][1]cuda:0", primals_24: "f32[64][1]cuda:0", primals_25: "f32[64][1]cuda:0", primals_26: "f32[64, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_27: "i64[][]cuda:0", primals_28: "f32[64][1]cuda:0", primals_29: "f32[64][1]cuda:0", primals_30: "f32[64][1]cuda:0", primals_31: "f32[64][1]cuda:0", primals_32: "f32[24, 64, 1, 1][64, 1, 64, 64]cuda:0", primals_33: "i64[][]cuda:0", primals_34: "f32[24][1]cuda:0", primals_35: "f32[24][1]cuda:0", primals_36: "f32[24][1]cuda:0", primals_37: "f32[24][1]cuda:0", primals_38: "f32[72, 24, 1, 1][24, 1, 24, 24]cuda:0", primals_39: "i64[][]cuda:0", primals_40: "f32[72][1]cuda:0", primals_41: "f32[72][1]cuda:0", primals_42: "f32[72][1]cuda:0", primals_43: "f32[72][1]cuda:0", primals_44: "f32[72, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_45: "i64[][]cuda:0", primals_46: "f32[72][1]cuda:0", primals_47: "f32[72][1]cuda:0", primals_48: "f32[72][1]cuda:0", primals_49: "f32[72][1]cuda:0", primals_50: "f32[24, 72, 1, 1][72, 1, 72, 72]cuda:0", primals_51: "i64[][]cuda:0", primals_52: "f32[24][1]cuda:0", primals_53: "f32[24][1]cuda:0", primals_54: "f32[24][1]cuda:0", primals_55: "f32[24][1]cuda:0", primals_56: "f32[72, 24, 1, 1][24, 1, 24, 24]cuda:0", primals_57: "i64[][]cuda:0", primals_58: "f32[72][1]cuda:0", primals_59: "f32[72][1]cuda:0", primals_60: "f32[72][1]cuda:0", primals_61: "f32[72][1]cuda:0", primals_62: "f32[72, 1, 5, 5][25, 1, 5, 1]cuda:0", primals_63: "i64[][]cuda:0", primals_64: "f32[72][1]cuda:0", primals_65: "f32[72][1]cuda:0", primals_66: "f32[72][1]cuda:0", primals_67: "f32[72][1]cuda:0", primals_68: "f32[24, 72, 1, 1][72, 1, 72, 72]cuda:0", primals_69: "f32[24][1]cuda:0", primals_70: "f32[72, 24, 1, 1][24, 1, 24, 24]cuda:0", primals_71: "f32[72][1]cuda:0", primals_72: "f32[40, 72, 1, 1][72, 1, 72, 72]cuda:0", primals_73: "i64[][]cuda:0", primals_74: "f32[40][1]cuda:0", primals_75: "f32[40][1]cuda:0", primals_76: "f32[40][1]cuda:0", primals_77: "f32[40][1]cuda:0", primals_78: "f32[120, 40, 1, 1][40, 1, 40, 40]cuda:0", primals_79: "i64[][]cuda:0", primals_80: "f32[120][1]cuda:0", primals_81: "f32[120][1]cuda:0", primals_82: "f32[120][1]cuda:0", primals_83: "f32[120][1]cuda:0", primals_84: "f32[120, 1, 5, 5][25, 1, 5, 1]cuda:0", primals_85: "i64[][]cuda:0", primals_86: "f32[120][1]cuda:0", primals_87: "f32[120][1]cuda:0", primals_88: "f32[120][1]cuda:0", primals_89: "f32[120][1]cuda:0", primals_90: "f32[32, 120, 1, 1][120, 1, 120, 120]cuda:0", primals_91: "f32[32][1]cuda:0", primals_92: "f32[120, 32, 1, 1][32, 1, 32, 32]cuda:0", primals_93: "f32[120][1]cuda:0", primals_94: "f32[40, 120, 1, 1][120, 1, 120, 120]cuda:0", primals_95: "i64[][]cuda:0", primals_96: "f32[40][1]cuda:0", primals_97: "f32[40][1]cuda:0", primals_98: "f32[40][1]cuda:0", primals_99: "f32[40][1]cuda:0", primals_100: "f32[120, 40, 1, 1][40, 1, 40, 40]cuda:0", primals_101: "i64[][]cuda:0", primals_102: "f32[120][1]cuda:0", primals_103: "f32[120][1]cuda:0", primals_104: "f32[120][1]cuda:0", primals_105: "f32[120][1]cuda:0", primals_106: "f32[120, 1, 5, 5][25, 1, 5, 1]cuda:0", primals_107: "i64[][]cuda:0", primals_108: "f32[120][1]cuda:0", primals_109: "f32[120][1]cuda:0", primals_110: "f32[120][1]cuda:0", primals_111: "f32[120][1]cuda:0", primals_112: "f32[32, 120, 1, 1][120, 1, 120, 120]cuda:0", primals_113: "f32[32][1]cuda:0", primals_114: "f32[120, 32, 1, 1][32, 1, 32, 32]cuda:0", primals_115: "f32[120][1]cuda:0", primals_116: "f32[40, 120, 1, 1][120, 1, 120, 120]cuda:0", primals_117: "i64[][]cuda:0", primals_118: "f32[40][1]cuda:0", primals_119: "f32[40][1]cuda:0", primals_120: "f32[40][1]cuda:0", primals_121: "f32[40][1]cuda:0", primals_122: "f32[240, 40, 1, 1][40, 1, 40, 40]cuda:0", primals_123: "i64[][]cuda:0", primals_124: "f32[240][1]cuda:0", primals_125: "f32[240][1]cuda:0", primals_126: "f32[240][1]cuda:0", primals_127: "f32[240][1]cuda:0", primals_128: "f32[240, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_129: "i64[][]cuda:0", primals_130: "f32[240][1]cuda:0", primals_131: "f32[240][1]cuda:0", primals_132: "f32[240][1]cuda:0", primals_133: "f32[240][1]cuda:0", primals_134: "f32[80, 240, 1, 1][240, 1, 240, 240]cuda:0", primals_135: "i64[][]cuda:0", primals_136: "f32[80][1]cuda:0", primals_137: "f32[80][1]cuda:0", primals_138: "f32[80][1]cuda:0", primals_139: "f32[80][1]cuda:0", primals_140: "f32[200, 80, 1, 1][80, 1, 80, 80]cuda:0", primals_141: "i64[][]cuda:0", primals_142: "f32[200][1]cuda:0", primals_143: "f32[200][1]cuda:0", primals_144: "f32[200][1]cuda:0", primals_145: "f32[200][1]cuda:0", primals_146: "f32[200, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_147: "i64[][]cuda:0", primals_148: "f32[200][1]cuda:0", primals_149: "f32[200][1]cuda:0", primals_150: "f32[200][1]cuda:0", primals_151: "f32[200][1]cuda:0", primals_152: "f32[80, 200, 1, 1][200, 1, 200, 200]cuda:0", primals_153: "i64[][]cuda:0", primals_154: "f32[80][1]cuda:0", primals_155: "f32[80][1]cuda:0", primals_156: "f32[80][1]cuda:0", primals_157: "f32[80][1]cuda:0", primals_158: "f32[184, 80, 1, 1][80, 1, 80, 80]cuda:0", primals_159: "i64[][]cuda:0", primals_160: "f32[184][1]cuda:0", primals_161: "f32[184][1]cuda:0", primals_162: "f32[184][1]cuda:0", primals_163: "f32[184][1]cuda:0", primals_164: "f32[184, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_165: "i64[][]cuda:0", primals_166: "f32[184][1]cuda:0", primals_167: "f32[184][1]cuda:0", primals_168: "f32[184][1]cuda:0", primals_169: "f32[184][1]cuda:0", primals_170: "f32[80, 184, 1, 1][184, 1, 184, 184]cuda:0", primals_171: "i64[][]cuda:0", primals_172: "f32[80][1]cuda:0", primals_173: "f32[80][1]cuda:0", primals_174: "f32[80][1]cuda:0", primals_175: "f32[80][1]cuda:0", primals_176: "f32[184, 80, 1, 1][80, 1, 80, 80]cuda:0", primals_177: "i64[][]cuda:0", primals_178: "f32[184][1]cuda:0", primals_179: "f32[184][1]cuda:0", primals_180: "f32[184][1]cuda:0", primals_181: "f32[184][1]cuda:0", primals_182: "f32[184, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_183: "i64[][]cuda:0", primals_184: "f32[184][1]cuda:0", primals_185: "f32[184][1]cuda:0", primals_186: "f32[184][1]cuda:0", primals_187: "f32[184][1]cuda:0", primals_188: "f32[80, 184, 1, 1][184, 1, 184, 184]cuda:0", primals_189: "i64[][]cuda:0", primals_190: "f32[80][1]cuda:0", primals_191: "f32[80][1]cuda:0", primals_192: "f32[80][1]cuda:0", primals_193: "f32[80][1]cuda:0", primals_194: "f32[480, 80, 1, 1][80, 1, 80, 80]cuda:0", primals_195: "i64[][]cuda:0", primals_196: "f32[480][1]cuda:0", primals_197: "f32[480][1]cuda:0", primals_198: "f32[480][1]cuda:0", primals_199: "f32[480][1]cuda:0", primals_200: "f32[480, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_201: "i64[][]cuda:0", primals_202: "f32[480][1]cuda:0", primals_203: "f32[480][1]cuda:0", primals_204: "f32[480][1]cuda:0", primals_205: "f32[480][1]cuda:0", primals_206: "f32[120, 480, 1, 1][480, 1, 480, 480]cuda:0", primals_207: "f32[120][1]cuda:0", primals_208: "f32[480, 120, 1, 1][120, 1, 120, 120]cuda:0", primals_209: "f32[480][1]cuda:0", primals_210: "f32[112, 480, 1, 1][480, 1, 480, 480]cuda:0", primals_211: "i64[][]cuda:0", primals_212: "f32[112][1]cuda:0", primals_213: "f32[112][1]cuda:0", primals_214: "f32[112][1]cuda:0", primals_215: "f32[112][1]cuda:0", primals_216: "f32[672, 112, 1, 1][112, 1, 112, 112]cuda:0", primals_217: "i64[][]cuda:0", primals_218: "f32[672][1]cuda:0", primals_219: "f32[672][1]cuda:0", primals_220: "f32[672][1]cuda:0", primals_221: "f32[672][1]cuda:0", primals_222: "f32[672, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_223: "i64[][]cuda:0", primals_224: "f32[672][1]cuda:0", primals_225: "f32[672][1]cuda:0", primals_226: "f32[672][1]cuda:0", primals_227: "f32[672][1]cuda:0", primals_228: "f32[168, 672, 1, 1][672, 1, 672, 672]cuda:0", primals_229: "f32[168][1]cuda:0", primals_230: "f32[672, 168, 1, 1][168, 1, 168, 168]cuda:0", primals_231: "f32[672][1]cuda:0", primals_232: "f32[112, 672, 1, 1][672, 1, 672, 672]cuda:0", primals_233: "i64[][]cuda:0", primals_234: "f32[112][1]cuda:0", primals_235: "f32[112][1]cuda:0", primals_236: "f32[112][1]cuda:0", primals_237: "f32[112][1]cuda:0", primals_238: "f32[672, 112, 1, 1][112, 1, 112, 112]cuda:0", primals_239: "i64[][]cuda:0", primals_240: "f32[672][1]cuda:0", primals_241: "f32[672][1]cuda:0", primals_242: "f32[672][1]cuda:0", primals_243: "f32[672][1]cuda:0", primals_244: "f32[672, 1, 5, 5][25, 1, 5, 1]cuda:0", primals_245: "i64[][]cuda:0", primals_246: "f32[672][1]cuda:0", primals_247: "f32[672][1]cuda:0", primals_248: "f32[672][1]cuda:0", primals_249: "f32[672][1]cuda:0", primals_250: "f32[168, 672, 1, 1][672, 1, 672, 672]cuda:0", primals_251: "f32[168][1]cuda:0", primals_252: "f32[672, 168, 1, 1][168, 1, 168, 168]cuda:0", primals_253: "f32[672][1]cuda:0", primals_254: "f32[160, 672, 1, 1][672, 1, 672, 672]cuda:0", primals_255: "i64[][]cuda:0", primals_256: "f32[160][1]cuda:0", primals_257: "f32[160][1]cuda:0", primals_258: "f32[160][1]cuda:0", primals_259: "f32[160][1]cuda:0", primals_260: "f32[960, 160, 1, 1][160, 1, 160, 160]cuda:0", primals_261: "i64[][]cuda:0", primals_262: "f32[960][1]cuda:0", primals_263: "f32[960][1]cuda:0", primals_264: "f32[960][1]cuda:0", primals_265: "f32[960][1]cuda:0", primals_266: "f32[960, 1, 5, 5][25, 1, 5, 1]cuda:0", primals_267: "i64[][]cuda:0", primals_268: "f32[960][1]cuda:0", primals_269: "f32[960][1]cuda:0", primals_270: "f32[960][1]cuda:0", primals_271: "f32[960][1]cuda:0", primals_272: "f32[240, 960, 1, 1][960, 1, 960, 960]cuda:0", primals_273: "f32[240][1]cuda:0", primals_274: "f32[960, 240, 1, 1][240, 1, 240, 240]cuda:0", primals_275: "f32[960][1]cuda:0", primals_276: "f32[160, 960, 1, 1][960, 1, 960, 960]cuda:0", primals_277: "i64[][]cuda:0", primals_278: "f32[160][1]cuda:0", primals_279: "f32[160][1]cuda:0", primals_280: "f32[160][1]cuda:0", primals_281: "f32[160][1]cuda:0", primals_282: "f32[960, 160, 1, 1][160, 1, 160, 160]cuda:0", primals_283: "i64[][]cuda:0", primals_284: "f32[960][1]cuda:0", primals_285: "f32[960][1]cuda:0", primals_286: "f32[960][1]cuda:0", primals_287: "f32[960][1]cuda:0", primals_288: "f32[960, 1, 5, 5][25, 1, 5, 1]cuda:0", primals_289: "i64[][]cuda:0", primals_290: "f32[960][1]cuda:0", primals_291: "f32[960][1]cuda:0", primals_292: "f32[960][1]cuda:0", primals_293: "f32[960][1]cuda:0", primals_294: "f32[240, 960, 1, 1][960, 1, 960, 960]cuda:0", primals_295: "f32[240][1]cuda:0", primals_296: "f32[960, 240, 1, 1][240, 1, 240, 240]cuda:0", primals_297: "f32[960][1]cuda:0", primals_298: "f32[160, 960, 1, 1][960, 1, 960, 960]cuda:0", primals_299: "i64[][]cuda:0", primals_300: "f32[160][1]cuda:0", primals_301: "f32[160][1]cuda:0", primals_302: "f32[160][1]cuda:0", primals_303: "f32[160][1]cuda:0", primals_304: "f32[960, 160, 1, 1][160, 1, 160, 160]cuda:0", primals_305: "i64[][]cuda:0", primals_306: "f32[960][1]cuda:0", primals_307: "f32[960][1]cuda:0", primals_308: "f32[960][1]cuda:0", primals_309: "f32[960][1]cuda:0", primals_310: "f32[1280, 960, 1, 1][960, 1, 960, 960]cuda:0", primals_311: "f32[1280][1]cuda:0", primals_312: "f32[1000, 1280][1280, 1]cuda:0", primals_313: "f32[1000][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilenetv3.py:304 in forward_features, code: x = self.conv_stem(x)
        convert_element_type: "bf16[16, 3, 3, 3][27, 1, 9, 3]cuda:0" = torch.ops.prims.convert_element_type.default(primals_1, torch.bfloat16);  primals_1 = None
        convert_element_type_1: "bf16[512, 3, 224, 224][150528, 1, 672, 3]cuda:0" = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16);  primals_2 = None
        convolution: "bf16[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_1, convert_element_type, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_2: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.prims.convert_element_type.default(convolution, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_2, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_2 = None
        getitem: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        add_1: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze: "f32[16][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3])
        mul_1: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze, 0.1);  squeeze = None
        mul_2: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_4, 0.9)
        add_2: "f32[16][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_2: "f32[16][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_3: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_2, 1.0000001557019536);  squeeze_2 = None
        mul_4: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
        mul_5: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_5, 0.9)
        add_3: "f32[16][1]cuda:0" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_6, -1)
        unsqueeze_1: "f32[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_7, -1)
        unsqueeze_3: "f32[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_4: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        convert_element_type_3: "bf16[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_4: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        add_5: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_4, 3)
        clamp_min: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.clamp_min.default(add_5, 0);  add_5 = None
        clamp_max: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min, 6);  clamp_min = None
        mul_7: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_4, clamp_max);  convert_element_type_4 = clamp_max = None
        div: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.div.Tensor(mul_7, 6);  mul_7 = None
        convert_element_type_5: "bf16[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:223 in forward, code: x = self.conv_dw(x)
        convert_element_type_6: "bf16[16, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_8, torch.bfloat16);  primals_8 = None
        convolution_1: "bf16[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_5, convert_element_type_6, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_6: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_9, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_7: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_7, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_7 = None
        getitem_2: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = var_mean_1[0]
        getitem_3: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        add_7: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
        rsqrt_1: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        sub_1: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
        mul_8: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        squeeze_3: "f32[16][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        squeeze_4: "f32[16][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_9: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
        mul_10: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_10, 0.9)
        add_8: "f32[16][1]cuda:0" = torch.ops.aten.add.Tensor(mul_9, mul_10);  mul_9 = mul_10 = None
        squeeze_5: "f32[16][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_11: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000001557019536);  squeeze_5 = None
        mul_12: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_11, 0.1);  mul_11 = None
        mul_13: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_11, 0.9)
        add_9: "f32[16][1]cuda:0" = torch.ops.aten.add.Tensor(mul_12, mul_13);  mul_12 = mul_13 = None
        unsqueeze_4: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_12, -1)
        unsqueeze_5: "f32[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_14: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(mul_8, unsqueeze_5);  mul_8 = unsqueeze_5 = None
        unsqueeze_6: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_13, -1);  primals_13 = None
        unsqueeze_7: "f32[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_10: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_7);  mul_14 = unsqueeze_7 = None
        convert_element_type_8: "bf16[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.prims.convert_element_type.default(add_10, torch.bfloat16);  add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu: "bf16[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.relu.default(convert_element_type_8);  convert_element_type_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:227 in forward, code: x = self.conv_pw(x)
        convert_element_type_9: "bf16[16, 16, 1, 1][16, 1, 16, 16]cuda:0" = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
        convolution_2: "bf16[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.convolution.default(relu, convert_element_type_9, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_11: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_15, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_10: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_2, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_10, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_10 = None
        getitem_4: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = var_mean_2[0]
        getitem_5: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        add_12: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
        rsqrt_2: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        sub_2: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
        mul_15: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        squeeze_6: "f32[16][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        squeeze_7: "f32[16][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_16: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
        mul_17: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_16, 0.9)
        add_13: "f32[16][1]cuda:0" = torch.ops.aten.add.Tensor(mul_16, mul_17);  mul_16 = mul_17 = None
        squeeze_8: "f32[16][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_18: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0000001557019536);  squeeze_8 = None
        mul_19: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_18, 0.1);  mul_18 = None
        mul_20: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_17, 0.9)
        add_14: "f32[16][1]cuda:0" = torch.ops.aten.add.Tensor(mul_19, mul_20);  mul_19 = mul_20 = None
        unsqueeze_8: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_18, -1)
        unsqueeze_9: "f32[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        mul_21: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(mul_15, unsqueeze_9);  mul_15 = unsqueeze_9 = None
        unsqueeze_10: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_19, -1);  primals_19 = None
        unsqueeze_11: "f32[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        add_15: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.add.Tensor(mul_21, unsqueeze_11);  mul_21 = unsqueeze_11 = None
        convert_element_type_11: "bf16[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.prims.convert_element_type.default(add_15, torch.bfloat16);  add_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:230 in forward, code: x = self.drop_path(x) + shortcut
        add_16: "bf16[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_11, convert_element_type_5);  convert_element_type_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convert_element_type_12: "bf16[64, 16, 1, 1][16, 1, 16, 16]cuda:0" = torch.ops.prims.convert_element_type.default(primals_20, torch.bfloat16);  primals_20 = None
        convolution_3: "bf16[512, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.convolution.default(add_16, convert_element_type_12, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_17: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_21, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_13: "f32[512, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_3, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_13, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_13 = None
        getitem_6: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_3[0]
        getitem_7: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        add_18: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
        rsqrt_3: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_3: "f32[512, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_3, getitem_7)
        mul_22: "f32[512, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        squeeze_9: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        squeeze_10: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_23: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
        mul_24: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_22, 0.9)
        add_19: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_23, mul_24);  mul_23 = mul_24 = None
        squeeze_11: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_25: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0000001557019536);  squeeze_11 = None
        mul_26: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_25, 0.1);  mul_25 = None
        mul_27: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_23, 0.9)
        add_20: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_26, mul_27);  mul_26 = mul_27 = None
        unsqueeze_12: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_24, -1)
        unsqueeze_13: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_28: "f32[512, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_13);  mul_22 = unsqueeze_13 = None
        unsqueeze_14: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_25, -1);  primals_25 = None
        unsqueeze_15: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_21: "f32[512, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_28, unsqueeze_15);  mul_28 = unsqueeze_15 = None
        convert_element_type_14: "bf16[512, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_21, torch.bfloat16);  add_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_1: "bf16[512, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.relu.default(convert_element_type_14);  convert_element_type_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convert_element_type_15: "bf16[64, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_26, torch.bfloat16);  primals_26 = None
        convolution_4: "bf16[512, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.convolution.default(relu_1, convert_element_type_15, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_22: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_27, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_16: "f32[512, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_4, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_16, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_16 = None
        getitem_8: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_4[0]
        getitem_9: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        add_23: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
        rsqrt_4: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        sub_4: "f32[512, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_4, getitem_9)
        mul_29: "f32[512, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        squeeze_12: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        squeeze_13: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_30: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
        mul_31: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_28, 0.9)
        add_24: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_30, mul_31);  mul_30 = mul_31 = None
        squeeze_14: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_32: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000006228081046);  squeeze_14 = None
        mul_33: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_32, 0.1);  mul_32 = None
        mul_34: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_29, 0.9)
        add_25: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_33, mul_34);  mul_33 = mul_34 = None
        unsqueeze_16: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_30, -1)
        unsqueeze_17: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        mul_35: "f32[512, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_29, unsqueeze_17);  mul_29 = unsqueeze_17 = None
        unsqueeze_18: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_31, -1);  primals_31 = None
        unsqueeze_19: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        add_26: "f32[512, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_35, unsqueeze_19);  mul_35 = unsqueeze_19 = None
        convert_element_type_17: "bf16[512, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_26, torch.bfloat16);  add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_2: "bf16[512, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.relu.default(convert_element_type_17);  convert_element_type_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convert_element_type_18: "bf16[24, 64, 1, 1][64, 1, 64, 64]cuda:0" = torch.ops.prims.convert_element_type.default(primals_32, torch.bfloat16);  primals_32 = None
        convolution_5: "bf16[512, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.convolution.default(relu_2, convert_element_type_18, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_27: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_33, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_19: "f32[512, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_5, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_19, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_19 = None
        getitem_10: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = var_mean_5[0]
        getitem_11: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        add_28: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
        rsqrt_5: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_5: "f32[512, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.sub.Tensor(convolution_5, getitem_11)
        mul_36: "f32[512, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        squeeze_15: "f32[24][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        squeeze_16: "f32[24][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_37: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
        mul_38: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_34, 0.9)
        add_29: "f32[24][1]cuda:0" = torch.ops.aten.add.Tensor(mul_37, mul_38);  mul_37 = mul_38 = None
        squeeze_17: "f32[24][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_39: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0000006228081046);  squeeze_17 = None
        mul_40: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_39, 0.1);  mul_39 = None
        mul_41: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_35, 0.9)
        add_30: "f32[24][1]cuda:0" = torch.ops.aten.add.Tensor(mul_40, mul_41);  mul_40 = mul_41 = None
        unsqueeze_20: "f32[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_36, -1)
        unsqueeze_21: "f32[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_42: "f32[512, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.mul.Tensor(mul_36, unsqueeze_21);  mul_36 = unsqueeze_21 = None
        unsqueeze_22: "f32[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_37, -1);  primals_37 = None
        unsqueeze_23: "f32[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_31: "f32[512, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.add.Tensor(mul_42, unsqueeze_23);  mul_42 = unsqueeze_23 = None
        convert_element_type_20: "bf16[512, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.prims.convert_element_type.default(add_31, torch.bfloat16);  add_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convert_element_type_21: "bf16[72, 24, 1, 1][24, 1, 24, 24]cuda:0" = torch.ops.prims.convert_element_type.default(primals_38, torch.bfloat16);  primals_38 = None
        convolution_6: "bf16[512, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_20, convert_element_type_21, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_32: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_39, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_22: "f32[512, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_6, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_22, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_22 = None
        getitem_12: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = var_mean_6[0]
        getitem_13: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        add_33: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
        rsqrt_6: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        sub_6: "f32[512, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.sub.Tensor(convolution_6, getitem_13)
        mul_43: "f32[512, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        squeeze_18: "f32[72][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        squeeze_19: "f32[72][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_44: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
        mul_45: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_40, 0.9)
        add_34: "f32[72][1]cuda:0" = torch.ops.aten.add.Tensor(mul_44, mul_45);  mul_44 = mul_45 = None
        squeeze_20: "f32[72][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
        mul_46: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0000006228081046);  squeeze_20 = None
        mul_47: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_46, 0.1);  mul_46 = None
        mul_48: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_41, 0.9)
        add_35: "f32[72][1]cuda:0" = torch.ops.aten.add.Tensor(mul_47, mul_48);  mul_47 = mul_48 = None
        unsqueeze_24: "f32[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_42, -1)
        unsqueeze_25: "f32[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        mul_49: "f32[512, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.mul.Tensor(mul_43, unsqueeze_25);  mul_43 = unsqueeze_25 = None
        unsqueeze_26: "f32[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_43, -1);  primals_43 = None
        unsqueeze_27: "f32[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        add_36: "f32[512, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.add.Tensor(mul_49, unsqueeze_27);  mul_49 = unsqueeze_27 = None
        convert_element_type_23: "bf16[512, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.prims.convert_element_type.default(add_36, torch.bfloat16);  add_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_3: "bf16[512, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.relu.default(convert_element_type_23);  convert_element_type_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convert_element_type_24: "bf16[72, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16);  primals_44 = None
        convolution_7: "bf16[512, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.convolution.default(relu_3, convert_element_type_24, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 72)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_37: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_45, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_25: "f32[512, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_7, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_25, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_25 = None
        getitem_14: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = var_mean_7[0]
        getitem_15: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        add_38: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
        rsqrt_7: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        sub_7: "f32[512, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.sub.Tensor(convolution_7, getitem_15)
        mul_50: "f32[512, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        squeeze_21: "f32[72][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        squeeze_22: "f32[72][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
        mul_51: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
        mul_52: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_46, 0.9)
        add_39: "f32[72][1]cuda:0" = torch.ops.aten.add.Tensor(mul_51, mul_52);  mul_51 = mul_52 = None
        squeeze_23: "f32[72][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
        mul_53: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0000006228081046);  squeeze_23 = None
        mul_54: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_53, 0.1);  mul_53 = None
        mul_55: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_47, 0.9)
        add_40: "f32[72][1]cuda:0" = torch.ops.aten.add.Tensor(mul_54, mul_55);  mul_54 = mul_55 = None
        unsqueeze_28: "f32[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_48, -1)
        unsqueeze_29: "f32[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_56: "f32[512, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.mul.Tensor(mul_50, unsqueeze_29);  mul_50 = unsqueeze_29 = None
        unsqueeze_30: "f32[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_49, -1);  primals_49 = None
        unsqueeze_31: "f32[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_41: "f32[512, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.add.Tensor(mul_56, unsqueeze_31);  mul_56 = unsqueeze_31 = None
        convert_element_type_26: "bf16[512, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.prims.convert_element_type.default(add_41, torch.bfloat16);  add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_4: "bf16[512, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.relu.default(convert_element_type_26);  convert_element_type_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convert_element_type_27: "bf16[24, 72, 1, 1][72, 1, 72, 72]cuda:0" = torch.ops.prims.convert_element_type.default(primals_50, torch.bfloat16);  primals_50 = None
        convolution_8: "bf16[512, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.convolution.default(relu_4, convert_element_type_27, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_42: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_51, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_28: "f32[512, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_8, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_28, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_28 = None
        getitem_16: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = var_mean_8[0]
        getitem_17: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        add_43: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
        rsqrt_8: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        sub_8: "f32[512, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.sub.Tensor(convolution_8, getitem_17)
        mul_57: "f32[512, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        squeeze_24: "f32[24][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
        squeeze_25: "f32[24][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_58: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
        mul_59: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_52, 0.9)
        add_44: "f32[24][1]cuda:0" = torch.ops.aten.add.Tensor(mul_58, mul_59);  mul_58 = mul_59 = None
        squeeze_26: "f32[24][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
        mul_60: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0000006228081046);  squeeze_26 = None
        mul_61: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_60, 0.1);  mul_60 = None
        mul_62: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_53, 0.9)
        add_45: "f32[24][1]cuda:0" = torch.ops.aten.add.Tensor(mul_61, mul_62);  mul_61 = mul_62 = None
        unsqueeze_32: "f32[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_54, -1)
        unsqueeze_33: "f32[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        mul_63: "f32[512, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.mul.Tensor(mul_57, unsqueeze_33);  mul_57 = unsqueeze_33 = None
        unsqueeze_34: "f32[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_55, -1);  primals_55 = None
        unsqueeze_35: "f32[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        add_46: "f32[512, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.add.Tensor(mul_63, unsqueeze_35);  mul_63 = unsqueeze_35 = None
        convert_element_type_29: "bf16[512, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.prims.convert_element_type.default(add_46, torch.bfloat16);  add_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:338 in forward, code: x = self.drop_path(x) + shortcut
        add_47: "bf16[512, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_29, convert_element_type_20);  convert_element_type_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convert_element_type_30: "bf16[72, 24, 1, 1][24, 1, 24, 24]cuda:0" = torch.ops.prims.convert_element_type.default(primals_56, torch.bfloat16);  primals_56 = None
        convolution_9: "bf16[512, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.convolution.default(add_47, convert_element_type_30, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_48: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_57, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_31: "f32[512, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_9, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_31, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_31 = None
        getitem_18: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = var_mean_9[0]
        getitem_19: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        add_49: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
        rsqrt_9: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        sub_9: "f32[512, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.sub.Tensor(convolution_9, getitem_19)
        mul_64: "f32[512, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        squeeze_27: "f32[72][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        squeeze_28: "f32[72][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_65: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
        mul_66: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_58, 0.9)
        add_50: "f32[72][1]cuda:0" = torch.ops.aten.add.Tensor(mul_65, mul_66);  mul_65 = mul_66 = None
        squeeze_29: "f32[72][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
        mul_67: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0000006228081046);  squeeze_29 = None
        mul_68: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_67, 0.1);  mul_67 = None
        mul_69: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_59, 0.9)
        add_51: "f32[72][1]cuda:0" = torch.ops.aten.add.Tensor(mul_68, mul_69);  mul_68 = mul_69 = None
        unsqueeze_36: "f32[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_60, -1)
        unsqueeze_37: "f32[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_70: "f32[512, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.mul.Tensor(mul_64, unsqueeze_37);  mul_64 = unsqueeze_37 = None
        unsqueeze_38: "f32[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_61, -1);  primals_61 = None
        unsqueeze_39: "f32[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_52: "f32[512, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.add.Tensor(mul_70, unsqueeze_39);  mul_70 = unsqueeze_39 = None
        convert_element_type_32: "bf16[512, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.prims.convert_element_type.default(add_52, torch.bfloat16);  add_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_5: "bf16[512, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.relu.default(convert_element_type_32);  convert_element_type_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convert_element_type_33: "bf16[72, 1, 5, 5][25, 1, 5, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_62, torch.bfloat16);  primals_62 = None
        convolution_10: "bf16[512, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.convolution.default(relu_5, convert_element_type_33, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 72)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_53: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_63, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_34: "f32[512, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_10, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_34, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_34 = None
        getitem_20: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = var_mean_10[0]
        getitem_21: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        add_54: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
        rsqrt_10: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        sub_10: "f32[512, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.sub.Tensor(convolution_10, getitem_21)
        mul_71: "f32[512, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        squeeze_30: "f32[72][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3])
        mul_72: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1);  squeeze_30 = None
        mul_73: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_64, 0.9)
        add_55: "f32[72][1]cuda:0" = torch.ops.aten.add.Tensor(mul_72, mul_73);  mul_72 = mul_73 = None
        squeeze_32: "f32[72][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        mul_74: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0000024912370735);  squeeze_32 = None
        mul_75: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_74, 0.1);  mul_74 = None
        mul_76: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_65, 0.9)
        add_56: "f32[72][1]cuda:0" = torch.ops.aten.add.Tensor(mul_75, mul_76);  mul_75 = mul_76 = None
        unsqueeze_40: "f32[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_66, -1)
        unsqueeze_41: "f32[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        mul_77: "f32[512, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.mul.Tensor(mul_71, unsqueeze_41);  mul_71 = unsqueeze_41 = None
        unsqueeze_42: "f32[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_67, -1)
        unsqueeze_43: "f32[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        add_57: "f32[512, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.add.Tensor(mul_77, unsqueeze_43);  mul_77 = unsqueeze_43 = None
        convert_element_type_35: "bf16[512, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.prims.convert_element_type.default(add_57, torch.bfloat16);  add_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_6: "bf16[512, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.relu.default(convert_element_type_35);  convert_element_type_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean: "bf16[512, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(relu_6, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convert_element_type_36: "bf16[24][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_69, torch.bfloat16);  primals_69 = None
        convert_element_type_37: "bf16[24, 72, 1, 1][72, 1, 72, 72]cuda:0" = torch.ops.prims.convert_element_type.default(primals_68, torch.bfloat16);  primals_68 = None
        convolution_11: "bf16[512, 24, 1, 1][24, 1, 24, 24]cuda:0" = torch.ops.aten.convolution.default(mean, convert_element_type_37, convert_element_type_36, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        relu_7: "bf16[512, 24, 1, 1][24, 1, 24, 24]cuda:0" = torch.ops.aten.relu.default(convolution_11);  convolution_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convert_element_type_38: "bf16[72][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_71, torch.bfloat16);  primals_71 = None
        convert_element_type_39: "bf16[72, 24, 1, 1][24, 1, 24, 24]cuda:0" = torch.ops.prims.convert_element_type.default(primals_70, torch.bfloat16);  primals_70 = None
        convolution_12: "bf16[512, 72, 1, 1][72, 1, 72, 72]cuda:0" = torch.ops.aten.convolution.default(relu_7, convert_element_type_39, convert_element_type_38, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        convert_element_type_40: "f32[512, 72, 1, 1][72, 1, 72, 72]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_12, torch.float32)
        add_58: "f32[512, 72, 1, 1][72, 1, 72, 72]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_40, 3);  convert_element_type_40 = None
        clamp_min_1: "f32[512, 72, 1, 1][72, 1, 72, 72]cuda:0" = torch.ops.aten.clamp_min.default(add_58, 0);  add_58 = None
        clamp_max_1: "f32[512, 72, 1, 1][72, 1, 72, 72]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_1, 6);  clamp_min_1 = None
        div_1: "f32[512, 72, 1, 1][72, 1, 72, 72]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_1, 6);  clamp_max_1 = None
        convert_element_type_41: "bf16[512, 72, 1, 1][72, 1, 72, 72]cuda:0" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        mul_78: "bf16[512, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.mul.Tensor(relu_6, convert_element_type_41);  relu_6 = convert_element_type_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convert_element_type_42: "bf16[40, 72, 1, 1][72, 1, 72, 72]cuda:0" = torch.ops.prims.convert_element_type.default(primals_72, torch.bfloat16);  primals_72 = None
        convolution_13: "bf16[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.convolution.default(mul_78, convert_element_type_42, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_59: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_73, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_43: "f32[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_13, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_43, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_43 = None
        getitem_22: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = var_mean_11[0]
        getitem_23: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        add_60: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
        rsqrt_11: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_11: "f32[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.sub.Tensor(convolution_13, getitem_23)
        mul_79: "f32[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        squeeze_33: "f32[40][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
        squeeze_34: "f32[40][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
        mul_80: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
        mul_81: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_74, 0.9)
        add_61: "f32[40][1]cuda:0" = torch.ops.aten.add.Tensor(mul_80, mul_81);  mul_80 = mul_81 = None
        squeeze_35: "f32[40][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
        mul_82: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0000024912370735);  squeeze_35 = None
        mul_83: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_82, 0.1);  mul_82 = None
        mul_84: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_75, 0.9)
        add_62: "f32[40][1]cuda:0" = torch.ops.aten.add.Tensor(mul_83, mul_84);  mul_83 = mul_84 = None
        unsqueeze_44: "f32[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_76, -1)
        unsqueeze_45: "f32[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_85: "f32[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.mul.Tensor(mul_79, unsqueeze_45);  mul_79 = unsqueeze_45 = None
        unsqueeze_46: "f32[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_77, -1);  primals_77 = None
        unsqueeze_47: "f32[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_63: "f32[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.add.Tensor(mul_85, unsqueeze_47);  mul_85 = unsqueeze_47 = None
        convert_element_type_44: "bf16[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.prims.convert_element_type.default(add_63, torch.bfloat16);  add_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convert_element_type_45: "bf16[120, 40, 1, 1][40, 1, 40, 40]cuda:0" = torch.ops.prims.convert_element_type.default(primals_78, torch.bfloat16);  primals_78 = None
        convolution_14: "bf16[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_44, convert_element_type_45, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_64: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_79, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_46: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_14, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_46, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_46 = None
        getitem_24: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = var_mean_12[0]
        getitem_25: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        add_65: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
        rsqrt_12: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        sub_12: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.sub.Tensor(convolution_14, getitem_25)
        mul_86: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        squeeze_36: "f32[120][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        squeeze_37: "f32[120][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_87: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
        mul_88: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_80, 0.9)
        add_66: "f32[120][1]cuda:0" = torch.ops.aten.add.Tensor(mul_87, mul_88);  mul_87 = mul_88 = None
        squeeze_38: "f32[120][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
        mul_89: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0000024912370735);  squeeze_38 = None
        mul_90: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_89, 0.1);  mul_89 = None
        mul_91: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_81, 0.9)
        add_67: "f32[120][1]cuda:0" = torch.ops.aten.add.Tensor(mul_90, mul_91);  mul_90 = mul_91 = None
        unsqueeze_48: "f32[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_82, -1)
        unsqueeze_49: "f32[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        mul_92: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(mul_86, unsqueeze_49);  mul_86 = unsqueeze_49 = None
        unsqueeze_50: "f32[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_83, -1);  primals_83 = None
        unsqueeze_51: "f32[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        add_68: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.add.Tensor(mul_92, unsqueeze_51);  mul_92 = unsqueeze_51 = None
        convert_element_type_47: "bf16[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.prims.convert_element_type.default(add_68, torch.bfloat16);  add_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_8: "bf16[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.relu.default(convert_element_type_47);  convert_element_type_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convert_element_type_48: "bf16[120, 1, 5, 5][25, 1, 5, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_84, torch.bfloat16);  primals_84 = None
        convolution_15: "bf16[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.convolution.default(relu_8, convert_element_type_48, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_69: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_85, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_49: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_15, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_49, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_49 = None
        getitem_26: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = var_mean_13[0]
        getitem_27: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        add_70: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
        rsqrt_13: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        sub_13: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.sub.Tensor(convolution_15, getitem_27)
        mul_93: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        squeeze_39: "f32[120][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3])
        mul_94: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1);  squeeze_39 = None
        mul_95: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_86, 0.9)
        add_71: "f32[120][1]cuda:0" = torch.ops.aten.add.Tensor(mul_94, mul_95);  mul_94 = mul_95 = None
        squeeze_41: "f32[120][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        mul_96: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0000024912370735);  squeeze_41 = None
        mul_97: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_96, 0.1);  mul_96 = None
        mul_98: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_87, 0.9)
        add_72: "f32[120][1]cuda:0" = torch.ops.aten.add.Tensor(mul_97, mul_98);  mul_97 = mul_98 = None
        unsqueeze_52: "f32[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_88, -1)
        unsqueeze_53: "f32[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_99: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(mul_93, unsqueeze_53);  mul_93 = unsqueeze_53 = None
        unsqueeze_54: "f32[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_89, -1)
        unsqueeze_55: "f32[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_73: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.add.Tensor(mul_99, unsqueeze_55);  mul_99 = unsqueeze_55 = None
        convert_element_type_50: "bf16[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.prims.convert_element_type.default(add_73, torch.bfloat16);  add_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_9: "bf16[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.relu.default(convert_element_type_50);  convert_element_type_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_1: "bf16[512, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(relu_9, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convert_element_type_51: "bf16[32][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_91, torch.bfloat16);  primals_91 = None
        convert_element_type_52: "bf16[32, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.prims.convert_element_type.default(primals_90, torch.bfloat16);  primals_90 = None
        convolution_16: "bf16[512, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.aten.convolution.default(mean_1, convert_element_type_52, convert_element_type_51, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        relu_10: "bf16[512, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.aten.relu.default(convolution_16);  convolution_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convert_element_type_53: "bf16[120][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_93, torch.bfloat16);  primals_93 = None
        convert_element_type_54: "bf16[120, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.prims.convert_element_type.default(primals_92, torch.bfloat16);  primals_92 = None
        convolution_17: "bf16[512, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.convolution.default(relu_10, convert_element_type_54, convert_element_type_53, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        convert_element_type_55: "f32[512, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_17, torch.float32)
        add_74: "f32[512, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_55, 3);  convert_element_type_55 = None
        clamp_min_2: "f32[512, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.clamp_min.default(add_74, 0);  add_74 = None
        clamp_max_2: "f32[512, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_2, 6);  clamp_min_2 = None
        div_2: "f32[512, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_2, 6);  clamp_max_2 = None
        convert_element_type_56: "bf16[512, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None
        mul_100: "bf16[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(relu_9, convert_element_type_56);  relu_9 = convert_element_type_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convert_element_type_57: "bf16[40, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.prims.convert_element_type.default(primals_94, torch.bfloat16);  primals_94 = None
        convolution_18: "bf16[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.convolution.default(mul_100, convert_element_type_57, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_75: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_95, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_58: "f32[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_18, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_58, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_58 = None
        getitem_28: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = var_mean_14[0]
        getitem_29: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        add_76: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
        rsqrt_14: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        sub_14: "f32[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.sub.Tensor(convolution_18, getitem_29)
        mul_101: "f32[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        squeeze_42: "f32[40][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        squeeze_43: "f32[40][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_102: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
        mul_103: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_96, 0.9)
        add_77: "f32[40][1]cuda:0" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
        squeeze_44: "f32[40][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        mul_104: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0000024912370735);  squeeze_44 = None
        mul_105: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_104, 0.1);  mul_104 = None
        mul_106: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_97, 0.9)
        add_78: "f32[40][1]cuda:0" = torch.ops.aten.add.Tensor(mul_105, mul_106);  mul_105 = mul_106 = None
        unsqueeze_56: "f32[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_98, -1)
        unsqueeze_57: "f32[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        mul_107: "f32[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.mul.Tensor(mul_101, unsqueeze_57);  mul_101 = unsqueeze_57 = None
        unsqueeze_58: "f32[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_99, -1);  primals_99 = None
        unsqueeze_59: "f32[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        add_79: "f32[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.add.Tensor(mul_107, unsqueeze_59);  mul_107 = unsqueeze_59 = None
        convert_element_type_59: "bf16[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.prims.convert_element_type.default(add_79, torch.bfloat16);  add_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:338 in forward, code: x = self.drop_path(x) + shortcut
        add_80: "bf16[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_59, convert_element_type_44);  convert_element_type_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convert_element_type_60: "bf16[120, 40, 1, 1][40, 1, 40, 40]cuda:0" = torch.ops.prims.convert_element_type.default(primals_100, torch.bfloat16);  primals_100 = None
        convolution_19: "bf16[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.convolution.default(add_80, convert_element_type_60, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_81: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_101, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_61: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_19, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_61, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_61 = None
        getitem_30: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = var_mean_15[0]
        getitem_31: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        add_82: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
        rsqrt_15: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_15: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.sub.Tensor(convolution_19, getitem_31)
        mul_108: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        squeeze_45: "f32[120][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
        squeeze_46: "f32[120][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_109: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
        mul_110: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_102, 0.9)
        add_83: "f32[120][1]cuda:0" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
        squeeze_47: "f32[120][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
        mul_111: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0000024912370735);  squeeze_47 = None
        mul_112: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_111, 0.1);  mul_111 = None
        mul_113: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_103, 0.9)
        add_84: "f32[120][1]cuda:0" = torch.ops.aten.add.Tensor(mul_112, mul_113);  mul_112 = mul_113 = None
        unsqueeze_60: "f32[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_104, -1)
        unsqueeze_61: "f32[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_114: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(mul_108, unsqueeze_61);  mul_108 = unsqueeze_61 = None
        unsqueeze_62: "f32[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_105, -1);  primals_105 = None
        unsqueeze_63: "f32[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_85: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.add.Tensor(mul_114, unsqueeze_63);  mul_114 = unsqueeze_63 = None
        convert_element_type_62: "bf16[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.prims.convert_element_type.default(add_85, torch.bfloat16);  add_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_11: "bf16[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.relu.default(convert_element_type_62);  convert_element_type_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convert_element_type_63: "bf16[120, 1, 5, 5][25, 1, 5, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_106, torch.bfloat16);  primals_106 = None
        convolution_20: "bf16[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.convolution.default(relu_11, convert_element_type_63, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_86: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_107, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_64: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_20, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_64, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_64 = None
        getitem_32: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = var_mean_16[0]
        getitem_33: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        add_87: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
        rsqrt_16: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
        sub_16: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.sub.Tensor(convolution_20, getitem_33)
        mul_115: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        squeeze_48: "f32[120][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3])
        mul_116: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1);  squeeze_48 = None
        mul_117: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_108, 0.9)
        add_88: "f32[120][1]cuda:0" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
        squeeze_50: "f32[120][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
        mul_118: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0000024912370735);  squeeze_50 = None
        mul_119: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_118, 0.1);  mul_118 = None
        mul_120: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_109, 0.9)
        add_89: "f32[120][1]cuda:0" = torch.ops.aten.add.Tensor(mul_119, mul_120);  mul_119 = mul_120 = None
        unsqueeze_64: "f32[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_110, -1)
        unsqueeze_65: "f32[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        mul_121: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(mul_115, unsqueeze_65);  mul_115 = unsqueeze_65 = None
        unsqueeze_66: "f32[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_111, -1)
        unsqueeze_67: "f32[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        add_90: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.add.Tensor(mul_121, unsqueeze_67);  mul_121 = unsqueeze_67 = None
        convert_element_type_65: "bf16[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.prims.convert_element_type.default(add_90, torch.bfloat16);  add_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_12: "bf16[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.relu.default(convert_element_type_65);  convert_element_type_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_2: "bf16[512, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(relu_12, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convert_element_type_66: "bf16[32][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_113, torch.bfloat16);  primals_113 = None
        convert_element_type_67: "bf16[32, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.prims.convert_element_type.default(primals_112, torch.bfloat16);  primals_112 = None
        convolution_21: "bf16[512, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.aten.convolution.default(mean_2, convert_element_type_67, convert_element_type_66, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        relu_13: "bf16[512, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.aten.relu.default(convolution_21);  convolution_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convert_element_type_68: "bf16[120][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_115, torch.bfloat16);  primals_115 = None
        convert_element_type_69: "bf16[120, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.prims.convert_element_type.default(primals_114, torch.bfloat16);  primals_114 = None
        convolution_22: "bf16[512, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.convolution.default(relu_13, convert_element_type_69, convert_element_type_68, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        convert_element_type_70: "f32[512, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_22, torch.float32)
        add_91: "f32[512, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_70, 3);  convert_element_type_70 = None
        clamp_min_3: "f32[512, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.clamp_min.default(add_91, 0);  add_91 = None
        clamp_max_3: "f32[512, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_3, 6);  clamp_min_3 = None
        div_3: "f32[512, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_3, 6);  clamp_max_3 = None
        convert_element_type_71: "bf16[512, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None
        mul_122: "bf16[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(relu_12, convert_element_type_71);  relu_12 = convert_element_type_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convert_element_type_72: "bf16[40, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.prims.convert_element_type.default(primals_116, torch.bfloat16);  primals_116 = None
        convolution_23: "bf16[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.convolution.default(mul_122, convert_element_type_72, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_92: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_117, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_73: "f32[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_23, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_73, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_73 = None
        getitem_34: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = var_mean_17[0]
        getitem_35: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        add_93: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
        rsqrt_17: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        sub_17: "f32[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.sub.Tensor(convolution_23, getitem_35)
        mul_123: "f32[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        squeeze_51: "f32[40][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
        squeeze_52: "f32[40][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
        mul_124: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
        mul_125: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_118, 0.9)
        add_94: "f32[40][1]cuda:0" = torch.ops.aten.add.Tensor(mul_124, mul_125);  mul_124 = mul_125 = None
        squeeze_53: "f32[40][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
        mul_126: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0000024912370735);  squeeze_53 = None
        mul_127: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_126, 0.1);  mul_126 = None
        mul_128: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_119, 0.9)
        add_95: "f32[40][1]cuda:0" = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
        unsqueeze_68: "f32[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_120, -1)
        unsqueeze_69: "f32[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_129: "f32[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.mul.Tensor(mul_123, unsqueeze_69);  mul_123 = unsqueeze_69 = None
        unsqueeze_70: "f32[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_121, -1);  primals_121 = None
        unsqueeze_71: "f32[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_96: "f32[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.add.Tensor(mul_129, unsqueeze_71);  mul_129 = unsqueeze_71 = None
        convert_element_type_74: "bf16[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.prims.convert_element_type.default(add_96, torch.bfloat16);  add_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:338 in forward, code: x = self.drop_path(x) + shortcut
        add_97: "bf16[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_74, add_80);  convert_element_type_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convert_element_type_75: "bf16[240, 40, 1, 1][40, 1, 40, 40]cuda:0" = torch.ops.prims.convert_element_type.default(primals_122, torch.bfloat16);  primals_122 = None
        convolution_24: "bf16[512, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.convolution.default(add_97, convert_element_type_75, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_98: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_123, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_76: "f32[512, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_24, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_76, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_76 = None
        getitem_36: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = var_mean_18[0]
        getitem_37: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        add_99: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
        rsqrt_18: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
        sub_18: "f32[512, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.sub.Tensor(convolution_24, getitem_37)
        mul_130: "f32[512, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        squeeze_54: "f32[240][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3])
        mul_131: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1);  squeeze_54 = None
        mul_132: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_124, 0.9)
        add_100: "f32[240][1]cuda:0" = torch.ops.aten.add.Tensor(mul_131, mul_132);  mul_131 = mul_132 = None
        squeeze_56: "f32[240][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
        mul_133: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0000024912370735);  squeeze_56 = None
        mul_134: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_133, 0.1);  mul_133 = None
        mul_135: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_125, 0.9)
        add_101: "f32[240][1]cuda:0" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
        unsqueeze_72: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_126, -1)
        unsqueeze_73: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        mul_136: "f32[512, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(mul_130, unsqueeze_73);  mul_130 = unsqueeze_73 = None
        unsqueeze_74: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_127, -1)
        unsqueeze_75: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        add_102: "f32[512, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.add.Tensor(mul_136, unsqueeze_75);  mul_136 = unsqueeze_75 = None
        convert_element_type_77: "bf16[512, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(add_102, torch.bfloat16);  add_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_78: "f32[512, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_77, torch.float32);  convert_element_type_77 = None
        add_103: "f32[512, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_78, 3)
        clamp_min_4: "f32[512, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.clamp_min.default(add_103, 0);  add_103 = None
        clamp_max_4: "f32[512, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_4, 6);  clamp_min_4 = None
        mul_137: "f32[512, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_78, clamp_max_4);  convert_element_type_78 = clamp_max_4 = None
        div_4: "f32[512, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.div.Tensor(mul_137, 6);  mul_137 = None
        convert_element_type_79: "bf16[512, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convert_element_type_80: "bf16[240, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_128, torch.bfloat16);  primals_128 = None
        convolution_25: "bf16[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_79, convert_element_type_80, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 240)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_104: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_129, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_81: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_25, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_81, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_81 = None
        getitem_38: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = var_mean_19[0]
        getitem_39: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        add_105: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_38, 1e-05)
        rsqrt_19: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
        sub_19: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.sub.Tensor(convolution_25, getitem_39)
        mul_138: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        squeeze_57: "f32[240][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3])
        mul_139: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1);  squeeze_57 = None
        mul_140: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_130, 0.9)
        add_106: "f32[240][1]cuda:0" = torch.ops.aten.add.Tensor(mul_139, mul_140);  mul_139 = mul_140 = None
        squeeze_59: "f32[240][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
        mul_141: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_59, 1.00000996502277);  squeeze_59 = None
        mul_142: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_141, 0.1);  mul_141 = None
        mul_143: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_131, 0.9)
        add_107: "f32[240][1]cuda:0" = torch.ops.aten.add.Tensor(mul_142, mul_143);  mul_142 = mul_143 = None
        unsqueeze_76: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_132, -1)
        unsqueeze_77: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_144: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(mul_138, unsqueeze_77);  mul_138 = unsqueeze_77 = None
        unsqueeze_78: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_133, -1)
        unsqueeze_79: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_108: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.add.Tensor(mul_144, unsqueeze_79);  mul_144 = unsqueeze_79 = None
        convert_element_type_82: "bf16[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.prims.convert_element_type.default(add_108, torch.bfloat16);  add_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_83: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_82, torch.float32);  convert_element_type_82 = None
        add_109: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_83, 3)
        clamp_min_5: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.clamp_min.default(add_109, 0);  add_109 = None
        clamp_max_5: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_5, 6);  clamp_min_5 = None
        mul_145: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_83, clamp_max_5);  convert_element_type_83 = clamp_max_5 = None
        div_5: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.div.Tensor(mul_145, 6);  mul_145 = None
        convert_element_type_84: "bf16[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convert_element_type_85: "bf16[80, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.prims.convert_element_type.default(primals_134, torch.bfloat16);  primals_134 = None
        convolution_26: "bf16[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_84, convert_element_type_85, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_110: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_135, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_86: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_26, torch.float32)
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_86, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_86 = None
        getitem_40: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = var_mean_20[0]
        getitem_41: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None
        add_111: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_40, 1e-05)
        rsqrt_20: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
        sub_20: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.sub.Tensor(convolution_26, getitem_41)
        mul_146: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
        squeeze_60: "f32[80][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
        squeeze_61: "f32[80][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
        mul_147: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
        mul_148: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_136, 0.9)
        add_112: "f32[80][1]cuda:0" = torch.ops.aten.add.Tensor(mul_147, mul_148);  mul_147 = mul_148 = None
        squeeze_62: "f32[80][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
        mul_149: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_62, 1.00000996502277);  squeeze_62 = None
        mul_150: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_149, 0.1);  mul_149 = None
        mul_151: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_137, 0.9)
        add_113: "f32[80][1]cuda:0" = torch.ops.aten.add.Tensor(mul_150, mul_151);  mul_150 = mul_151 = None
        unsqueeze_80: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_138, -1)
        unsqueeze_81: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        mul_152: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(mul_146, unsqueeze_81);  mul_146 = unsqueeze_81 = None
        unsqueeze_82: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_139, -1);  primals_139 = None
        unsqueeze_83: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        add_114: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.add.Tensor(mul_152, unsqueeze_83);  mul_152 = unsqueeze_83 = None
        convert_element_type_87: "bf16[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(add_114, torch.bfloat16);  add_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convert_element_type_88: "bf16[200, 80, 1, 1][80, 1, 80, 80]cuda:0" = torch.ops.prims.convert_element_type.default(primals_140, torch.bfloat16);  primals_140 = None
        convolution_27: "bf16[512, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_87, convert_element_type_88, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_115: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_141, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_89: "f32[512, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_27, torch.float32)
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_89, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_89 = None
        getitem_42: "f32[1, 200, 1, 1][200, 1, 1, 1]cuda:0" = var_mean_21[0]
        getitem_43: "f32[1, 200, 1, 1][200, 1, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        add_116: "f32[1, 200, 1, 1][200, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_42, 1e-05)
        rsqrt_21: "f32[1, 200, 1, 1][200, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        sub_21: "f32[512, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.sub.Tensor(convolution_27, getitem_43)
        mul_153: "f32[512, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
        squeeze_63: "f32[200][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3])
        mul_154: "f32[200][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1);  squeeze_63 = None
        mul_155: "f32[200][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_142, 0.9)
        add_117: "f32[200][1]cuda:0" = torch.ops.aten.add.Tensor(mul_154, mul_155);  mul_154 = mul_155 = None
        squeeze_65: "f32[200][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
        mul_156: "f32[200][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_65, 1.00000996502277);  squeeze_65 = None
        mul_157: "f32[200][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_156, 0.1);  mul_156 = None
        mul_158: "f32[200][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_143, 0.9)
        add_118: "f32[200][1]cuda:0" = torch.ops.aten.add.Tensor(mul_157, mul_158);  mul_157 = mul_158 = None
        unsqueeze_84: "f32[200, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_144, -1)
        unsqueeze_85: "f32[200, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_159: "f32[512, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.mul.Tensor(mul_153, unsqueeze_85);  mul_153 = unsqueeze_85 = None
        unsqueeze_86: "f32[200, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_145, -1)
        unsqueeze_87: "f32[200, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_119: "f32[512, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.add.Tensor(mul_159, unsqueeze_87);  mul_159 = unsqueeze_87 = None
        convert_element_type_90: "bf16[512, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.prims.convert_element_type.default(add_119, torch.bfloat16);  add_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_91: "f32[512, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_90, torch.float32);  convert_element_type_90 = None
        add_120: "f32[512, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_91, 3)
        clamp_min_6: "f32[512, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.clamp_min.default(add_120, 0);  add_120 = None
        clamp_max_6: "f32[512, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_6, 6);  clamp_min_6 = None
        mul_160: "f32[512, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_91, clamp_max_6);  convert_element_type_91 = clamp_max_6 = None
        div_6: "f32[512, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.div.Tensor(mul_160, 6);  mul_160 = None
        convert_element_type_92: "bf16[512, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convert_element_type_93: "bf16[200, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_146, torch.bfloat16);  primals_146 = None
        convolution_28: "bf16[512, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_92, convert_element_type_93, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 200)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_121: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_147, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_94: "f32[512, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_28, torch.float32)
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_94, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_94 = None
        getitem_44: "f32[1, 200, 1, 1][200, 1, 1, 1]cuda:0" = var_mean_22[0]
        getitem_45: "f32[1, 200, 1, 1][200, 1, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None
        add_122: "f32[1, 200, 1, 1][200, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_44, 1e-05)
        rsqrt_22: "f32[1, 200, 1, 1][200, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        sub_22: "f32[512, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.sub.Tensor(convolution_28, getitem_45)
        mul_161: "f32[512, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        squeeze_66: "f32[200][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3])
        mul_162: "f32[200][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1);  squeeze_66 = None
        mul_163: "f32[200][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_148, 0.9)
        add_123: "f32[200][1]cuda:0" = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
        squeeze_68: "f32[200][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
        mul_164: "f32[200][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_68, 1.00000996502277);  squeeze_68 = None
        mul_165: "f32[200][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_164, 0.1);  mul_164 = None
        mul_166: "f32[200][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_149, 0.9)
        add_124: "f32[200][1]cuda:0" = torch.ops.aten.add.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
        unsqueeze_88: "f32[200, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_150, -1)
        unsqueeze_89: "f32[200, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        mul_167: "f32[512, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_89);  mul_161 = unsqueeze_89 = None
        unsqueeze_90: "f32[200, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_151, -1)
        unsqueeze_91: "f32[200, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        add_125: "f32[512, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_91);  mul_167 = unsqueeze_91 = None
        convert_element_type_95: "bf16[512, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.prims.convert_element_type.default(add_125, torch.bfloat16);  add_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_96: "f32[512, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_95, torch.float32);  convert_element_type_95 = None
        add_126: "f32[512, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_96, 3)
        clamp_min_7: "f32[512, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.clamp_min.default(add_126, 0);  add_126 = None
        clamp_max_7: "f32[512, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_7, 6);  clamp_min_7 = None
        mul_168: "f32[512, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_96, clamp_max_7);  convert_element_type_96 = clamp_max_7 = None
        div_7: "f32[512, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.div.Tensor(mul_168, 6);  mul_168 = None
        convert_element_type_97: "bf16[512, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convert_element_type_98: "bf16[80, 200, 1, 1][200, 1, 200, 200]cuda:0" = torch.ops.prims.convert_element_type.default(primals_152, torch.bfloat16);  primals_152 = None
        convolution_29: "bf16[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_97, convert_element_type_98, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_127: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_153, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_99: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_29, torch.float32)
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_99, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_99 = None
        getitem_46: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = var_mean_23[0]
        getitem_47: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        add_128: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
        rsqrt_23: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
        sub_23: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.sub.Tensor(convolution_29, getitem_47)
        mul_169: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
        squeeze_69: "f32[80][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
        squeeze_70: "f32[80][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
        mul_170: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
        mul_171: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_154, 0.9)
        add_129: "f32[80][1]cuda:0" = torch.ops.aten.add.Tensor(mul_170, mul_171);  mul_170 = mul_171 = None
        squeeze_71: "f32[80][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
        mul_172: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_71, 1.00000996502277);  squeeze_71 = None
        mul_173: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_172, 0.1);  mul_172 = None
        mul_174: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_155, 0.9)
        add_130: "f32[80][1]cuda:0" = torch.ops.aten.add.Tensor(mul_173, mul_174);  mul_173 = mul_174 = None
        unsqueeze_92: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_156, -1)
        unsqueeze_93: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        mul_175: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(mul_169, unsqueeze_93);  mul_169 = unsqueeze_93 = None
        unsqueeze_94: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_157, -1);  primals_157 = None
        unsqueeze_95: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        add_131: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.add.Tensor(mul_175, unsqueeze_95);  mul_175 = unsqueeze_95 = None
        convert_element_type_100: "bf16[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(add_131, torch.bfloat16);  add_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:338 in forward, code: x = self.drop_path(x) + shortcut
        add_132: "bf16[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_100, convert_element_type_87);  convert_element_type_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convert_element_type_101: "bf16[184, 80, 1, 1][80, 1, 80, 80]cuda:0" = torch.ops.prims.convert_element_type.default(primals_158, torch.bfloat16);  primals_158 = None
        convolution_30: "bf16[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.convolution.default(add_132, convert_element_type_101, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_133: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_159, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_102: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_30, torch.float32)
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_102, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_102 = None
        getitem_48: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = var_mean_24[0]
        getitem_49: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None
        add_134: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_48, 1e-05)
        rsqrt_24: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        sub_24: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.sub.Tensor(convolution_30, getitem_49)
        mul_176: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        squeeze_72: "f32[184][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3])
        mul_177: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1);  squeeze_72 = None
        mul_178: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_160, 0.9)
        add_135: "f32[184][1]cuda:0" = torch.ops.aten.add.Tensor(mul_177, mul_178);  mul_177 = mul_178 = None
        squeeze_74: "f32[184][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
        mul_179: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_74, 1.00000996502277);  squeeze_74 = None
        mul_180: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_179, 0.1);  mul_179 = None
        mul_181: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_161, 0.9)
        add_136: "f32[184][1]cuda:0" = torch.ops.aten.add.Tensor(mul_180, mul_181);  mul_180 = mul_181 = None
        unsqueeze_96: "f32[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_162, -1)
        unsqueeze_97: "f32[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        mul_182: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(mul_176, unsqueeze_97);  mul_176 = unsqueeze_97 = None
        unsqueeze_98: "f32[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_163, -1)
        unsqueeze_99: "f32[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        add_137: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.add.Tensor(mul_182, unsqueeze_99);  mul_182 = unsqueeze_99 = None
        convert_element_type_103: "bf16[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(add_137, torch.bfloat16);  add_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_104: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_103, torch.float32);  convert_element_type_103 = None
        add_138: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_104, 3)
        clamp_min_8: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.clamp_min.default(add_138, 0);  add_138 = None
        clamp_max_8: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_8, 6);  clamp_min_8 = None
        mul_183: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_104, clamp_max_8);  convert_element_type_104 = clamp_max_8 = None
        div_8: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.div.Tensor(mul_183, 6);  mul_183 = None
        convert_element_type_105: "bf16[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convert_element_type_106: "bf16[184, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_164, torch.bfloat16);  primals_164 = None
        convolution_31: "bf16[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_105, convert_element_type_106, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 184)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_139: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_165, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_107: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_31, torch.float32)
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_107, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_107 = None
        getitem_50: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = var_mean_25[0]
        getitem_51: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = var_mean_25[1];  var_mean_25 = None
        add_140: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_50, 1e-05)
        rsqrt_25: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
        sub_25: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.sub.Tensor(convolution_31, getitem_51)
        mul_184: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        squeeze_75: "f32[184][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3])
        mul_185: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1);  squeeze_75 = None
        mul_186: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_166, 0.9)
        add_141: "f32[184][1]cuda:0" = torch.ops.aten.add.Tensor(mul_185, mul_186);  mul_185 = mul_186 = None
        squeeze_77: "f32[184][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
        mul_187: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_77, 1.00000996502277);  squeeze_77 = None
        mul_188: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_187, 0.1);  mul_187 = None
        mul_189: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_167, 0.9)
        add_142: "f32[184][1]cuda:0" = torch.ops.aten.add.Tensor(mul_188, mul_189);  mul_188 = mul_189 = None
        unsqueeze_100: "f32[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_168, -1)
        unsqueeze_101: "f32[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_190: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(mul_184, unsqueeze_101);  mul_184 = unsqueeze_101 = None
        unsqueeze_102: "f32[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_169, -1)
        unsqueeze_103: "f32[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_143: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.add.Tensor(mul_190, unsqueeze_103);  mul_190 = unsqueeze_103 = None
        convert_element_type_108: "bf16[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(add_143, torch.bfloat16);  add_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_109: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_108, torch.float32);  convert_element_type_108 = None
        add_144: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_109, 3)
        clamp_min_9: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.clamp_min.default(add_144, 0);  add_144 = None
        clamp_max_9: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_9, 6);  clamp_min_9 = None
        mul_191: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_109, clamp_max_9);  convert_element_type_109 = clamp_max_9 = None
        div_9: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.div.Tensor(mul_191, 6);  mul_191 = None
        convert_element_type_110: "bf16[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convert_element_type_111: "bf16[80, 184, 1, 1][184, 1, 184, 184]cuda:0" = torch.ops.prims.convert_element_type.default(primals_170, torch.bfloat16);  primals_170 = None
        convolution_32: "bf16[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_110, convert_element_type_111, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_145: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_171, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_112: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_32, torch.float32)
        var_mean_26 = torch.ops.aten.var_mean.correction(convert_element_type_112, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_112 = None
        getitem_52: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = var_mean_26[0]
        getitem_53: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = var_mean_26[1];  var_mean_26 = None
        add_146: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_52, 1e-05)
        rsqrt_26: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
        sub_26: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.sub.Tensor(convolution_32, getitem_53)
        mul_192: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
        squeeze_78: "f32[80][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
        squeeze_79: "f32[80][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
        mul_193: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
        mul_194: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_172, 0.9)
        add_147: "f32[80][1]cuda:0" = torch.ops.aten.add.Tensor(mul_193, mul_194);  mul_193 = mul_194 = None
        squeeze_80: "f32[80][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
        mul_195: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_80, 1.00000996502277);  squeeze_80 = None
        mul_196: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_195, 0.1);  mul_195 = None
        mul_197: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_173, 0.9)
        add_148: "f32[80][1]cuda:0" = torch.ops.aten.add.Tensor(mul_196, mul_197);  mul_196 = mul_197 = None
        unsqueeze_104: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_174, -1)
        unsqueeze_105: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        mul_198: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(mul_192, unsqueeze_105);  mul_192 = unsqueeze_105 = None
        unsqueeze_106: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_175, -1);  primals_175 = None
        unsqueeze_107: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
        add_149: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.add.Tensor(mul_198, unsqueeze_107);  mul_198 = unsqueeze_107 = None
        convert_element_type_113: "bf16[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(add_149, torch.bfloat16);  add_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:338 in forward, code: x = self.drop_path(x) + shortcut
        add_150: "bf16[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_113, add_132);  convert_element_type_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convert_element_type_114: "bf16[184, 80, 1, 1][80, 1, 80, 80]cuda:0" = torch.ops.prims.convert_element_type.default(primals_176, torch.bfloat16);  primals_176 = None
        convolution_33: "bf16[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.convolution.default(add_150, convert_element_type_114, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_151: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_177, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_115: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_33, torch.float32)
        var_mean_27 = torch.ops.aten.var_mean.correction(convert_element_type_115, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_115 = None
        getitem_54: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = var_mean_27[0]
        getitem_55: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = var_mean_27[1];  var_mean_27 = None
        add_152: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_54, 1e-05)
        rsqrt_27: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
        sub_27: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.sub.Tensor(convolution_33, getitem_55)
        mul_199: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
        squeeze_81: "f32[184][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3])
        mul_200: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1);  squeeze_81 = None
        mul_201: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_178, 0.9)
        add_153: "f32[184][1]cuda:0" = torch.ops.aten.add.Tensor(mul_200, mul_201);  mul_200 = mul_201 = None
        squeeze_83: "f32[184][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
        mul_202: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_83, 1.00000996502277);  squeeze_83 = None
        mul_203: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_202, 0.1);  mul_202 = None
        mul_204: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_179, 0.9)
        add_154: "f32[184][1]cuda:0" = torch.ops.aten.add.Tensor(mul_203, mul_204);  mul_203 = mul_204 = None
        unsqueeze_108: "f32[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_180, -1)
        unsqueeze_109: "f32[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
        mul_205: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(mul_199, unsqueeze_109);  mul_199 = unsqueeze_109 = None
        unsqueeze_110: "f32[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_181, -1)
        unsqueeze_111: "f32[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
        add_155: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.add.Tensor(mul_205, unsqueeze_111);  mul_205 = unsqueeze_111 = None
        convert_element_type_116: "bf16[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(add_155, torch.bfloat16);  add_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_117: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_116, torch.float32);  convert_element_type_116 = None
        add_156: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_117, 3)
        clamp_min_10: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.clamp_min.default(add_156, 0);  add_156 = None
        clamp_max_10: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_10, 6);  clamp_min_10 = None
        mul_206: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_117, clamp_max_10);  convert_element_type_117 = clamp_max_10 = None
        div_10: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.div.Tensor(mul_206, 6);  mul_206 = None
        convert_element_type_118: "bf16[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convert_element_type_119: "bf16[184, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_182, torch.bfloat16);  primals_182 = None
        convolution_34: "bf16[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_118, convert_element_type_119, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 184)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_157: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_183, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_120: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_34, torch.float32)
        var_mean_28 = torch.ops.aten.var_mean.correction(convert_element_type_120, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_120 = None
        getitem_56: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = var_mean_28[0]
        getitem_57: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = var_mean_28[1];  var_mean_28 = None
        add_158: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_56, 1e-05)
        rsqrt_28: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
        sub_28: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.sub.Tensor(convolution_34, getitem_57)
        mul_207: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
        squeeze_84: "f32[184][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3])
        mul_208: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1);  squeeze_84 = None
        mul_209: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_184, 0.9)
        add_159: "f32[184][1]cuda:0" = torch.ops.aten.add.Tensor(mul_208, mul_209);  mul_208 = mul_209 = None
        squeeze_86: "f32[184][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
        mul_210: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_86, 1.00000996502277);  squeeze_86 = None
        mul_211: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_210, 0.1);  mul_210 = None
        mul_212: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_185, 0.9)
        add_160: "f32[184][1]cuda:0" = torch.ops.aten.add.Tensor(mul_211, mul_212);  mul_211 = mul_212 = None
        unsqueeze_112: "f32[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_186, -1)
        unsqueeze_113: "f32[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        mul_213: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(mul_207, unsqueeze_113);  mul_207 = unsqueeze_113 = None
        unsqueeze_114: "f32[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_187, -1)
        unsqueeze_115: "f32[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        add_161: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.add.Tensor(mul_213, unsqueeze_115);  mul_213 = unsqueeze_115 = None
        convert_element_type_121: "bf16[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(add_161, torch.bfloat16);  add_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_122: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_121, torch.float32);  convert_element_type_121 = None
        add_162: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_122, 3)
        clamp_min_11: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.clamp_min.default(add_162, 0);  add_162 = None
        clamp_max_11: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_11, 6);  clamp_min_11 = None
        mul_214: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_122, clamp_max_11);  convert_element_type_122 = clamp_max_11 = None
        div_11: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.div.Tensor(mul_214, 6);  mul_214 = None
        convert_element_type_123: "bf16[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convert_element_type_124: "bf16[80, 184, 1, 1][184, 1, 184, 184]cuda:0" = torch.ops.prims.convert_element_type.default(primals_188, torch.bfloat16);  primals_188 = None
        convolution_35: "bf16[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_123, convert_element_type_124, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_163: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_189, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_125: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_35, torch.float32)
        var_mean_29 = torch.ops.aten.var_mean.correction(convert_element_type_125, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_125 = None
        getitem_58: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = var_mean_29[0]
        getitem_59: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = var_mean_29[1];  var_mean_29 = None
        add_164: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_58, 1e-05)
        rsqrt_29: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
        sub_29: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.sub.Tensor(convolution_35, getitem_59)
        mul_215: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
        squeeze_87: "f32[80][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
        squeeze_88: "f32[80][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
        mul_216: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
        mul_217: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_190, 0.9)
        add_165: "f32[80][1]cuda:0" = torch.ops.aten.add.Tensor(mul_216, mul_217);  mul_216 = mul_217 = None
        squeeze_89: "f32[80][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
        mul_218: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_89, 1.00000996502277);  squeeze_89 = None
        mul_219: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_218, 0.1);  mul_218 = None
        mul_220: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_191, 0.9)
        add_166: "f32[80][1]cuda:0" = torch.ops.aten.add.Tensor(mul_219, mul_220);  mul_219 = mul_220 = None
        unsqueeze_116: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_192, -1)
        unsqueeze_117: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        mul_221: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(mul_215, unsqueeze_117);  mul_215 = unsqueeze_117 = None
        unsqueeze_118: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_193, -1);  primals_193 = None
        unsqueeze_119: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
        add_167: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.add.Tensor(mul_221, unsqueeze_119);  mul_221 = unsqueeze_119 = None
        convert_element_type_126: "bf16[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(add_167, torch.bfloat16);  add_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:338 in forward, code: x = self.drop_path(x) + shortcut
        add_168: "bf16[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_126, add_150);  convert_element_type_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convert_element_type_127: "bf16[480, 80, 1, 1][80, 1, 80, 80]cuda:0" = torch.ops.prims.convert_element_type.default(primals_194, torch.bfloat16);  primals_194 = None
        convolution_36: "bf16[512, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.convolution.default(add_168, convert_element_type_127, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_169: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_195, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_128: "f32[512, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_36, torch.float32)
        var_mean_30 = torch.ops.aten.var_mean.correction(convert_element_type_128, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_128 = None
        getitem_60: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = var_mean_30[0]
        getitem_61: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = var_mean_30[1];  var_mean_30 = None
        add_170: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_60, 1e-05)
        rsqrt_30: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        sub_30: "f32[512, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convolution_36, getitem_61)
        mul_222: "f32[512, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
        squeeze_90: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3])
        mul_223: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1);  squeeze_90 = None
        mul_224: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_196, 0.9)
        add_171: "f32[480][1]cuda:0" = torch.ops.aten.add.Tensor(mul_223, mul_224);  mul_223 = mul_224 = None
        squeeze_92: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
        mul_225: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_92, 1.00000996502277);  squeeze_92 = None
        mul_226: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_225, 0.1);  mul_225 = None
        mul_227: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_197, 0.9)
        add_172: "f32[480][1]cuda:0" = torch.ops.aten.add.Tensor(mul_226, mul_227);  mul_226 = mul_227 = None
        unsqueeze_120: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_198, -1)
        unsqueeze_121: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        mul_228: "f32[512, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(mul_222, unsqueeze_121);  mul_222 = unsqueeze_121 = None
        unsqueeze_122: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_199, -1)
        unsqueeze_123: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        add_173: "f32[512, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_228, unsqueeze_123);  mul_228 = unsqueeze_123 = None
        convert_element_type_129: "bf16[512, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(add_173, torch.bfloat16);  add_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_130: "f32[512, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_129, torch.float32);  convert_element_type_129 = None
        add_174: "f32[512, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_130, 3)
        clamp_min_12: "f32[512, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.clamp_min.default(add_174, 0);  add_174 = None
        clamp_max_12: "f32[512, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_12, 6);  clamp_min_12 = None
        mul_229: "f32[512, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_130, clamp_max_12);  convert_element_type_130 = clamp_max_12 = None
        div_12: "f32[512, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.div.Tensor(mul_229, 6);  mul_229 = None
        convert_element_type_131: "bf16[512, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(div_12, torch.bfloat16);  div_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convert_element_type_132: "bf16[480, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_200, torch.bfloat16);  primals_200 = None
        convolution_37: "bf16[512, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_131, convert_element_type_132, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_175: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_201, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_133: "f32[512, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_37, torch.float32)
        var_mean_31 = torch.ops.aten.var_mean.correction(convert_element_type_133, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_133 = None
        getitem_62: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = var_mean_31[0]
        getitem_63: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = var_mean_31[1];  var_mean_31 = None
        add_176: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
        rsqrt_31: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_176);  add_176 = None
        sub_31: "f32[512, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convolution_37, getitem_63)
        mul_230: "f32[512, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
        squeeze_93: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3])
        mul_231: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1);  squeeze_93 = None
        mul_232: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_202, 0.9)
        add_177: "f32[480][1]cuda:0" = torch.ops.aten.add.Tensor(mul_231, mul_232);  mul_231 = mul_232 = None
        squeeze_95: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
        mul_233: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_95, 1.00000996502277);  squeeze_95 = None
        mul_234: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_233, 0.1);  mul_233 = None
        mul_235: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_203, 0.9)
        add_178: "f32[480][1]cuda:0" = torch.ops.aten.add.Tensor(mul_234, mul_235);  mul_234 = mul_235 = None
        unsqueeze_124: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_204, -1)
        unsqueeze_125: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        mul_236: "f32[512, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(mul_230, unsqueeze_125);  mul_230 = unsqueeze_125 = None
        unsqueeze_126: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_205, -1)
        unsqueeze_127: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        add_179: "f32[512, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_236, unsqueeze_127);  mul_236 = unsqueeze_127 = None
        convert_element_type_134: "bf16[512, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(add_179, torch.bfloat16);  add_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_135: "f32[512, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_134, torch.float32);  convert_element_type_134 = None
        add_180: "f32[512, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_135, 3)
        clamp_min_13: "f32[512, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.clamp_min.default(add_180, 0);  add_180 = None
        clamp_max_13: "f32[512, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_13, 6);  clamp_min_13 = None
        mul_237: "f32[512, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_135, clamp_max_13);  convert_element_type_135 = clamp_max_13 = None
        div_13: "f32[512, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.div.Tensor(mul_237, 6);  mul_237 = None
        convert_element_type_136: "bf16[512, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_3: "bf16[512, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_136, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convert_element_type_137: "bf16[120][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_207, torch.bfloat16);  primals_207 = None
        convert_element_type_138: "bf16[120, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.prims.convert_element_type.default(primals_206, torch.bfloat16);  primals_206 = None
        convolution_38: "bf16[512, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.convolution.default(mean_3, convert_element_type_138, convert_element_type_137, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        relu_14: "bf16[512, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.relu.default(convolution_38);  convolution_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convert_element_type_139: "bf16[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_209, torch.bfloat16);  primals_209 = None
        convert_element_type_140: "bf16[480, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.prims.convert_element_type.default(primals_208, torch.bfloat16);  primals_208 = None
        convolution_39: "bf16[512, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.convolution.default(relu_14, convert_element_type_140, convert_element_type_139, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        convert_element_type_141: "f32[512, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_39, torch.float32)
        add_181: "f32[512, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_141, 3);  convert_element_type_141 = None
        clamp_min_14: "f32[512, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.clamp_min.default(add_181, 0);  add_181 = None
        clamp_max_14: "f32[512, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_14, 6);  clamp_min_14 = None
        div_14: "f32[512, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_14, 6);  clamp_max_14 = None
        convert_element_type_142: "bf16[512, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.prims.convert_element_type.default(div_14, torch.bfloat16);  div_14 = None
        mul_238: "bf16[512, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_136, convert_element_type_142);  convert_element_type_136 = convert_element_type_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convert_element_type_143: "bf16[112, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.prims.convert_element_type.default(primals_210, torch.bfloat16);  primals_210 = None
        convolution_40: "bf16[512, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.convolution.default(mul_238, convert_element_type_143, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_182: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_211, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_144: "f32[512, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_40, torch.float32)
        var_mean_32 = torch.ops.aten.var_mean.correction(convert_element_type_144, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_144 = None
        getitem_64: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = var_mean_32[0]
        getitem_65: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = var_mean_32[1];  var_mean_32 = None
        add_183: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
        rsqrt_32: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_183);  add_183 = None
        sub_32: "f32[512, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.sub.Tensor(convolution_40, getitem_65)
        mul_239: "f32[512, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
        squeeze_96: "f32[112][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
        squeeze_97: "f32[112][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
        mul_240: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
        mul_241: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_212, 0.9)
        add_184: "f32[112][1]cuda:0" = torch.ops.aten.add.Tensor(mul_240, mul_241);  mul_240 = mul_241 = None
        squeeze_98: "f32[112][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
        mul_242: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_98, 1.00000996502277);  squeeze_98 = None
        mul_243: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_242, 0.1);  mul_242 = None
        mul_244: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_213, 0.9)
        add_185: "f32[112][1]cuda:0" = torch.ops.aten.add.Tensor(mul_243, mul_244);  mul_243 = mul_244 = None
        unsqueeze_128: "f32[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_214, -1)
        unsqueeze_129: "f32[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
        mul_245: "f32[512, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.mul.Tensor(mul_239, unsqueeze_129);  mul_239 = unsqueeze_129 = None
        unsqueeze_130: "f32[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_215, -1);  primals_215 = None
        unsqueeze_131: "f32[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
        add_186: "f32[512, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.add.Tensor(mul_245, unsqueeze_131);  mul_245 = unsqueeze_131 = None
        convert_element_type_145: "bf16[512, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.prims.convert_element_type.default(add_186, torch.bfloat16);  add_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convert_element_type_146: "bf16[672, 112, 1, 1][112, 1, 112, 112]cuda:0" = torch.ops.prims.convert_element_type.default(primals_216, torch.bfloat16);  primals_216 = None
        convolution_41: "bf16[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_145, convert_element_type_146, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_187: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_217, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_147: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_41, torch.float32)
        var_mean_33 = torch.ops.aten.var_mean.correction(convert_element_type_147, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_147 = None
        getitem_66: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = var_mean_33[0]
        getitem_67: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = var_mean_33[1];  var_mean_33 = None
        add_188: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_66, 1e-05)
        rsqrt_33: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
        sub_33: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convolution_41, getitem_67)
        mul_246: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
        squeeze_99: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3])
        mul_247: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1);  squeeze_99 = None
        mul_248: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_218, 0.9)
        add_189: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(mul_247, mul_248);  mul_247 = mul_248 = None
        squeeze_101: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
        mul_249: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_101, 1.00000996502277);  squeeze_101 = None
        mul_250: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_249, 0.1);  mul_249 = None
        mul_251: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_219, 0.9)
        add_190: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(mul_250, mul_251);  mul_250 = mul_251 = None
        unsqueeze_132: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_220, -1)
        unsqueeze_133: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
        mul_252: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(mul_246, unsqueeze_133);  mul_246 = unsqueeze_133 = None
        unsqueeze_134: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_221, -1)
        unsqueeze_135: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
        add_191: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_252, unsqueeze_135);  mul_252 = unsqueeze_135 = None
        convert_element_type_148: "bf16[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(add_191, torch.bfloat16);  add_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_149: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_148, torch.float32);  convert_element_type_148 = None
        add_192: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_149, 3)
        clamp_min_15: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.clamp_min.default(add_192, 0);  add_192 = None
        clamp_max_15: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_15, 6);  clamp_min_15 = None
        mul_253: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_149, clamp_max_15);  convert_element_type_149 = clamp_max_15 = None
        div_15: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.div.Tensor(mul_253, 6);  mul_253 = None
        convert_element_type_150: "bf16[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convert_element_type_151: "bf16[672, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_222, torch.bfloat16);  primals_222 = None
        convolution_42: "bf16[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_150, convert_element_type_151, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 672)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_193: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_223, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_152: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_42, torch.float32)
        var_mean_34 = torch.ops.aten.var_mean.correction(convert_element_type_152, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_152 = None
        getitem_68: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = var_mean_34[0]
        getitem_69: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = var_mean_34[1];  var_mean_34 = None
        add_194: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_68, 1e-05)
        rsqrt_34: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_194);  add_194 = None
        sub_34: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convolution_42, getitem_69)
        mul_254: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
        squeeze_102: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3])
        mul_255: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1);  squeeze_102 = None
        mul_256: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_224, 0.9)
        add_195: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(mul_255, mul_256);  mul_255 = mul_256 = None
        squeeze_104: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
        mul_257: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_104, 1.00000996502277);  squeeze_104 = None
        mul_258: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_257, 0.1);  mul_257 = None
        mul_259: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_225, 0.9)
        add_196: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(mul_258, mul_259);  mul_258 = mul_259 = None
        unsqueeze_136: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_226, -1)
        unsqueeze_137: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
        mul_260: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(mul_254, unsqueeze_137);  mul_254 = unsqueeze_137 = None
        unsqueeze_138: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_227, -1)
        unsqueeze_139: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
        add_197: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_260, unsqueeze_139);  mul_260 = unsqueeze_139 = None
        convert_element_type_153: "bf16[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(add_197, torch.bfloat16);  add_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_154: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_153, torch.float32);  convert_element_type_153 = None
        add_198: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_154, 3)
        clamp_min_16: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.clamp_min.default(add_198, 0);  add_198 = None
        clamp_max_16: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_16, 6);  clamp_min_16 = None
        mul_261: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_154, clamp_max_16);  convert_element_type_154 = clamp_max_16 = None
        div_16: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.div.Tensor(mul_261, 6);  mul_261 = None
        convert_element_type_155: "bf16[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(div_16, torch.bfloat16);  div_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_4: "bf16[512, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_155, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convert_element_type_156: "bf16[168][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_229, torch.bfloat16);  primals_229 = None
        convert_element_type_157: "bf16[168, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(primals_228, torch.bfloat16);  primals_228 = None
        convolution_43: "bf16[512, 168, 1, 1][168, 1, 168, 168]cuda:0" = torch.ops.aten.convolution.default(mean_4, convert_element_type_157, convert_element_type_156, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        relu_15: "bf16[512, 168, 1, 1][168, 1, 168, 168]cuda:0" = torch.ops.aten.relu.default(convolution_43);  convolution_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convert_element_type_158: "bf16[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_231, torch.bfloat16);  primals_231 = None
        convert_element_type_159: "bf16[672, 168, 1, 1][168, 1, 168, 168]cuda:0" = torch.ops.prims.convert_element_type.default(primals_230, torch.bfloat16);  primals_230 = None
        convolution_44: "bf16[512, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.convolution.default(relu_15, convert_element_type_159, convert_element_type_158, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        convert_element_type_160: "f32[512, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_44, torch.float32)
        add_199: "f32[512, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_160, 3);  convert_element_type_160 = None
        clamp_min_17: "f32[512, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.clamp_min.default(add_199, 0);  add_199 = None
        clamp_max_17: "f32[512, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_17, 6);  clamp_min_17 = None
        div_17: "f32[512, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_17, 6);  clamp_max_17 = None
        convert_element_type_161: "bf16[512, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(div_17, torch.bfloat16);  div_17 = None
        mul_262: "bf16[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_155, convert_element_type_161);  convert_element_type_155 = convert_element_type_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convert_element_type_162: "bf16[112, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(primals_232, torch.bfloat16);  primals_232 = None
        convolution_45: "bf16[512, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.convolution.default(mul_262, convert_element_type_162, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_200: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_233, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_163: "f32[512, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_45, torch.float32)
        var_mean_35 = torch.ops.aten.var_mean.correction(convert_element_type_163, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_163 = None
        getitem_70: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = var_mean_35[0]
        getitem_71: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = var_mean_35[1];  var_mean_35 = None
        add_201: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_70, 1e-05)
        rsqrt_35: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_201);  add_201 = None
        sub_35: "f32[512, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.sub.Tensor(convolution_45, getitem_71)
        mul_263: "f32[512, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
        squeeze_105: "f32[112][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
        squeeze_106: "f32[112][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
        mul_264: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
        mul_265: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_234, 0.9)
        add_202: "f32[112][1]cuda:0" = torch.ops.aten.add.Tensor(mul_264, mul_265);  mul_264 = mul_265 = None
        squeeze_107: "f32[112][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
        mul_266: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_107, 1.00000996502277);  squeeze_107 = None
        mul_267: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_266, 0.1);  mul_266 = None
        mul_268: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_235, 0.9)
        add_203: "f32[112][1]cuda:0" = torch.ops.aten.add.Tensor(mul_267, mul_268);  mul_267 = mul_268 = None
        unsqueeze_140: "f32[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_236, -1)
        unsqueeze_141: "f32[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
        mul_269: "f32[512, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.mul.Tensor(mul_263, unsqueeze_141);  mul_263 = unsqueeze_141 = None
        unsqueeze_142: "f32[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_237, -1);  primals_237 = None
        unsqueeze_143: "f32[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
        add_204: "f32[512, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.add.Tensor(mul_269, unsqueeze_143);  mul_269 = unsqueeze_143 = None
        convert_element_type_164: "bf16[512, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.prims.convert_element_type.default(add_204, torch.bfloat16);  add_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:338 in forward, code: x = self.drop_path(x) + shortcut
        add_205: "bf16[512, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_164, convert_element_type_145);  convert_element_type_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convert_element_type_165: "bf16[672, 112, 1, 1][112, 1, 112, 112]cuda:0" = torch.ops.prims.convert_element_type.default(primals_238, torch.bfloat16);  primals_238 = None
        convolution_46: "bf16[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.convolution.default(add_205, convert_element_type_165, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_206: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_239, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_166: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_46, torch.float32)
        var_mean_36 = torch.ops.aten.var_mean.correction(convert_element_type_166, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_166 = None
        getitem_72: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = var_mean_36[0]
        getitem_73: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = var_mean_36[1];  var_mean_36 = None
        add_207: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_72, 1e-05)
        rsqrt_36: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_207);  add_207 = None
        sub_36: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convolution_46, getitem_73)
        mul_270: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
        squeeze_108: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3])
        mul_271: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1);  squeeze_108 = None
        mul_272: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_240, 0.9)
        add_208: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(mul_271, mul_272);  mul_271 = mul_272 = None
        squeeze_110: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
        mul_273: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_110, 1.00000996502277);  squeeze_110 = None
        mul_274: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_273, 0.1);  mul_273 = None
        mul_275: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_241, 0.9)
        add_209: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(mul_274, mul_275);  mul_274 = mul_275 = None
        unsqueeze_144: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_242, -1)
        unsqueeze_145: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
        mul_276: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(mul_270, unsqueeze_145);  mul_270 = unsqueeze_145 = None
        unsqueeze_146: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_243, -1)
        unsqueeze_147: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
        add_210: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_276, unsqueeze_147);  mul_276 = unsqueeze_147 = None
        convert_element_type_167: "bf16[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(add_210, torch.bfloat16);  add_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_168: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_167, torch.float32);  convert_element_type_167 = None
        add_211: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_168, 3)
        clamp_min_18: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.clamp_min.default(add_211, 0);  add_211 = None
        clamp_max_18: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_18, 6);  clamp_min_18 = None
        mul_277: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_168, clamp_max_18);  convert_element_type_168 = clamp_max_18 = None
        div_18: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.div.Tensor(mul_277, 6);  mul_277 = None
        convert_element_type_169: "bf16[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(div_18, torch.bfloat16);  div_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convert_element_type_170: "bf16[672, 1, 5, 5][25, 1, 5, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_244, torch.bfloat16);  primals_244 = None
        convolution_47: "bf16[512, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_169, convert_element_type_170, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 672)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_212: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_245, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_171: "f32[512, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_47, torch.float32)
        var_mean_37 = torch.ops.aten.var_mean.correction(convert_element_type_171, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_171 = None
        getitem_74: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = var_mean_37[0]
        getitem_75: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = var_mean_37[1];  var_mean_37 = None
        add_213: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_74, 1e-05)
        rsqrt_37: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
        sub_37: "f32[512, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.sub.Tensor(convolution_47, getitem_75)
        mul_278: "f32[512, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
        squeeze_111: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3])
        mul_279: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1);  squeeze_111 = None
        mul_280: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_246, 0.9)
        add_214: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(mul_279, mul_280);  mul_279 = mul_280 = None
        squeeze_113: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
        mul_281: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0000398612827361);  squeeze_113 = None
        mul_282: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_281, 0.1);  mul_281 = None
        mul_283: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_247, 0.9)
        add_215: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(mul_282, mul_283);  mul_282 = mul_283 = None
        unsqueeze_148: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_248, -1)
        unsqueeze_149: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
        mul_284: "f32[512, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(mul_278, unsqueeze_149);  mul_278 = unsqueeze_149 = None
        unsqueeze_150: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_249, -1)
        unsqueeze_151: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
        add_216: "f32[512, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_284, unsqueeze_151);  mul_284 = unsqueeze_151 = None
        convert_element_type_172: "bf16[512, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.prims.convert_element_type.default(add_216, torch.bfloat16);  add_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_173: "f32[512, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_172, torch.float32);  convert_element_type_172 = None
        add_217: "f32[512, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_173, 3)
        clamp_min_19: "f32[512, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.clamp_min.default(add_217, 0);  add_217 = None
        clamp_max_19: "f32[512, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_19, 6);  clamp_min_19 = None
        mul_285: "f32[512, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_173, clamp_max_19);  convert_element_type_173 = clamp_max_19 = None
        div_19: "f32[512, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.div.Tensor(mul_285, 6);  mul_285 = None
        convert_element_type_174: "bf16[512, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16);  div_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_5: "bf16[512, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_174, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convert_element_type_175: "bf16[168][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_251, torch.bfloat16);  primals_251 = None
        convert_element_type_176: "bf16[168, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(primals_250, torch.bfloat16);  primals_250 = None
        convolution_48: "bf16[512, 168, 1, 1][168, 1, 168, 168]cuda:0" = torch.ops.aten.convolution.default(mean_5, convert_element_type_176, convert_element_type_175, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        relu_16: "bf16[512, 168, 1, 1][168, 1, 168, 168]cuda:0" = torch.ops.aten.relu.default(convolution_48);  convolution_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convert_element_type_177: "bf16[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_253, torch.bfloat16);  primals_253 = None
        convert_element_type_178: "bf16[672, 168, 1, 1][168, 1, 168, 168]cuda:0" = torch.ops.prims.convert_element_type.default(primals_252, torch.bfloat16);  primals_252 = None
        convolution_49: "bf16[512, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.convolution.default(relu_16, convert_element_type_178, convert_element_type_177, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        convert_element_type_179: "f32[512, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_49, torch.float32)
        add_218: "f32[512, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_179, 3);  convert_element_type_179 = None
        clamp_min_20: "f32[512, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.clamp_min.default(add_218, 0);  add_218 = None
        clamp_max_20: "f32[512, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_20, 6);  clamp_min_20 = None
        div_20: "f32[512, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_20, 6);  clamp_max_20 = None
        convert_element_type_180: "bf16[512, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(div_20, torch.bfloat16);  div_20 = None
        mul_286: "bf16[512, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_174, convert_element_type_180);  convert_element_type_174 = convert_element_type_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convert_element_type_181: "bf16[160, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(primals_254, torch.bfloat16);  primals_254 = None
        convolution_50: "bf16[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.convolution.default(mul_286, convert_element_type_181, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_219: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_255, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_182: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_50, torch.float32)
        var_mean_38 = torch.ops.aten.var_mean.correction(convert_element_type_182, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_182 = None
        getitem_76: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_38[0]
        getitem_77: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_38[1];  var_mean_38 = None
        add_220: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_76, 1e-05)
        rsqrt_38: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_220);  add_220 = None
        sub_38: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.sub.Tensor(convolution_50, getitem_77)
        mul_287: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
        squeeze_114: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
        squeeze_115: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
        mul_288: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
        mul_289: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_256, 0.9)
        add_221: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_288, mul_289);  mul_288 = mul_289 = None
        squeeze_116: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
        mul_290: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0000398612827361);  squeeze_116 = None
        mul_291: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_290, 0.1);  mul_290 = None
        mul_292: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_257, 0.9)
        add_222: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_291, mul_292);  mul_291 = mul_292 = None
        unsqueeze_152: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_258, -1)
        unsqueeze_153: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
        mul_293: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_287, unsqueeze_153);  mul_287 = unsqueeze_153 = None
        unsqueeze_154: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_259, -1);  primals_259 = None
        unsqueeze_155: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
        add_223: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.add.Tensor(mul_293, unsqueeze_155);  mul_293 = unsqueeze_155 = None
        convert_element_type_183: "bf16[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(add_223, torch.bfloat16);  add_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convert_element_type_184: "bf16[960, 160, 1, 1][160, 1, 160, 160]cuda:0" = torch.ops.prims.convert_element_type.default(primals_260, torch.bfloat16);  primals_260 = None
        convolution_51: "bf16[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_183, convert_element_type_184, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_224: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_261, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_185: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_51, torch.float32)
        var_mean_39 = torch.ops.aten.var_mean.correction(convert_element_type_185, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_185 = None
        getitem_78: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = var_mean_39[0]
        getitem_79: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = var_mean_39[1];  var_mean_39 = None
        add_225: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_78, 1e-05)
        rsqrt_39: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_225);  add_225 = None
        sub_39: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convolution_51, getitem_79)
        mul_294: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
        squeeze_117: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3])
        mul_295: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1);  squeeze_117 = None
        mul_296: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_262, 0.9)
        add_226: "f32[960][1]cuda:0" = torch.ops.aten.add.Tensor(mul_295, mul_296);  mul_295 = mul_296 = None
        squeeze_119: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
        mul_297: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_119, 1.0000398612827361);  squeeze_119 = None
        mul_298: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_297, 0.1);  mul_297 = None
        mul_299: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_263, 0.9)
        add_227: "f32[960][1]cuda:0" = torch.ops.aten.add.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None
        unsqueeze_156: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_264, -1)
        unsqueeze_157: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
        mul_300: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(mul_294, unsqueeze_157);  mul_294 = unsqueeze_157 = None
        unsqueeze_158: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_265, -1)
        unsqueeze_159: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
        add_228: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.add.Tensor(mul_300, unsqueeze_159);  mul_300 = unsqueeze_159 = None
        convert_element_type_186: "bf16[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(add_228, torch.bfloat16);  add_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_187: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_186, torch.float32);  convert_element_type_186 = None
        add_229: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_187, 3)
        clamp_min_21: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.clamp_min.default(add_229, 0);  add_229 = None
        clamp_max_21: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_21, 6);  clamp_min_21 = None
        mul_301: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_187, clamp_max_21);  convert_element_type_187 = clamp_max_21 = None
        div_21: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.div.Tensor(mul_301, 6);  mul_301 = None
        convert_element_type_188: "bf16[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16);  div_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convert_element_type_189: "bf16[960, 1, 5, 5][25, 1, 5, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_266, torch.bfloat16);  primals_266 = None
        convolution_52: "bf16[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_188, convert_element_type_189, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 960)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_230: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_267, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_190: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_52, torch.float32)
        var_mean_40 = torch.ops.aten.var_mean.correction(convert_element_type_190, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_190 = None
        getitem_80: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = var_mean_40[0]
        getitem_81: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = var_mean_40[1];  var_mean_40 = None
        add_231: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_80, 1e-05)
        rsqrt_40: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_231);  add_231 = None
        sub_40: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convolution_52, getitem_81)
        mul_302: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
        squeeze_120: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3])
        mul_303: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1);  squeeze_120 = None
        mul_304: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_268, 0.9)
        add_232: "f32[960][1]cuda:0" = torch.ops.aten.add.Tensor(mul_303, mul_304);  mul_303 = mul_304 = None
        squeeze_122: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
        mul_305: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_122, 1.0000398612827361);  squeeze_122 = None
        mul_306: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_305, 0.1);  mul_305 = None
        mul_307: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_269, 0.9)
        add_233: "f32[960][1]cuda:0" = torch.ops.aten.add.Tensor(mul_306, mul_307);  mul_306 = mul_307 = None
        unsqueeze_160: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_270, -1)
        unsqueeze_161: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
        mul_308: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(mul_302, unsqueeze_161);  mul_302 = unsqueeze_161 = None
        unsqueeze_162: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_271, -1)
        unsqueeze_163: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
        add_234: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.add.Tensor(mul_308, unsqueeze_163);  mul_308 = unsqueeze_163 = None
        convert_element_type_191: "bf16[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(add_234, torch.bfloat16);  add_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_192: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_191, torch.float32);  convert_element_type_191 = None
        add_235: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_192, 3)
        clamp_min_22: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.clamp_min.default(add_235, 0);  add_235 = None
        clamp_max_22: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_22, 6);  clamp_min_22 = None
        mul_309: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_192, clamp_max_22);  convert_element_type_192 = clamp_max_22 = None
        div_22: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.div.Tensor(mul_309, 6);  mul_309 = None
        convert_element_type_193: "bf16[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(div_22, torch.bfloat16);  div_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_6: "bf16[512, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_193, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convert_element_type_194: "bf16[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_273, torch.bfloat16);  primals_273 = None
        convert_element_type_195: "bf16[240, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.prims.convert_element_type.default(primals_272, torch.bfloat16);  primals_272 = None
        convolution_53: "bf16[512, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.aten.convolution.default(mean_6, convert_element_type_195, convert_element_type_194, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        relu_17: "bf16[512, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.aten.relu.default(convolution_53);  convolution_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convert_element_type_196: "bf16[960][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_275, torch.bfloat16);  primals_275 = None
        convert_element_type_197: "bf16[960, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.prims.convert_element_type.default(primals_274, torch.bfloat16);  primals_274 = None
        convolution_54: "bf16[512, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.convolution.default(relu_17, convert_element_type_197, convert_element_type_196, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        convert_element_type_198: "f32[512, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_54, torch.float32)
        add_236: "f32[512, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_198, 3);  convert_element_type_198 = None
        clamp_min_23: "f32[512, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.clamp_min.default(add_236, 0);  add_236 = None
        clamp_max_23: "f32[512, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_23, 6);  clamp_min_23 = None
        div_23: "f32[512, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_23, 6);  clamp_max_23 = None
        convert_element_type_199: "bf16[512, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.prims.convert_element_type.default(div_23, torch.bfloat16);  div_23 = None
        mul_310: "bf16[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_193, convert_element_type_199);  convert_element_type_193 = convert_element_type_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convert_element_type_200: "bf16[160, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.prims.convert_element_type.default(primals_276, torch.bfloat16);  primals_276 = None
        convolution_55: "bf16[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.convolution.default(mul_310, convert_element_type_200, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_237: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_277, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_201: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_55, torch.float32)
        var_mean_41 = torch.ops.aten.var_mean.correction(convert_element_type_201, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_201 = None
        getitem_82: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_41[0]
        getitem_83: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_41[1];  var_mean_41 = None
        add_238: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_82, 1e-05)
        rsqrt_41: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_238);  add_238 = None
        sub_41: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.sub.Tensor(convolution_55, getitem_83)
        mul_311: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
        squeeze_123: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
        squeeze_124: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
        mul_312: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
        mul_313: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_278, 0.9)
        add_239: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_312, mul_313);  mul_312 = mul_313 = None
        squeeze_125: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
        mul_314: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_125, 1.0000398612827361);  squeeze_125 = None
        mul_315: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_314, 0.1);  mul_314 = None
        mul_316: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_279, 0.9)
        add_240: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_315, mul_316);  mul_315 = mul_316 = None
        unsqueeze_164: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_280, -1)
        unsqueeze_165: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
        mul_317: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_311, unsqueeze_165);  mul_311 = unsqueeze_165 = None
        unsqueeze_166: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_281, -1);  primals_281 = None
        unsqueeze_167: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
        add_241: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.add.Tensor(mul_317, unsqueeze_167);  mul_317 = unsqueeze_167 = None
        convert_element_type_202: "bf16[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(add_241, torch.bfloat16);  add_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:338 in forward, code: x = self.drop_path(x) + shortcut
        add_242: "bf16[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_202, convert_element_type_183);  convert_element_type_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convert_element_type_203: "bf16[960, 160, 1, 1][160, 1, 160, 160]cuda:0" = torch.ops.prims.convert_element_type.default(primals_282, torch.bfloat16);  primals_282 = None
        convolution_56: "bf16[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.convolution.default(add_242, convert_element_type_203, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_243: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_283, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_204: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_56, torch.float32)
        var_mean_42 = torch.ops.aten.var_mean.correction(convert_element_type_204, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_204 = None
        getitem_84: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = var_mean_42[0]
        getitem_85: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = var_mean_42[1];  var_mean_42 = None
        add_244: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_84, 1e-05)
        rsqrt_42: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_244);  add_244 = None
        sub_42: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convolution_56, getitem_85)
        mul_318: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
        squeeze_126: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3])
        mul_319: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1);  squeeze_126 = None
        mul_320: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_284, 0.9)
        add_245: "f32[960][1]cuda:0" = torch.ops.aten.add.Tensor(mul_319, mul_320);  mul_319 = mul_320 = None
        squeeze_128: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
        mul_321: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_128, 1.0000398612827361);  squeeze_128 = None
        mul_322: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_321, 0.1);  mul_321 = None
        mul_323: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_285, 0.9)
        add_246: "f32[960][1]cuda:0" = torch.ops.aten.add.Tensor(mul_322, mul_323);  mul_322 = mul_323 = None
        unsqueeze_168: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_286, -1)
        unsqueeze_169: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
        mul_324: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(mul_318, unsqueeze_169);  mul_318 = unsqueeze_169 = None
        unsqueeze_170: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_287, -1)
        unsqueeze_171: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
        add_247: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.add.Tensor(mul_324, unsqueeze_171);  mul_324 = unsqueeze_171 = None
        convert_element_type_205: "bf16[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(add_247, torch.bfloat16);  add_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_206: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_205, torch.float32);  convert_element_type_205 = None
        add_248: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_206, 3)
        clamp_min_24: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.clamp_min.default(add_248, 0);  add_248 = None
        clamp_max_24: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_24, 6);  clamp_min_24 = None
        mul_325: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_206, clamp_max_24);  convert_element_type_206 = clamp_max_24 = None
        div_24: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.div.Tensor(mul_325, 6);  mul_325 = None
        convert_element_type_207: "bf16[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(div_24, torch.bfloat16);  div_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convert_element_type_208: "bf16[960, 1, 5, 5][25, 1, 5, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_288, torch.bfloat16);  primals_288 = None
        convolution_57: "bf16[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_207, convert_element_type_208, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 960)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_249: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_289, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_209: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_57, torch.float32)
        var_mean_43 = torch.ops.aten.var_mean.correction(convert_element_type_209, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_209 = None
        getitem_86: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = var_mean_43[0]
        getitem_87: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = var_mean_43[1];  var_mean_43 = None
        add_250: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_86, 1e-05)
        rsqrt_43: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_250);  add_250 = None
        sub_43: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convolution_57, getitem_87)
        mul_326: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
        squeeze_129: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3])
        mul_327: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1);  squeeze_129 = None
        mul_328: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_290, 0.9)
        add_251: "f32[960][1]cuda:0" = torch.ops.aten.add.Tensor(mul_327, mul_328);  mul_327 = mul_328 = None
        squeeze_131: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
        mul_329: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_131, 1.0000398612827361);  squeeze_131 = None
        mul_330: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_329, 0.1);  mul_329 = None
        mul_331: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_291, 0.9)
        add_252: "f32[960][1]cuda:0" = torch.ops.aten.add.Tensor(mul_330, mul_331);  mul_330 = mul_331 = None
        unsqueeze_172: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_292, -1)
        unsqueeze_173: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
        mul_332: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(mul_326, unsqueeze_173);  mul_326 = unsqueeze_173 = None
        unsqueeze_174: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_293, -1)
        unsqueeze_175: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
        add_253: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.add.Tensor(mul_332, unsqueeze_175);  mul_332 = unsqueeze_175 = None
        convert_element_type_210: "bf16[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(add_253, torch.bfloat16);  add_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_211: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_210, torch.float32);  convert_element_type_210 = None
        add_254: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_211, 3)
        clamp_min_25: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.clamp_min.default(add_254, 0);  add_254 = None
        clamp_max_25: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_25, 6);  clamp_min_25 = None
        mul_333: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_211, clamp_max_25);  convert_element_type_211 = clamp_max_25 = None
        div_25: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.div.Tensor(mul_333, 6);  mul_333 = None
        convert_element_type_212: "bf16[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(div_25, torch.bfloat16);  div_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_7: "bf16[512, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_212, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convert_element_type_213: "bf16[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_295, torch.bfloat16);  primals_295 = None
        convert_element_type_214: "bf16[240, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.prims.convert_element_type.default(primals_294, torch.bfloat16);  primals_294 = None
        convolution_58: "bf16[512, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.aten.convolution.default(mean_7, convert_element_type_214, convert_element_type_213, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        relu_18: "bf16[512, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.aten.relu.default(convolution_58);  convolution_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convert_element_type_215: "bf16[960][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_297, torch.bfloat16);  primals_297 = None
        convert_element_type_216: "bf16[960, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.prims.convert_element_type.default(primals_296, torch.bfloat16);  primals_296 = None
        convolution_59: "bf16[512, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.convolution.default(relu_18, convert_element_type_216, convert_element_type_215, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        convert_element_type_217: "f32[512, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_59, torch.float32)
        add_255: "f32[512, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_217, 3);  convert_element_type_217 = None
        clamp_min_26: "f32[512, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.clamp_min.default(add_255, 0);  add_255 = None
        clamp_max_26: "f32[512, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_26, 6);  clamp_min_26 = None
        div_26: "f32[512, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_26, 6);  clamp_max_26 = None
        convert_element_type_218: "bf16[512, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.prims.convert_element_type.default(div_26, torch.bfloat16);  div_26 = None
        mul_334: "bf16[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_212, convert_element_type_218);  convert_element_type_212 = convert_element_type_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convert_element_type_219: "bf16[160, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.prims.convert_element_type.default(primals_298, torch.bfloat16);  primals_298 = None
        convolution_60: "bf16[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.convolution.default(mul_334, convert_element_type_219, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_256: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_299, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_220: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_60, torch.float32)
        var_mean_44 = torch.ops.aten.var_mean.correction(convert_element_type_220, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_220 = None
        getitem_88: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_44[0]
        getitem_89: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_44[1];  var_mean_44 = None
        add_257: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_88, 1e-05)
        rsqrt_44: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_257);  add_257 = None
        sub_44: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.sub.Tensor(convolution_60, getitem_89)
        mul_335: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
        squeeze_132: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
        squeeze_133: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
        mul_336: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_132, 0.1)
        mul_337: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_300, 0.9)
        add_258: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_336, mul_337);  mul_336 = mul_337 = None
        squeeze_134: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
        mul_338: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_134, 1.0000398612827361);  squeeze_134 = None
        mul_339: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_338, 0.1);  mul_338 = None
        mul_340: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_301, 0.9)
        add_259: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_339, mul_340);  mul_339 = mul_340 = None
        unsqueeze_176: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_302, -1)
        unsqueeze_177: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
        mul_341: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_335, unsqueeze_177);  mul_335 = unsqueeze_177 = None
        unsqueeze_178: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_303, -1);  primals_303 = None
        unsqueeze_179: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
        add_260: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.add.Tensor(mul_341, unsqueeze_179);  mul_341 = unsqueeze_179 = None
        convert_element_type_221: "bf16[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(add_260, torch.bfloat16);  add_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:338 in forward, code: x = self.drop_path(x) + shortcut
        add_261: "bf16[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_221, add_242);  convert_element_type_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:135 in forward, code: x = self.conv(x)
        convert_element_type_222: "bf16[960, 160, 1, 1][160, 1, 160, 160]cuda:0" = torch.ops.prims.convert_element_type.default(primals_304, torch.bfloat16);  primals_304 = None
        convolution_61: "bf16[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.convolution.default(add_261, convert_element_type_222, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_262: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_305, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_223: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_61, torch.float32)
        var_mean_45 = torch.ops.aten.var_mean.correction(convert_element_type_223, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_223 = None
        getitem_90: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = var_mean_45[0]
        getitem_91: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = var_mean_45[1];  var_mean_45 = None
        add_263: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_90, 1e-05)
        rsqrt_45: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_263);  add_263 = None
        sub_45: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convolution_61, getitem_91)
        mul_342: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
        squeeze_135: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3])
        mul_343: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_135, 0.1);  squeeze_135 = None
        mul_344: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_306, 0.9)
        add_264: "f32[960][1]cuda:0" = torch.ops.aten.add.Tensor(mul_343, mul_344);  mul_343 = mul_344 = None
        squeeze_137: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
        mul_345: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_137, 1.0000398612827361);  squeeze_137 = None
        mul_346: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_345, 0.1);  mul_345 = None
        mul_347: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_307, 0.9)
        add_265: "f32[960][1]cuda:0" = torch.ops.aten.add.Tensor(mul_346, mul_347);  mul_346 = mul_347 = None
        unsqueeze_180: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_308, -1)
        unsqueeze_181: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
        mul_348: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(mul_342, unsqueeze_181);  mul_342 = unsqueeze_181 = None
        unsqueeze_182: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_309, -1)
        unsqueeze_183: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
        add_266: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.add.Tensor(mul_348, unsqueeze_183);  mul_348 = unsqueeze_183 = None
        convert_element_type_224: "bf16[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(add_266, torch.bfloat16);  add_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_225: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_224, torch.float32);  convert_element_type_224 = None
        add_267: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_225, 3)
        clamp_min_27: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.clamp_min.default(add_267, 0);  add_267 = None
        clamp_max_27: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_27, 6);  clamp_min_27 = None
        mul_349: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_225, clamp_max_27);  convert_element_type_225 = clamp_max_27 = None
        div_27: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.div.Tensor(mul_349, 6);  mul_349 = None
        convert_element_type_226: "bf16[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(div_27, torch.bfloat16);  div_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_8: "bf16[512, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_226, [-1, -2], True);  convert_element_type_226 = None
        as_strided: "bf16[512, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.as_strided.default(mean_8, [512, 960, 1, 1], [960, 1, 960, 960]);  mean_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilenetv3.py:323 in forward_head, code: x = self.conv_head(x)
        convert_element_type_227: "bf16[1280][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_311, torch.bfloat16);  primals_311 = None
        convert_element_type_228: "bf16[1280, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.prims.convert_element_type.default(primals_310, torch.bfloat16);  primals_310 = None
        convolution_62: "bf16[512, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.convolution.default(as_strided, convert_element_type_228, convert_element_type_227, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilenetv3.py:325 in forward_head, code: x = self.act2(x)
        convert_element_type_229: "f32[512, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_62, torch.float32)
        add_268: "f32[512, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_229, 3)
        clamp_min_28: "f32[512, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.clamp_min.default(add_268, 0);  add_268 = None
        clamp_max_28: "f32[512, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_28, 6);  clamp_min_28 = None
        mul_350: "f32[512, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_229, clamp_max_28);  convert_element_type_229 = clamp_max_28 = None
        div_28: "f32[512, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.div.Tensor(mul_350, 6);  mul_350 = None
        convert_element_type_230: "bf16[512, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(div_28, torch.bfloat16);  div_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/linear.py:19 in forward, code: return F.linear(input, self.weight, self.bias)
        convert_element_type_231: "bf16[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_313, torch.bfloat16);  primals_313 = None
        convert_element_type_232: "bf16[1000, 1280][1280, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_312, torch.bfloat16);  primals_312 = None
        permute: "bf16[1280, 1000][1, 1280]cuda:0" = torch.ops.aten.permute.default(convert_element_type_232, [1, 0]);  convert_element_type_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilenetv3.py:326 in forward_head, code: x = self.flatten(x)
        view_1: "bf16[512, 1280][1280, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_230, [512, 1280]);  convert_element_type_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/linear.py:19 in forward, code: return F.linear(input, self.weight, self.bias)
        addmm: "bf16[512, 1000][1000, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_231, view_1, permute);  convert_element_type_231 = None
        permute_1: "bf16[1000, 1280][1280, 1]cuda:0" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_196: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
        unsqueeze_197: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_196, 2);  unsqueeze_196 = None
        unsqueeze_198: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 3);  unsqueeze_197 = None
        unsqueeze_232: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
        unsqueeze_233: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 2);  unsqueeze_232 = None
        unsqueeze_234: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 3);  unsqueeze_233 = None
        unsqueeze_268: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
        unsqueeze_269: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 2);  unsqueeze_268 = None
        unsqueeze_270: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 3);  unsqueeze_269 = None
        unsqueeze_304: "f32[1, 112][112, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
        unsqueeze_305: "f32[1, 112, 1][112, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_304, 2);  unsqueeze_304 = None
        unsqueeze_306: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 3);  unsqueeze_305 = None
        unsqueeze_340: "f32[1, 112][112, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
        unsqueeze_341: "f32[1, 112, 1][112, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_340, 2);  unsqueeze_340 = None
        unsqueeze_342: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 3);  unsqueeze_341 = None
        unsqueeze_376: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
        unsqueeze_377: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 2);  unsqueeze_376 = None
        unsqueeze_378: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 3);  unsqueeze_377 = None
        unsqueeze_412: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
        unsqueeze_413: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 2);  unsqueeze_412 = None
        unsqueeze_414: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 3);  unsqueeze_413 = None
        unsqueeze_448: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
        unsqueeze_449: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_448, 2);  unsqueeze_448 = None
        unsqueeze_450: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 3);  unsqueeze_449 = None
        unsqueeze_484: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
        unsqueeze_485: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 2);  unsqueeze_484 = None
        unsqueeze_486: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 3);  unsqueeze_485 = None
        unsqueeze_520: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
        unsqueeze_521: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 2);  unsqueeze_520 = None
        unsqueeze_522: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 3);  unsqueeze_521 = None
        unsqueeze_544: "f32[1, 120][120, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
        unsqueeze_545: "f32[1, 120, 1][120, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_544, 2);  unsqueeze_544 = None
        unsqueeze_546: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 3);  unsqueeze_545 = None
        unsqueeze_556: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
        unsqueeze_557: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_556, 2);  unsqueeze_556 = None
        unsqueeze_558: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 3);  unsqueeze_557 = None
        unsqueeze_580: "f32[1, 120][120, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
        unsqueeze_581: "f32[1, 120, 1][120, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 2);  unsqueeze_580 = None
        unsqueeze_582: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 3);  unsqueeze_581 = None
        unsqueeze_592: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
        unsqueeze_593: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 2);  unsqueeze_592 = None
        unsqueeze_594: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 3);  unsqueeze_593 = None
        unsqueeze_616: "f32[1, 72][72, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
        unsqueeze_617: "f32[1, 72, 1][72, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_616, 2);  unsqueeze_616 = None
        unsqueeze_618: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 3);  unsqueeze_617 = None
        unsqueeze_628: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
        unsqueeze_629: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_628, 2);  unsqueeze_628 = None
        unsqueeze_630: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 3);  unsqueeze_629 = None
        unsqueeze_640: "f32[1, 72][72, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
        unsqueeze_641: "f32[1, 72, 1][72, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_640, 2);  unsqueeze_640 = None
        unsqueeze_642: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 3);  unsqueeze_641 = None
        unsqueeze_652: "f32[1, 72][72, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
        unsqueeze_653: "f32[1, 72, 1][72, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_652, 2);  unsqueeze_652 = None
        unsqueeze_654: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 3);  unsqueeze_653 = None
        unsqueeze_664: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
        unsqueeze_665: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_664, 2);  unsqueeze_664 = None
        unsqueeze_666: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 3);  unsqueeze_665 = None
        unsqueeze_676: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
        unsqueeze_677: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_676, 2);  unsqueeze_676 = None
        unsqueeze_678: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 3);  unsqueeze_677 = None
        unsqueeze_688: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
        unsqueeze_689: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_688, 2);  unsqueeze_688 = None
        unsqueeze_690: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 3);  unsqueeze_689 = None
        unsqueeze_700: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
        unsqueeze_701: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_700, 2);  unsqueeze_700 = None
        unsqueeze_702: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 3);  unsqueeze_701 = None
        unsqueeze_712: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
        unsqueeze_713: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_712, 2);  unsqueeze_712 = None
        unsqueeze_714: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 3);  unsqueeze_713 = None

        # No stacktrace found for following nodes
        copy_: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_3, add);  primals_3 = add = copy_ = None
        copy__1: "f32[16][1]cuda:0" = torch.ops.aten.copy_.default(primals_4, add_2);  primals_4 = add_2 = copy__1 = None
        copy__2: "f32[16][1]cuda:0" = torch.ops.aten.copy_.default(primals_5, add_3);  primals_5 = add_3 = copy__2 = None
        copy__3: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_9, add_6);  primals_9 = add_6 = copy__3 = None
        copy__4: "f32[16][1]cuda:0" = torch.ops.aten.copy_.default(primals_10, add_8);  primals_10 = add_8 = copy__4 = None
        copy__5: "f32[16][1]cuda:0" = torch.ops.aten.copy_.default(primals_11, add_9);  primals_11 = add_9 = copy__5 = None
        copy__6: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_15, add_11);  primals_15 = add_11 = copy__6 = None
        copy__7: "f32[16][1]cuda:0" = torch.ops.aten.copy_.default(primals_16, add_13);  primals_16 = add_13 = copy__7 = None
        copy__8: "f32[16][1]cuda:0" = torch.ops.aten.copy_.default(primals_17, add_14);  primals_17 = add_14 = copy__8 = None
        copy__9: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_21, add_17);  primals_21 = add_17 = copy__9 = None
        copy__10: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_22, add_19);  primals_22 = add_19 = copy__10 = None
        copy__11: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_23, add_20);  primals_23 = add_20 = copy__11 = None
        copy__12: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_27, add_22);  primals_27 = add_22 = copy__12 = None
        copy__13: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_28, add_24);  primals_28 = add_24 = copy__13 = None
        copy__14: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_29, add_25);  primals_29 = add_25 = copy__14 = None
        copy__15: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_33, add_27);  primals_33 = add_27 = copy__15 = None
        copy__16: "f32[24][1]cuda:0" = torch.ops.aten.copy_.default(primals_34, add_29);  primals_34 = add_29 = copy__16 = None
        copy__17: "f32[24][1]cuda:0" = torch.ops.aten.copy_.default(primals_35, add_30);  primals_35 = add_30 = copy__17 = None
        copy__18: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_39, add_32);  primals_39 = add_32 = copy__18 = None
        copy__19: "f32[72][1]cuda:0" = torch.ops.aten.copy_.default(primals_40, add_34);  primals_40 = add_34 = copy__19 = None
        copy__20: "f32[72][1]cuda:0" = torch.ops.aten.copy_.default(primals_41, add_35);  primals_41 = add_35 = copy__20 = None
        copy__21: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_45, add_37);  primals_45 = add_37 = copy__21 = None
        copy__22: "f32[72][1]cuda:0" = torch.ops.aten.copy_.default(primals_46, add_39);  primals_46 = add_39 = copy__22 = None
        copy__23: "f32[72][1]cuda:0" = torch.ops.aten.copy_.default(primals_47, add_40);  primals_47 = add_40 = copy__23 = None
        copy__24: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_51, add_42);  primals_51 = add_42 = copy__24 = None
        copy__25: "f32[24][1]cuda:0" = torch.ops.aten.copy_.default(primals_52, add_44);  primals_52 = add_44 = copy__25 = None
        copy__26: "f32[24][1]cuda:0" = torch.ops.aten.copy_.default(primals_53, add_45);  primals_53 = add_45 = copy__26 = None
        copy__27: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_57, add_48);  primals_57 = add_48 = copy__27 = None
        copy__28: "f32[72][1]cuda:0" = torch.ops.aten.copy_.default(primals_58, add_50);  primals_58 = add_50 = copy__28 = None
        copy__29: "f32[72][1]cuda:0" = torch.ops.aten.copy_.default(primals_59, add_51);  primals_59 = add_51 = copy__29 = None
        copy__30: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_63, add_53);  primals_63 = add_53 = copy__30 = None
        copy__31: "f32[72][1]cuda:0" = torch.ops.aten.copy_.default(primals_64, add_55);  primals_64 = add_55 = copy__31 = None
        copy__32: "f32[72][1]cuda:0" = torch.ops.aten.copy_.default(primals_65, add_56);  primals_65 = add_56 = copy__32 = None
        copy__33: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_73, add_59);  primals_73 = add_59 = copy__33 = None
        copy__34: "f32[40][1]cuda:0" = torch.ops.aten.copy_.default(primals_74, add_61);  primals_74 = add_61 = copy__34 = None
        copy__35: "f32[40][1]cuda:0" = torch.ops.aten.copy_.default(primals_75, add_62);  primals_75 = add_62 = copy__35 = None
        copy__36: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_79, add_64);  primals_79 = add_64 = copy__36 = None
        copy__37: "f32[120][1]cuda:0" = torch.ops.aten.copy_.default(primals_80, add_66);  primals_80 = add_66 = copy__37 = None
        copy__38: "f32[120][1]cuda:0" = torch.ops.aten.copy_.default(primals_81, add_67);  primals_81 = add_67 = copy__38 = None
        copy__39: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_85, add_69);  primals_85 = add_69 = copy__39 = None
        copy__40: "f32[120][1]cuda:0" = torch.ops.aten.copy_.default(primals_86, add_71);  primals_86 = add_71 = copy__40 = None
        copy__41: "f32[120][1]cuda:0" = torch.ops.aten.copy_.default(primals_87, add_72);  primals_87 = add_72 = copy__41 = None
        copy__42: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_95, add_75);  primals_95 = add_75 = copy__42 = None
        copy__43: "f32[40][1]cuda:0" = torch.ops.aten.copy_.default(primals_96, add_77);  primals_96 = add_77 = copy__43 = None
        copy__44: "f32[40][1]cuda:0" = torch.ops.aten.copy_.default(primals_97, add_78);  primals_97 = add_78 = copy__44 = None
        copy__45: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_101, add_81);  primals_101 = add_81 = copy__45 = None
        copy__46: "f32[120][1]cuda:0" = torch.ops.aten.copy_.default(primals_102, add_83);  primals_102 = add_83 = copy__46 = None
        copy__47: "f32[120][1]cuda:0" = torch.ops.aten.copy_.default(primals_103, add_84);  primals_103 = add_84 = copy__47 = None
        copy__48: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_107, add_86);  primals_107 = add_86 = copy__48 = None
        copy__49: "f32[120][1]cuda:0" = torch.ops.aten.copy_.default(primals_108, add_88);  primals_108 = add_88 = copy__49 = None
        copy__50: "f32[120][1]cuda:0" = torch.ops.aten.copy_.default(primals_109, add_89);  primals_109 = add_89 = copy__50 = None
        copy__51: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_117, add_92);  primals_117 = add_92 = copy__51 = None
        copy__52: "f32[40][1]cuda:0" = torch.ops.aten.copy_.default(primals_118, add_94);  primals_118 = add_94 = copy__52 = None
        copy__53: "f32[40][1]cuda:0" = torch.ops.aten.copy_.default(primals_119, add_95);  primals_119 = add_95 = copy__53 = None
        copy__54: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_123, add_98);  primals_123 = add_98 = copy__54 = None
        copy__55: "f32[240][1]cuda:0" = torch.ops.aten.copy_.default(primals_124, add_100);  primals_124 = add_100 = copy__55 = None
        copy__56: "f32[240][1]cuda:0" = torch.ops.aten.copy_.default(primals_125, add_101);  primals_125 = add_101 = copy__56 = None
        copy__57: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_129, add_104);  primals_129 = add_104 = copy__57 = None
        copy__58: "f32[240][1]cuda:0" = torch.ops.aten.copy_.default(primals_130, add_106);  primals_130 = add_106 = copy__58 = None
        copy__59: "f32[240][1]cuda:0" = torch.ops.aten.copy_.default(primals_131, add_107);  primals_131 = add_107 = copy__59 = None
        copy__60: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_135, add_110);  primals_135 = add_110 = copy__60 = None
        copy__61: "f32[80][1]cuda:0" = torch.ops.aten.copy_.default(primals_136, add_112);  primals_136 = add_112 = copy__61 = None
        copy__62: "f32[80][1]cuda:0" = torch.ops.aten.copy_.default(primals_137, add_113);  primals_137 = add_113 = copy__62 = None
        copy__63: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_141, add_115);  primals_141 = add_115 = copy__63 = None
        copy__64: "f32[200][1]cuda:0" = torch.ops.aten.copy_.default(primals_142, add_117);  primals_142 = add_117 = copy__64 = None
        copy__65: "f32[200][1]cuda:0" = torch.ops.aten.copy_.default(primals_143, add_118);  primals_143 = add_118 = copy__65 = None
        copy__66: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_147, add_121);  primals_147 = add_121 = copy__66 = None
        copy__67: "f32[200][1]cuda:0" = torch.ops.aten.copy_.default(primals_148, add_123);  primals_148 = add_123 = copy__67 = None
        copy__68: "f32[200][1]cuda:0" = torch.ops.aten.copy_.default(primals_149, add_124);  primals_149 = add_124 = copy__68 = None
        copy__69: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_153, add_127);  primals_153 = add_127 = copy__69 = None
        copy__70: "f32[80][1]cuda:0" = torch.ops.aten.copy_.default(primals_154, add_129);  primals_154 = add_129 = copy__70 = None
        copy__71: "f32[80][1]cuda:0" = torch.ops.aten.copy_.default(primals_155, add_130);  primals_155 = add_130 = copy__71 = None
        copy__72: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_159, add_133);  primals_159 = add_133 = copy__72 = None
        copy__73: "f32[184][1]cuda:0" = torch.ops.aten.copy_.default(primals_160, add_135);  primals_160 = add_135 = copy__73 = None
        copy__74: "f32[184][1]cuda:0" = torch.ops.aten.copy_.default(primals_161, add_136);  primals_161 = add_136 = copy__74 = None
        copy__75: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_165, add_139);  primals_165 = add_139 = copy__75 = None
        copy__76: "f32[184][1]cuda:0" = torch.ops.aten.copy_.default(primals_166, add_141);  primals_166 = add_141 = copy__76 = None
        copy__77: "f32[184][1]cuda:0" = torch.ops.aten.copy_.default(primals_167, add_142);  primals_167 = add_142 = copy__77 = None
        copy__78: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_171, add_145);  primals_171 = add_145 = copy__78 = None
        copy__79: "f32[80][1]cuda:0" = torch.ops.aten.copy_.default(primals_172, add_147);  primals_172 = add_147 = copy__79 = None
        copy__80: "f32[80][1]cuda:0" = torch.ops.aten.copy_.default(primals_173, add_148);  primals_173 = add_148 = copy__80 = None
        copy__81: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_177, add_151);  primals_177 = add_151 = copy__81 = None
        copy__82: "f32[184][1]cuda:0" = torch.ops.aten.copy_.default(primals_178, add_153);  primals_178 = add_153 = copy__82 = None
        copy__83: "f32[184][1]cuda:0" = torch.ops.aten.copy_.default(primals_179, add_154);  primals_179 = add_154 = copy__83 = None
        copy__84: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_183, add_157);  primals_183 = add_157 = copy__84 = None
        copy__85: "f32[184][1]cuda:0" = torch.ops.aten.copy_.default(primals_184, add_159);  primals_184 = add_159 = copy__85 = None
        copy__86: "f32[184][1]cuda:0" = torch.ops.aten.copy_.default(primals_185, add_160);  primals_185 = add_160 = copy__86 = None
        copy__87: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_189, add_163);  primals_189 = add_163 = copy__87 = None
        copy__88: "f32[80][1]cuda:0" = torch.ops.aten.copy_.default(primals_190, add_165);  primals_190 = add_165 = copy__88 = None
        copy__89: "f32[80][1]cuda:0" = torch.ops.aten.copy_.default(primals_191, add_166);  primals_191 = add_166 = copy__89 = None
        copy__90: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_195, add_169);  primals_195 = add_169 = copy__90 = None
        copy__91: "f32[480][1]cuda:0" = torch.ops.aten.copy_.default(primals_196, add_171);  primals_196 = add_171 = copy__91 = None
        copy__92: "f32[480][1]cuda:0" = torch.ops.aten.copy_.default(primals_197, add_172);  primals_197 = add_172 = copy__92 = None
        copy__93: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_201, add_175);  primals_201 = add_175 = copy__93 = None
        copy__94: "f32[480][1]cuda:0" = torch.ops.aten.copy_.default(primals_202, add_177);  primals_202 = add_177 = copy__94 = None
        copy__95: "f32[480][1]cuda:0" = torch.ops.aten.copy_.default(primals_203, add_178);  primals_203 = add_178 = copy__95 = None
        copy__96: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_211, add_182);  primals_211 = add_182 = copy__96 = None
        copy__97: "f32[112][1]cuda:0" = torch.ops.aten.copy_.default(primals_212, add_184);  primals_212 = add_184 = copy__97 = None
        copy__98: "f32[112][1]cuda:0" = torch.ops.aten.copy_.default(primals_213, add_185);  primals_213 = add_185 = copy__98 = None
        copy__99: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_217, add_187);  primals_217 = add_187 = copy__99 = None
        copy__100: "f32[672][1]cuda:0" = torch.ops.aten.copy_.default(primals_218, add_189);  primals_218 = add_189 = copy__100 = None
        copy__101: "f32[672][1]cuda:0" = torch.ops.aten.copy_.default(primals_219, add_190);  primals_219 = add_190 = copy__101 = None
        copy__102: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_223, add_193);  primals_223 = add_193 = copy__102 = None
        copy__103: "f32[672][1]cuda:0" = torch.ops.aten.copy_.default(primals_224, add_195);  primals_224 = add_195 = copy__103 = None
        copy__104: "f32[672][1]cuda:0" = torch.ops.aten.copy_.default(primals_225, add_196);  primals_225 = add_196 = copy__104 = None
        copy__105: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_233, add_200);  primals_233 = add_200 = copy__105 = None
        copy__106: "f32[112][1]cuda:0" = torch.ops.aten.copy_.default(primals_234, add_202);  primals_234 = add_202 = copy__106 = None
        copy__107: "f32[112][1]cuda:0" = torch.ops.aten.copy_.default(primals_235, add_203);  primals_235 = add_203 = copy__107 = None
        copy__108: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_239, add_206);  primals_239 = add_206 = copy__108 = None
        copy__109: "f32[672][1]cuda:0" = torch.ops.aten.copy_.default(primals_240, add_208);  primals_240 = add_208 = copy__109 = None
        copy__110: "f32[672][1]cuda:0" = torch.ops.aten.copy_.default(primals_241, add_209);  primals_241 = add_209 = copy__110 = None
        copy__111: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_245, add_212);  primals_245 = add_212 = copy__111 = None
        copy__112: "f32[672][1]cuda:0" = torch.ops.aten.copy_.default(primals_246, add_214);  primals_246 = add_214 = copy__112 = None
        copy__113: "f32[672][1]cuda:0" = torch.ops.aten.copy_.default(primals_247, add_215);  primals_247 = add_215 = copy__113 = None
        copy__114: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_255, add_219);  primals_255 = add_219 = copy__114 = None
        copy__115: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_256, add_221);  primals_256 = add_221 = copy__115 = None
        copy__116: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_257, add_222);  primals_257 = add_222 = copy__116 = None
        copy__117: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_261, add_224);  primals_261 = add_224 = copy__117 = None
        copy__118: "f32[960][1]cuda:0" = torch.ops.aten.copy_.default(primals_262, add_226);  primals_262 = add_226 = copy__118 = None
        copy__119: "f32[960][1]cuda:0" = torch.ops.aten.copy_.default(primals_263, add_227);  primals_263 = add_227 = copy__119 = None
        copy__120: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_267, add_230);  primals_267 = add_230 = copy__120 = None
        copy__121: "f32[960][1]cuda:0" = torch.ops.aten.copy_.default(primals_268, add_232);  primals_268 = add_232 = copy__121 = None
        copy__122: "f32[960][1]cuda:0" = torch.ops.aten.copy_.default(primals_269, add_233);  primals_269 = add_233 = copy__122 = None
        copy__123: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_277, add_237);  primals_277 = add_237 = copy__123 = None
        copy__124: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_278, add_239);  primals_278 = add_239 = copy__124 = None
        copy__125: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_279, add_240);  primals_279 = add_240 = copy__125 = None
        copy__126: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_283, add_243);  primals_283 = add_243 = copy__126 = None
        copy__127: "f32[960][1]cuda:0" = torch.ops.aten.copy_.default(primals_284, add_245);  primals_284 = add_245 = copy__127 = None
        copy__128: "f32[960][1]cuda:0" = torch.ops.aten.copy_.default(primals_285, add_246);  primals_285 = add_246 = copy__128 = None
        copy__129: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_289, add_249);  primals_289 = add_249 = copy__129 = None
        copy__130: "f32[960][1]cuda:0" = torch.ops.aten.copy_.default(primals_290, add_251);  primals_290 = add_251 = copy__130 = None
        copy__131: "f32[960][1]cuda:0" = torch.ops.aten.copy_.default(primals_291, add_252);  primals_291 = add_252 = copy__131 = None
        copy__132: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_299, add_256);  primals_299 = add_256 = copy__132 = None
        copy__133: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_300, add_258);  primals_300 = add_258 = copy__133 = None
        copy__134: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_301, add_259);  primals_301 = add_259 = copy__134 = None
        copy__135: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_305, add_262);  primals_305 = add_262 = copy__135 = None
        copy__136: "f32[960][1]cuda:0" = torch.ops.aten.copy_.default(primals_306, add_264);  primals_306 = add_264 = copy__136 = None
        copy__137: "f32[960][1]cuda:0" = torch.ops.aten.copy_.default(primals_307, add_265);  primals_307 = add_265 = copy__137 = None
        return (addmm, primals_6, primals_7, primals_12, primals_18, primals_24, primals_30, primals_36, primals_42, primals_48, primals_54, primals_60, primals_66, primals_67, primals_76, primals_82, primals_88, primals_89, primals_98, primals_104, primals_110, primals_111, primals_120, primals_126, primals_127, primals_132, primals_133, primals_138, primals_144, primals_145, primals_150, primals_151, primals_156, primals_162, primals_163, primals_168, primals_169, primals_174, primals_180, primals_181, primals_186, primals_187, primals_192, primals_198, primals_199, primals_204, primals_205, primals_214, primals_220, primals_221, primals_226, primals_227, primals_236, primals_242, primals_243, primals_248, primals_249, primals_258, primals_264, primals_265, primals_270, primals_271, primals_280, primals_286, primals_287, primals_292, primals_293, primals_302, primals_308, primals_309, convert_element_type, convert_element_type_1, convolution, getitem_1, rsqrt, convert_element_type_5, convert_element_type_6, convolution_1, squeeze_4, relu, convert_element_type_9, convolution_2, squeeze_7, add_16, convert_element_type_12, convolution_3, squeeze_10, relu_1, convert_element_type_15, convolution_4, squeeze_13, relu_2, convert_element_type_18, convolution_5, squeeze_16, convert_element_type_20, convert_element_type_21, convolution_6, squeeze_19, relu_3, convert_element_type_24, convolution_7, squeeze_22, relu_4, convert_element_type_27, convolution_8, squeeze_25, add_47, convert_element_type_30, convolution_9, squeeze_28, relu_5, convert_element_type_33, convolution_10, getitem_21, rsqrt_10, mean, convert_element_type_37, relu_7, convert_element_type_39, convolution_12, mul_78, convert_element_type_42, convolution_13, squeeze_34, convert_element_type_44, convert_element_type_45, convolution_14, squeeze_37, relu_8, convert_element_type_48, convolution_15, getitem_27, rsqrt_13, mean_1, convert_element_type_52, relu_10, convert_element_type_54, convolution_17, mul_100, convert_element_type_57, convolution_18, squeeze_43, add_80, convert_element_type_60, convolution_19, squeeze_46, relu_11, convert_element_type_63, convolution_20, getitem_33, rsqrt_16, mean_2, convert_element_type_67, relu_13, convert_element_type_69, convolution_22, mul_122, convert_element_type_72, convolution_23, squeeze_52, add_97, convert_element_type_75, convolution_24, getitem_37, rsqrt_18, convert_element_type_79, convert_element_type_80, convolution_25, getitem_39, rsqrt_19, convert_element_type_84, convert_element_type_85, convolution_26, squeeze_61, convert_element_type_87, convert_element_type_88, convolution_27, getitem_43, rsqrt_21, convert_element_type_92, convert_element_type_93, convolution_28, getitem_45, rsqrt_22, convert_element_type_97, convert_element_type_98, convolution_29, squeeze_70, add_132, convert_element_type_101, convolution_30, getitem_49, rsqrt_24, convert_element_type_105, convert_element_type_106, convolution_31, getitem_51, rsqrt_25, convert_element_type_110, convert_element_type_111, convolution_32, squeeze_79, add_150, convert_element_type_114, convolution_33, getitem_55, rsqrt_27, convert_element_type_118, convert_element_type_119, convolution_34, getitem_57, rsqrt_28, convert_element_type_123, convert_element_type_124, convolution_35, squeeze_88, add_168, convert_element_type_127, convolution_36, getitem_61, rsqrt_30, convert_element_type_131, convert_element_type_132, convolution_37, getitem_63, rsqrt_31, mean_3, convert_element_type_138, relu_14, convert_element_type_140, convolution_39, mul_238, convert_element_type_143, convolution_40, squeeze_97, convert_element_type_145, convert_element_type_146, convolution_41, getitem_67, rsqrt_33, convert_element_type_150, convert_element_type_151, convolution_42, getitem_69, rsqrt_34, mean_4, convert_element_type_157, relu_15, convert_element_type_159, convolution_44, mul_262, convert_element_type_162, convolution_45, squeeze_106, add_205, convert_element_type_165, convolution_46, getitem_73, rsqrt_36, convert_element_type_169, convert_element_type_170, convolution_47, getitem_75, rsqrt_37, mean_5, convert_element_type_176, relu_16, convert_element_type_178, convolution_49, mul_286, convert_element_type_181, convolution_50, squeeze_115, convert_element_type_183, convert_element_type_184, convolution_51, getitem_79, rsqrt_39, convert_element_type_188, convert_element_type_189, convolution_52, getitem_81, rsqrt_40, mean_6, convert_element_type_195, relu_17, convert_element_type_197, convolution_54, mul_310, convert_element_type_200, convolution_55, squeeze_124, add_242, convert_element_type_203, convolution_56, getitem_85, rsqrt_42, convert_element_type_207, convert_element_type_208, convolution_57, getitem_87, rsqrt_43, mean_7, convert_element_type_214, relu_18, convert_element_type_216, convolution_59, mul_334, convert_element_type_219, convolution_60, squeeze_133, add_261, convert_element_type_222, convolution_61, getitem_91, rsqrt_45, as_strided, convert_element_type_228, convolution_62, view_1, permute_1, unsqueeze_198, unsqueeze_234, unsqueeze_270, unsqueeze_306, unsqueeze_342, unsqueeze_378, unsqueeze_414, unsqueeze_450, unsqueeze_486, unsqueeze_522, unsqueeze_546, unsqueeze_558, unsqueeze_582, unsqueeze_594, unsqueeze_618, unsqueeze_630, unsqueeze_642, unsqueeze_654, unsqueeze_666, unsqueeze_678, unsqueeze_690, unsqueeze_702, unsqueeze_714)
