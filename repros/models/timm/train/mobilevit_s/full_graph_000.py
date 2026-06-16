class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[16, 3, 3, 3][27, 1, 9, 3]cuda:0", primals_2: "f32[128, 3, 256, 256][196608, 1, 768, 3]cuda:0", primals_3: "i64[][]cuda:0", primals_4: "f32[16][1]cuda:0", primals_5: "f32[16][1]cuda:0", primals_6: "f32[16][1]cuda:0", primals_7: "f32[16][1]cuda:0", primals_8: "f32[64, 16, 1, 1][16, 1, 16, 16]cuda:0", primals_9: "i64[][]cuda:0", primals_10: "f32[64][1]cuda:0", primals_11: "f32[64][1]cuda:0", primals_12: "f32[64][1]cuda:0", primals_13: "f32[64][1]cuda:0", primals_14: "f32[64, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_15: "i64[][]cuda:0", primals_16: "f32[64][1]cuda:0", primals_17: "f32[64][1]cuda:0", primals_18: "f32[64][1]cuda:0", primals_19: "f32[64][1]cuda:0", primals_20: "f32[32, 64, 1, 1][64, 1, 64, 64]cuda:0", primals_21: "i64[][]cuda:0", primals_22: "f32[32][1]cuda:0", primals_23: "f32[32][1]cuda:0", primals_24: "f32[32][1]cuda:0", primals_25: "f32[32][1]cuda:0", primals_26: "f32[128, 32, 1, 1][32, 1, 32, 32]cuda:0", primals_27: "i64[][]cuda:0", primals_28: "f32[128][1]cuda:0", primals_29: "f32[128][1]cuda:0", primals_30: "f32[128][1]cuda:0", primals_31: "f32[128][1]cuda:0", primals_32: "f32[128, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_33: "i64[][]cuda:0", primals_34: "f32[128][1]cuda:0", primals_35: "f32[128][1]cuda:0", primals_36: "f32[128][1]cuda:0", primals_37: "f32[128][1]cuda:0", primals_38: "f32[64, 128, 1, 1][128, 1, 128, 128]cuda:0", primals_39: "i64[][]cuda:0", primals_40: "f32[64][1]cuda:0", primals_41: "f32[64][1]cuda:0", primals_42: "f32[64][1]cuda:0", primals_43: "f32[64][1]cuda:0", primals_44: "f32[256, 64, 1, 1][64, 1, 64, 64]cuda:0", primals_45: "i64[][]cuda:0", primals_46: "f32[256][1]cuda:0", primals_47: "f32[256][1]cuda:0", primals_48: "f32[256][1]cuda:0", primals_49: "f32[256][1]cuda:0", primals_50: "f32[256, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_51: "i64[][]cuda:0", primals_52: "f32[256][1]cuda:0", primals_53: "f32[256][1]cuda:0", primals_54: "f32[256][1]cuda:0", primals_55: "f32[256][1]cuda:0", primals_56: "f32[64, 256, 1, 1][256, 1, 256, 256]cuda:0", primals_57: "i64[][]cuda:0", primals_58: "f32[64][1]cuda:0", primals_59: "f32[64][1]cuda:0", primals_60: "f32[64][1]cuda:0", primals_61: "f32[64][1]cuda:0", primals_62: "f32[256, 64, 1, 1][64, 1, 64, 64]cuda:0", primals_63: "i64[][]cuda:0", primals_64: "f32[256][1]cuda:0", primals_65: "f32[256][1]cuda:0", primals_66: "f32[256][1]cuda:0", primals_67: "f32[256][1]cuda:0", primals_68: "f32[256, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_69: "i64[][]cuda:0", primals_70: "f32[256][1]cuda:0", primals_71: "f32[256][1]cuda:0", primals_72: "f32[256][1]cuda:0", primals_73: "f32[256][1]cuda:0", primals_74: "f32[64, 256, 1, 1][256, 1, 256, 256]cuda:0", primals_75: "i64[][]cuda:0", primals_76: "f32[64][1]cuda:0", primals_77: "f32[64][1]cuda:0", primals_78: "f32[64][1]cuda:0", primals_79: "f32[64][1]cuda:0", primals_80: "f32[256, 64, 1, 1][64, 1, 64, 64]cuda:0", primals_81: "i64[][]cuda:0", primals_82: "f32[256][1]cuda:0", primals_83: "f32[256][1]cuda:0", primals_84: "f32[256][1]cuda:0", primals_85: "f32[256][1]cuda:0", primals_86: "f32[256, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_87: "i64[][]cuda:0", primals_88: "f32[256][1]cuda:0", primals_89: "f32[256][1]cuda:0", primals_90: "f32[256][1]cuda:0", primals_91: "f32[256][1]cuda:0", primals_92: "f32[96, 256, 1, 1][256, 1, 256, 256]cuda:0", primals_93: "i64[][]cuda:0", primals_94: "f32[96][1]cuda:0", primals_95: "f32[96][1]cuda:0", primals_96: "f32[96][1]cuda:0", primals_97: "f32[96][1]cuda:0", primals_98: "f32[96, 96, 3, 3][864, 1, 288, 96]cuda:0", primals_99: "i64[][]cuda:0", primals_100: "f32[96][1]cuda:0", primals_101: "f32[96][1]cuda:0", primals_102: "f32[96][1]cuda:0", primals_103: "f32[96][1]cuda:0", primals_104: "f32[144, 96, 1, 1][96, 1, 96, 96]cuda:0", primals_105: "f32[144][1]cuda:0", primals_106: "f32[144][1]cuda:0", primals_107: "f32[432, 144][144, 1]cuda:0", primals_108: "f32[432][1]cuda:0", primals_109: "f32[144, 144][144, 1]cuda:0", primals_110: "f32[144][1]cuda:0", primals_111: "f32[144][1]cuda:0", primals_112: "f32[144][1]cuda:0", primals_113: "f32[288, 144][144, 1]cuda:0", primals_114: "f32[288][1]cuda:0", primals_115: "f32[144, 288][288, 1]cuda:0", primals_116: "f32[144][1]cuda:0", primals_117: "f32[144][1]cuda:0", primals_118: "f32[144][1]cuda:0", primals_119: "f32[432, 144][144, 1]cuda:0", primals_120: "f32[432][1]cuda:0", primals_121: "f32[144, 144][144, 1]cuda:0", primals_122: "f32[144][1]cuda:0", primals_123: "f32[144][1]cuda:0", primals_124: "f32[144][1]cuda:0", primals_125: "f32[288, 144][144, 1]cuda:0", primals_126: "f32[288][1]cuda:0", primals_127: "f32[144, 288][288, 1]cuda:0", primals_128: "f32[144][1]cuda:0", primals_129: "f32[144][1]cuda:0", primals_130: "f32[144][1]cuda:0", primals_131: "f32[96, 144, 1, 1][144, 1, 144, 144]cuda:0", primals_132: "i64[][]cuda:0", primals_133: "f32[96][1]cuda:0", primals_134: "f32[96][1]cuda:0", primals_135: "f32[96][1]cuda:0", primals_136: "f32[96][1]cuda:0", primals_137: "f32[96, 192, 3, 3][1728, 1, 576, 192]cuda:0", primals_138: "i64[][]cuda:0", primals_139: "f32[96][1]cuda:0", primals_140: "f32[96][1]cuda:0", primals_141: "f32[96][1]cuda:0", primals_142: "f32[96][1]cuda:0", primals_143: "f32[384, 96, 1, 1][96, 1, 96, 96]cuda:0", primals_144: "i64[][]cuda:0", primals_145: "f32[384][1]cuda:0", primals_146: "f32[384][1]cuda:0", primals_147: "f32[384][1]cuda:0", primals_148: "f32[384][1]cuda:0", primals_149: "f32[384, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_150: "i64[][]cuda:0", primals_151: "f32[384][1]cuda:0", primals_152: "f32[384][1]cuda:0", primals_153: "f32[384][1]cuda:0", primals_154: "f32[384][1]cuda:0", primals_155: "f32[128, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_156: "i64[][]cuda:0", primals_157: "f32[128][1]cuda:0", primals_158: "f32[128][1]cuda:0", primals_159: "f32[128][1]cuda:0", primals_160: "f32[128][1]cuda:0", primals_161: "f32[128, 128, 3, 3][1152, 1, 384, 128]cuda:0", primals_162: "i64[][]cuda:0", primals_163: "f32[128][1]cuda:0", primals_164: "f32[128][1]cuda:0", primals_165: "f32[128][1]cuda:0", primals_166: "f32[128][1]cuda:0", primals_167: "f32[192, 128, 1, 1][128, 1, 128, 128]cuda:0", primals_168: "f32[192][1]cuda:0", primals_169: "f32[192][1]cuda:0", primals_170: "f32[576, 192][192, 1]cuda:0", primals_171: "f32[576][1]cuda:0", primals_172: "f32[192, 192][192, 1]cuda:0", primals_173: "f32[192][1]cuda:0", primals_174: "f32[192][1]cuda:0", primals_175: "f32[192][1]cuda:0", primals_176: "f32[384, 192][192, 1]cuda:0", primals_177: "f32[384][1]cuda:0", primals_178: "f32[192, 384][384, 1]cuda:0", primals_179: "f32[192][1]cuda:0", primals_180: "f32[192][1]cuda:0", primals_181: "f32[192][1]cuda:0", primals_182: "f32[576, 192][192, 1]cuda:0", primals_183: "f32[576][1]cuda:0", primals_184: "f32[192, 192][192, 1]cuda:0", primals_185: "f32[192][1]cuda:0", primals_186: "f32[192][1]cuda:0", primals_187: "f32[192][1]cuda:0", primals_188: "f32[384, 192][192, 1]cuda:0", primals_189: "f32[384][1]cuda:0", primals_190: "f32[192, 384][384, 1]cuda:0", primals_191: "f32[192][1]cuda:0", primals_192: "f32[192][1]cuda:0", primals_193: "f32[192][1]cuda:0", primals_194: "f32[576, 192][192, 1]cuda:0", primals_195: "f32[576][1]cuda:0", primals_196: "f32[192, 192][192, 1]cuda:0", primals_197: "f32[192][1]cuda:0", primals_198: "f32[192][1]cuda:0", primals_199: "f32[192][1]cuda:0", primals_200: "f32[384, 192][192, 1]cuda:0", primals_201: "f32[384][1]cuda:0", primals_202: "f32[192, 384][384, 1]cuda:0", primals_203: "f32[192][1]cuda:0", primals_204: "f32[192][1]cuda:0", primals_205: "f32[192][1]cuda:0", primals_206: "f32[576, 192][192, 1]cuda:0", primals_207: "f32[576][1]cuda:0", primals_208: "f32[192, 192][192, 1]cuda:0", primals_209: "f32[192][1]cuda:0", primals_210: "f32[192][1]cuda:0", primals_211: "f32[192][1]cuda:0", primals_212: "f32[384, 192][192, 1]cuda:0", primals_213: "f32[384][1]cuda:0", primals_214: "f32[192, 384][384, 1]cuda:0", primals_215: "f32[192][1]cuda:0", primals_216: "f32[192][1]cuda:0", primals_217: "f32[192][1]cuda:0", primals_218: "f32[128, 192, 1, 1][192, 1, 192, 192]cuda:0", primals_219: "i64[][]cuda:0", primals_220: "f32[128][1]cuda:0", primals_221: "f32[128][1]cuda:0", primals_222: "f32[128][1]cuda:0", primals_223: "f32[128][1]cuda:0", primals_224: "f32[128, 256, 3, 3][2304, 1, 768, 256]cuda:0", primals_225: "i64[][]cuda:0", primals_226: "f32[128][1]cuda:0", primals_227: "f32[128][1]cuda:0", primals_228: "f32[128][1]cuda:0", primals_229: "f32[128][1]cuda:0", primals_230: "f32[512, 128, 1, 1][128, 1, 128, 128]cuda:0", primals_231: "i64[][]cuda:0", primals_232: "f32[512][1]cuda:0", primals_233: "f32[512][1]cuda:0", primals_234: "f32[512][1]cuda:0", primals_235: "f32[512][1]cuda:0", primals_236: "f32[512, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_237: "i64[][]cuda:0", primals_238: "f32[512][1]cuda:0", primals_239: "f32[512][1]cuda:0", primals_240: "f32[512][1]cuda:0", primals_241: "f32[512][1]cuda:0", primals_242: "f32[160, 512, 1, 1][512, 1, 512, 512]cuda:0", primals_243: "i64[][]cuda:0", primals_244: "f32[160][1]cuda:0", primals_245: "f32[160][1]cuda:0", primals_246: "f32[160][1]cuda:0", primals_247: "f32[160][1]cuda:0", primals_248: "f32[160, 160, 3, 3][1440, 1, 480, 160]cuda:0", primals_249: "i64[][]cuda:0", primals_250: "f32[160][1]cuda:0", primals_251: "f32[160][1]cuda:0", primals_252: "f32[160][1]cuda:0", primals_253: "f32[160][1]cuda:0", primals_254: "f32[240, 160, 1, 1][160, 1, 160, 160]cuda:0", primals_255: "f32[240][1]cuda:0", primals_256: "f32[240][1]cuda:0", primals_257: "f32[720, 240][240, 1]cuda:0", primals_258: "f32[720][1]cuda:0", primals_259: "f32[240, 240][240, 1]cuda:0", primals_260: "f32[240][1]cuda:0", primals_261: "f32[240][1]cuda:0", primals_262: "f32[240][1]cuda:0", primals_263: "f32[480, 240][240, 1]cuda:0", primals_264: "f32[480][1]cuda:0", primals_265: "f32[240, 480][480, 1]cuda:0", primals_266: "f32[240][1]cuda:0", primals_267: "f32[240][1]cuda:0", primals_268: "f32[240][1]cuda:0", primals_269: "f32[720, 240][240, 1]cuda:0", primals_270: "f32[720][1]cuda:0", primals_271: "f32[240, 240][240, 1]cuda:0", primals_272: "f32[240][1]cuda:0", primals_273: "f32[240][1]cuda:0", primals_274: "f32[240][1]cuda:0", primals_275: "f32[480, 240][240, 1]cuda:0", primals_276: "f32[480][1]cuda:0", primals_277: "f32[240, 480][480, 1]cuda:0", primals_278: "f32[240][1]cuda:0", primals_279: "f32[240][1]cuda:0", primals_280: "f32[240][1]cuda:0", primals_281: "f32[720, 240][240, 1]cuda:0", primals_282: "f32[720][1]cuda:0", primals_283: "f32[240, 240][240, 1]cuda:0", primals_284: "f32[240][1]cuda:0", primals_285: "f32[240][1]cuda:0", primals_286: "f32[240][1]cuda:0", primals_287: "f32[480, 240][240, 1]cuda:0", primals_288: "f32[480][1]cuda:0", primals_289: "f32[240, 480][480, 1]cuda:0", primals_290: "f32[240][1]cuda:0", primals_291: "f32[240][1]cuda:0", primals_292: "f32[240][1]cuda:0", primals_293: "f32[160, 240, 1, 1][240, 1, 240, 240]cuda:0", primals_294: "i64[][]cuda:0", primals_295: "f32[160][1]cuda:0", primals_296: "f32[160][1]cuda:0", primals_297: "f32[160][1]cuda:0", primals_298: "f32[160][1]cuda:0", primals_299: "f32[160, 320, 3, 3][2880, 1, 960, 320]cuda:0", primals_300: "i64[][]cuda:0", primals_301: "f32[160][1]cuda:0", primals_302: "f32[160][1]cuda:0", primals_303: "f32[160][1]cuda:0", primals_304: "f32[160][1]cuda:0", primals_305: "f32[640, 160, 1, 1][160, 1, 160, 160]cuda:0", primals_306: "i64[][]cuda:0", primals_307: "f32[640][1]cuda:0", primals_308: "f32[640][1]cuda:0", primals_309: "f32[640][1]cuda:0", primals_310: "f32[640][1]cuda:0", primals_311: "f32[1000, 640][640, 1]cuda:0", primals_312: "f32[1000][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type: "bf16[16, 3, 3, 3][27, 1, 9, 3]cuda:0" = torch.ops.prims.convert_element_type.default(primals_1, torch.bfloat16);  primals_1 = None
        convert_element_type_1: "bf16[128, 3, 256, 256][196608, 1, 768, 3]cuda:0" = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16);  primals_2 = None
        convolution: "bf16[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_1, convert_element_type, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_2: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.prims.convert_element_type.default(convolution, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_2, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_2 = None
        getitem: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        add_1: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze: "f32[16][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3])
        mul_1: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze, 0.1);  squeeze = None
        mul_2: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_4, 0.9)
        add_2: "f32[16][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_2: "f32[16][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_3: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_2, 1.0000004768373856);  squeeze_2 = None
        mul_4: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
        mul_5: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_5, 0.9)
        add_3: "f32[16][1]cuda:0" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_6, -1)
        unsqueeze_1: "f32[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_7, -1)
        unsqueeze_3: "f32[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_4: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        convert_element_type_3: "bf16[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_4: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        neg: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.neg.default(convert_element_type_4)
        exp: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.exp.default(neg);  neg = None
        add_5: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_4, add_5);  convert_element_type_4 = add_5 = None
        convert_element_type_5: "bf16[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_6: "bf16[64, 16, 1, 1][16, 1, 16, 16]cuda:0" = torch.ops.prims.convert_element_type.default(primals_8, torch.bfloat16);  primals_8 = None
        convolution_1: "bf16[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_5, convert_element_type_6, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_6: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_9, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_7: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_7, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_7 = None
        getitem_2: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_1[0]
        getitem_3: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        add_7: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
        rsqrt_1: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        sub_1: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
        mul_7: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        squeeze_3: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3])
        mul_8: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1);  squeeze_3 = None
        mul_9: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_10, 0.9)
        add_8: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        squeeze_5: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_10: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000004768373856);  squeeze_5 = None
        mul_11: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
        mul_12: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_11, 0.9)
        add_9: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        unsqueeze_4: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_12, -1)
        unsqueeze_5: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_13: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
        unsqueeze_6: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_13, -1)
        unsqueeze_7: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_10: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
        convert_element_type_8: "bf16[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_10, torch.bfloat16);  add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_9: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_8, torch.float32);  convert_element_type_8 = None
        neg_1: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.neg.default(convert_element_type_9)
        exp_1: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.exp.default(neg_1);  neg_1 = None
        add_11: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.add.Tensor(exp_1, 1);  exp_1 = None
        div_1: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_9, add_11);  convert_element_type_9 = add_11 = None
        convert_element_type_10: "bf16[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_11: "bf16[64, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
        convolution_2: "bf16[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_10, convert_element_type_11, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 64)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_12: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_15, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_12: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_2, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_12, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_12 = None
        getitem_4: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_2[0]
        getitem_5: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        add_13: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
        rsqrt_2: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_2: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
        mul_14: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        squeeze_6: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3])
        mul_15: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1);  squeeze_6 = None
        mul_16: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_16, 0.9)
        add_14: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
        squeeze_8: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_17: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0000004768373856);  squeeze_8 = None
        mul_18: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
        mul_19: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_17, 0.9)
        add_15: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
        unsqueeze_8: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_18, -1)
        unsqueeze_9: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        mul_20: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
        unsqueeze_10: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_19, -1)
        unsqueeze_11: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        add_16: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
        convert_element_type_13: "bf16[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_16, torch.bfloat16);  add_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_14: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_13, torch.float32);  convert_element_type_13 = None
        neg_2: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.neg.default(convert_element_type_14)
        exp_2: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.exp.default(neg_2);  neg_2 = None
        add_17: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.add.Tensor(exp_2, 1);  exp_2 = None
        div_2: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_14, add_17);  convert_element_type_14 = add_17 = None
        convert_element_type_15: "bf16[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_16: "bf16[32, 64, 1, 1][64, 1, 64, 64]cuda:0" = torch.ops.prims.convert_element_type.default(primals_20, torch.bfloat16);  primals_20 = None
        convolution_3: "bf16[128, 32, 128, 128][524288, 1, 4096, 32]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_15, convert_element_type_16, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_18: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_21, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_17: "f32[128, 32, 128, 128][524288, 1, 4096, 32]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_3, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_17, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_17 = None
        getitem_6: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_3[0]
        getitem_7: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        add_19: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
        rsqrt_3: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        sub_3: "f32[128, 32, 128, 128][524288, 1, 4096, 32]cuda:0" = torch.ops.aten.sub.Tensor(convolution_3, getitem_7)
        mul_21: "f32[128, 32, 128, 128][524288, 1, 4096, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        squeeze_9: "f32[32][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        squeeze_10: "f32[32][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_22: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
        mul_23: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_22, 0.9)
        add_20: "f32[32][1]cuda:0" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
        squeeze_11: "f32[32][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_24: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0000004768373856);  squeeze_11 = None
        mul_25: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
        mul_26: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_23, 0.9)
        add_21: "f32[32][1]cuda:0" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
        unsqueeze_12: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_24, -1)
        unsqueeze_13: "f32[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_27: "f32[128, 32, 128, 128][524288, 1, 4096, 32]cuda:0" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
        unsqueeze_14: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_25, -1);  primals_25 = None
        unsqueeze_15: "f32[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_22: "f32[128, 32, 128, 128][524288, 1, 4096, 32]cuda:0" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
        convert_element_type_18: "bf16[128, 32, 128, 128][524288, 1, 4096, 32]cuda:0" = torch.ops.prims.convert_element_type.default(add_22, torch.bfloat16);  add_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_19: "bf16[128, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.prims.convert_element_type.default(primals_26, torch.bfloat16);  primals_26 = None
        convolution_4: "bf16[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_18, convert_element_type_19, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_23: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_27, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_20: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_4, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_20, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_20 = None
        getitem_8: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_4[0]
        getitem_9: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        add_24: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
        rsqrt_4: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        sub_4: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.sub.Tensor(convolution_4, getitem_9)
        mul_28: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        squeeze_12: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3])
        mul_29: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1);  squeeze_12 = None
        mul_30: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_28, 0.9)
        add_25: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
        squeeze_14: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_31: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000004768373856);  squeeze_14 = None
        mul_32: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
        mul_33: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_29, 0.9)
        add_26: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
        unsqueeze_16: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_30, -1)
        unsqueeze_17: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        mul_34: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
        unsqueeze_18: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_31, -1)
        unsqueeze_19: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        add_27: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None
        convert_element_type_21: "bf16[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.prims.convert_element_type.default(add_27, torch.bfloat16);  add_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_22: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_21, torch.float32);  convert_element_type_21 = None
        neg_3: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.neg.default(convert_element_type_22)
        exp_3: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.exp.default(neg_3);  neg_3 = None
        add_28: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.add.Tensor(exp_3, 1);  exp_3 = None
        div_3: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_22, add_28);  convert_element_type_22 = add_28 = None
        convert_element_type_23: "bf16[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_24: "bf16[128, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_32, torch.bfloat16);  primals_32 = None
        convolution_5: "bf16[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_23, convert_element_type_24, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 128)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_29: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_33, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_25: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_5, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_25, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_25 = None
        getitem_10: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_5[0]
        getitem_11: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        add_30: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
        rsqrt_5: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        sub_5: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.sub.Tensor(convolution_5, getitem_11)
        mul_35: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        squeeze_15: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3])
        mul_36: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1);  squeeze_15 = None
        mul_37: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_34, 0.9)
        add_31: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
        squeeze_17: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_38: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0000019073522708);  squeeze_17 = None
        mul_39: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
        mul_40: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_35, 0.9)
        add_32: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
        unsqueeze_20: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_36, -1)
        unsqueeze_21: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_41: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
        unsqueeze_22: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_37, -1)
        unsqueeze_23: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_33: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
        convert_element_type_26: "bf16[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.prims.convert_element_type.default(add_33, torch.bfloat16);  add_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_27: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_26, torch.float32);  convert_element_type_26 = None
        neg_4: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.neg.default(convert_element_type_27)
        exp_4: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.exp.default(neg_4);  neg_4 = None
        add_34: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.add.Tensor(exp_4, 1);  exp_4 = None
        div_4: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_27, add_34);  convert_element_type_27 = add_34 = None
        convert_element_type_28: "bf16[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_29: "bf16[64, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.prims.convert_element_type.default(primals_38, torch.bfloat16);  primals_38 = None
        convolution_6: "bf16[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_28, convert_element_type_29, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_35: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_39, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_30: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_6, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_30, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_30 = None
        getitem_12: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_6[0]
        getitem_13: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        add_36: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
        rsqrt_6: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_6: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_6, getitem_13)
        mul_42: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        squeeze_18: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        squeeze_19: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_43: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
        mul_44: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_40, 0.9)
        add_37: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
        squeeze_20: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
        mul_45: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0000019073522708);  squeeze_20 = None
        mul_46: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_45, 0.1);  mul_45 = None
        mul_47: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_41, 0.9)
        add_38: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
        unsqueeze_24: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_42, -1)
        unsqueeze_25: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        mul_48: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_25);  mul_42 = unsqueeze_25 = None
        unsqueeze_26: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_43, -1);  primals_43 = None
        unsqueeze_27: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        add_39: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_48, unsqueeze_27);  mul_48 = unsqueeze_27 = None
        convert_element_type_31: "bf16[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_39, torch.bfloat16);  add_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_32: "bf16[256, 64, 1, 1][64, 1, 64, 64]cuda:0" = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16);  primals_44 = None
        convolution_7: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_31, convert_element_type_32, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_40: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_45, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_33: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_7, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_33, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_33 = None
        getitem_14: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_7[0]
        getitem_15: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        add_41: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
        rsqrt_7: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
        sub_7: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(convolution_7, getitem_15)
        mul_49: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        squeeze_21: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3])
        mul_50: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1);  squeeze_21 = None
        mul_51: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_46, 0.9)
        add_42: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
        squeeze_23: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
        mul_52: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0000019073522708);  squeeze_23 = None
        mul_53: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_52, 0.1);  mul_52 = None
        mul_54: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_47, 0.9)
        add_43: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
        unsqueeze_28: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_48, -1)
        unsqueeze_29: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_55: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
        unsqueeze_30: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_49, -1)
        unsqueeze_31: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_44: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None
        convert_element_type_34: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(add_44, torch.bfloat16);  add_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_35: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_34, torch.float32);  convert_element_type_34 = None
        neg_5: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.neg.default(convert_element_type_35)
        exp_5: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.exp.default(neg_5);  neg_5 = None
        add_45: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(exp_5, 1);  exp_5 = None
        div_5: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_35, add_45);  convert_element_type_35 = add_45 = None
        convert_element_type_36: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_37: "bf16[256, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_50, torch.bfloat16);  primals_50 = None
        convolution_8: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_36, convert_element_type_37, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 256)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_46: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_51, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_38: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_8, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_38, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_38 = None
        getitem_16: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_8[0]
        getitem_17: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        add_47: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
        rsqrt_8: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        sub_8: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(convolution_8, getitem_17)
        mul_56: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        squeeze_24: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3])
        mul_57: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1);  squeeze_24 = None
        mul_58: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_52, 0.9)
        add_48: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
        squeeze_26: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
        mul_59: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0000019073522708);  squeeze_26 = None
        mul_60: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
        mul_61: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_53, 0.9)
        add_49: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
        unsqueeze_32: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_54, -1)
        unsqueeze_33: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        mul_62: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
        unsqueeze_34: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_55, -1)
        unsqueeze_35: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        add_50: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None
        convert_element_type_39: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(add_50, torch.bfloat16);  add_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_40: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_39, torch.float32);  convert_element_type_39 = None
        neg_6: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.neg.default(convert_element_type_40)
        exp_6: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.exp.default(neg_6);  neg_6 = None
        add_51: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(exp_6, 1);  exp_6 = None
        div_6: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_40, add_51);  convert_element_type_40 = add_51 = None
        convert_element_type_41: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_42: "bf16[64, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(primals_56, torch.bfloat16);  primals_56 = None
        convolution_9: "bf16[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_41, convert_element_type_42, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_52: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_57, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_43: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_9, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_43, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_43 = None
        getitem_18: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_9[0]
        getitem_19: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        add_53: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
        rsqrt_9: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_9: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_9, getitem_19)
        mul_63: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        squeeze_27: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        squeeze_28: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_64: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
        mul_65: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_58, 0.9)
        add_54: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
        squeeze_29: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
        mul_66: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0000019073522708);  squeeze_29 = None
        mul_67: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, 0.1);  mul_66 = None
        mul_68: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_59, 0.9)
        add_55: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
        unsqueeze_36: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_60, -1)
        unsqueeze_37: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_69: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_37);  mul_63 = unsqueeze_37 = None
        unsqueeze_38: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_61, -1);  primals_61 = None
        unsqueeze_39: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_56: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_39);  mul_69 = unsqueeze_39 = None
        convert_element_type_44: "bf16[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_56, torch.bfloat16);  add_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:501 in forward, code: x = x + self.shortcut(shortcut)
        add_57: "bf16[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_44, convert_element_type_31);  convert_element_type_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_45: "bf16[256, 64, 1, 1][64, 1, 64, 64]cuda:0" = torch.ops.prims.convert_element_type.default(primals_62, torch.bfloat16);  primals_62 = None
        convolution_10: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.convolution.default(add_57, convert_element_type_45, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_58: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_63, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_46: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_10, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_46, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_46 = None
        getitem_20: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_10[0]
        getitem_21: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        add_59: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
        rsqrt_10: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
        sub_10: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(convolution_10, getitem_21)
        mul_70: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        squeeze_30: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3])
        mul_71: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1);  squeeze_30 = None
        mul_72: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_64, 0.9)
        add_60: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_71, mul_72);  mul_71 = mul_72 = None
        squeeze_32: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        mul_73: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0000019073522708);  squeeze_32 = None
        mul_74: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_73, 0.1);  mul_73 = None
        mul_75: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_65, 0.9)
        add_61: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
        unsqueeze_40: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_66, -1)
        unsqueeze_41: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        mul_76: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_41);  mul_70 = unsqueeze_41 = None
        unsqueeze_42: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_67, -1)
        unsqueeze_43: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        add_62: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_76, unsqueeze_43);  mul_76 = unsqueeze_43 = None
        convert_element_type_47: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(add_62, torch.bfloat16);  add_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_48: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_47, torch.float32);  convert_element_type_47 = None
        neg_7: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.neg.default(convert_element_type_48)
        exp_7: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.exp.default(neg_7);  neg_7 = None
        add_63: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(exp_7, 1);  exp_7 = None
        div_7: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_48, add_63);  convert_element_type_48 = add_63 = None
        convert_element_type_49: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_50: "bf16[256, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_68, torch.bfloat16);  primals_68 = None
        convolution_11: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_49, convert_element_type_50, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 256)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_64: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_69, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_51: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_11, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_51, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_51 = None
        getitem_22: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_11[0]
        getitem_23: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        add_65: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
        rsqrt_11: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        sub_11: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(convolution_11, getitem_23)
        mul_77: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        squeeze_33: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3])
        mul_78: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1);  squeeze_33 = None
        mul_79: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_70, 0.9)
        add_66: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
        squeeze_35: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
        mul_80: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0000019073522708);  squeeze_35 = None
        mul_81: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_80, 0.1);  mul_80 = None
        mul_82: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_71, 0.9)
        add_67: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
        unsqueeze_44: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_72, -1)
        unsqueeze_45: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_83: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_45);  mul_77 = unsqueeze_45 = None
        unsqueeze_46: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_73, -1)
        unsqueeze_47: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_68: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_47);  mul_83 = unsqueeze_47 = None
        convert_element_type_52: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(add_68, torch.bfloat16);  add_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_53: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_52, torch.float32);  convert_element_type_52 = None
        neg_8: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.neg.default(convert_element_type_53)
        exp_8: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.exp.default(neg_8);  neg_8 = None
        add_69: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(exp_8, 1);  exp_8 = None
        div_8: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_53, add_69);  convert_element_type_53 = add_69 = None
        convert_element_type_54: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_55: "bf16[64, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(primals_74, torch.bfloat16);  primals_74 = None
        convolution_12: "bf16[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_54, convert_element_type_55, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_70: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_75, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_56: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_12, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_56, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_56 = None
        getitem_24: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_12[0]
        getitem_25: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        add_71: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
        rsqrt_12: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        sub_12: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_12, getitem_25)
        mul_84: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        squeeze_36: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        squeeze_37: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_85: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
        mul_86: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_76, 0.9)
        add_72: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
        squeeze_38: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
        mul_87: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0000019073522708);  squeeze_38 = None
        mul_88: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_87, 0.1);  mul_87 = None
        mul_89: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_77, 0.9)
        add_73: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
        unsqueeze_48: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_78, -1)
        unsqueeze_49: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        mul_90: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_49);  mul_84 = unsqueeze_49 = None
        unsqueeze_50: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_79, -1);  primals_79 = None
        unsqueeze_51: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        add_74: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_51);  mul_90 = unsqueeze_51 = None
        convert_element_type_57: "bf16[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_74, torch.bfloat16);  add_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:501 in forward, code: x = x + self.shortcut(shortcut)
        add_75: "bf16[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_57, add_57);  convert_element_type_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_58: "bf16[256, 64, 1, 1][64, 1, 64, 64]cuda:0" = torch.ops.prims.convert_element_type.default(primals_80, torch.bfloat16);  primals_80 = None
        convolution_13: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.convolution.default(add_75, convert_element_type_58, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_76: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_81, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_59: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_13, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_59, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_59 = None
        getitem_26: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_13[0]
        getitem_27: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        add_77: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
        rsqrt_13: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        sub_13: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(convolution_13, getitem_27)
        mul_91: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        squeeze_39: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3])
        mul_92: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1);  squeeze_39 = None
        mul_93: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_82, 0.9)
        add_78: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
        squeeze_41: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        mul_94: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0000019073522708);  squeeze_41 = None
        mul_95: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_94, 0.1);  mul_94 = None
        mul_96: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_83, 0.9)
        add_79: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None
        unsqueeze_52: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_84, -1)
        unsqueeze_53: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_97: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_53);  mul_91 = unsqueeze_53 = None
        unsqueeze_54: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_85, -1)
        unsqueeze_55: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_80: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_55);  mul_97 = unsqueeze_55 = None
        convert_element_type_60: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(add_80, torch.bfloat16);  add_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_61: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_60, torch.float32);  convert_element_type_60 = None
        neg_9: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.neg.default(convert_element_type_61)
        exp_9: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.exp.default(neg_9);  neg_9 = None
        add_81: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(exp_9, 1);  exp_9 = None
        div_9: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_61, add_81);  convert_element_type_61 = add_81 = None
        convert_element_type_62: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_63: "bf16[256, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_86, torch.bfloat16);  primals_86 = None
        convolution_14: "bf16[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_62, convert_element_type_63, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 256)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_82: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_87, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_64: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_14, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_64, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_64 = None
        getitem_28: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_14[0]
        getitem_29: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        add_83: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
        rsqrt_14: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
        sub_14: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.sub.Tensor(convolution_14, getitem_29)
        mul_98: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        squeeze_42: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3])
        mul_99: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1);  squeeze_42 = None
        mul_100: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_88, 0.9)
        add_84: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
        squeeze_44: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        mul_101: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0000076294527394);  squeeze_44 = None
        mul_102: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_101, 0.1);  mul_101 = None
        mul_103: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_89, 0.9)
        add_85: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
        unsqueeze_56: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_90, -1)
        unsqueeze_57: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        mul_104: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_57);  mul_98 = unsqueeze_57 = None
        unsqueeze_58: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_91, -1)
        unsqueeze_59: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        add_86: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_59);  mul_104 = unsqueeze_59 = None
        convert_element_type_65: "bf16[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.prims.convert_element_type.default(add_86, torch.bfloat16);  add_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_66: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_65, torch.float32);  convert_element_type_65 = None
        neg_10: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.neg.default(convert_element_type_66)
        exp_10: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.exp.default(neg_10);  neg_10 = None
        add_87: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.add.Tensor(exp_10, 1);  exp_10 = None
        div_10: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_66, add_87);  convert_element_type_66 = add_87 = None
        convert_element_type_67: "bf16[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_68: "bf16[96, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(primals_92, torch.bfloat16);  primals_92 = None
        convolution_15: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_67, convert_element_type_68, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_88: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_93, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_69: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_15, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_69, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_69 = None
        getitem_30: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = var_mean_15[0]
        getitem_31: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        add_89: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
        rsqrt_15: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
        sub_15: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_15, getitem_31)
        mul_105: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        squeeze_45: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
        squeeze_46: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_106: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
        mul_107: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_94, 0.9)
        add_90: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
        squeeze_47: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
        mul_108: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0000076294527394);  squeeze_47 = None
        mul_109: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_108, 0.1);  mul_108 = None
        mul_110: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_95, 0.9)
        add_91: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
        unsqueeze_60: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_96, -1)
        unsqueeze_61: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_111: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_61);  mul_105 = unsqueeze_61 = None
        unsqueeze_62: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_97, -1);  primals_97 = None
        unsqueeze_63: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_92: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_63);  mul_111 = unsqueeze_63 = None
        convert_element_type_70: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_92, torch.bfloat16);  add_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_71: "bf16[96, 96, 3, 3][864, 1, 288, 96]cuda:0" = torch.ops.prims.convert_element_type.default(primals_98, torch.bfloat16);  primals_98 = None
        convolution_16: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_70, convert_element_type_71, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_93: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_99, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_72: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_16, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_72, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_72 = None
        getitem_32: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = var_mean_16[0]
        getitem_33: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        add_94: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
        rsqrt_16: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        sub_16: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_16, getitem_33)
        mul_112: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        squeeze_48: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3])
        mul_113: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1);  squeeze_48 = None
        mul_114: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_100, 0.9)
        add_95: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
        squeeze_50: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
        mul_115: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0000076294527394);  squeeze_50 = None
        mul_116: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_115, 0.1);  mul_115 = None
        mul_117: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_101, 0.9)
        add_96: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
        unsqueeze_64: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_102, -1)
        unsqueeze_65: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        mul_118: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_65);  mul_112 = unsqueeze_65 = None
        unsqueeze_66: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_103, -1)
        unsqueeze_67: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        add_97: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_67);  mul_118 = unsqueeze_67 = None
        convert_element_type_73: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_97, torch.bfloat16);  add_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_74: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_73, torch.float32);  convert_element_type_73 = None
        neg_11: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.neg.default(convert_element_type_74)
        exp_11: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.exp.default(neg_11);  neg_11 = None
        add_98: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.add.Tensor(exp_11, 1);  exp_11 = None
        div_11: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_74, add_98);  convert_element_type_74 = add_98 = None
        convert_element_type_75: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:243 in forward, code: x = self.conv_1x1(x)
        convert_element_type_76: "bf16[144, 96, 1, 1][96, 1, 96, 96]cuda:0" = torch.ops.prims.convert_element_type.default(primals_104, torch.bfloat16);  primals_104 = None
        convolution_17: "bf16[128, 144, 32, 32][147456, 1, 4608, 144]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_75, convert_element_type_76, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:258 in forward, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
        clone_12: "bf16[128, 144, 32, 32][147456, 1024, 32, 1]cuda:0" = torch.ops.aten.clone.default(convolution_17, memory_format = torch.contiguous_format);  convolution_17 = None
        view: "bf16[294912, 2, 16, 2][64, 32, 2, 1]cuda:0" = torch.ops.aten.reshape.default(clone_12, [294912, 2, 16, 2]);  clone_12 = None
        permute: "bf16[294912, 16, 2, 2][64, 2, 32, 1]cuda:0" = torch.ops.aten.permute.default(view, [0, 2, 1, 3]);  view = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:260 in forward, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
        clone_13: "bf16[294912, 16, 2, 2][64, 4, 2, 1]cuda:0" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_1: "bf16[128, 144, 256, 4][147456, 1024, 4, 1]cuda:0" = torch.ops.aten.reshape.default(clone_13, [128, 144, 256, 4]);  clone_13 = None
        permute_1: "bf16[128, 4, 256, 144][147456, 1, 4, 1024]cuda:0" = torch.ops.aten.permute.default(view_1, [0, 3, 2, 1]);  view_1 = None
        clone_14: "bf16[128, 4, 256, 144][147456, 36864, 144, 1]cuda:0" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_2: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.reshape.default(clone_14, [512, 256, 144]);  clone_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        convert_element_type_77: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_77, [2], correction = 0, keepdim = True)
        getitem_34: "f32[512, 256, 1][256, 1, 1]cuda:0" = var_mean_17[0]
        getitem_35: "f32[512, 256, 1][256, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        add_99: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_17: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
        sub_17: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_77, getitem_35);  convert_element_type_77 = None
        mul_119: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        mul_120: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_119, primals_105);  mul_119 = None
        add_100: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_120, primals_106);  mul_120 = primals_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        convert_element_type_78: "bf16[432][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_108, torch.bfloat16);  primals_108 = None
        convert_element_type_79: "bf16[432, 144][144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_107, torch.bfloat16);  primals_107 = None
        convert_element_type_80: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_100, torch.bfloat16);  add_100 = None
        view_3: "bf16[131072, 144][144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_80, [131072, 144]);  convert_element_type_80 = None
        permute_2: "bf16[144, 432][1, 144]cuda:0" = torch.ops.aten.permute.default(convert_element_type_79, [1, 0]);  convert_element_type_79 = None
        addmm: "bf16[131072, 432][432, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_78, view_3, permute_2);  convert_element_type_78 = None
        view_4: "bf16[512, 256, 432][110592, 432, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [512, 256, 432]);  addmm = None
        view_5: "bf16[512, 256, 3, 4, 36][110592, 432, 144, 36, 1]cuda:0" = torch.ops.aten.reshape.default(view_4, [512, 256, 3, 4, 36]);  view_4 = None
        permute_3: "bf16[3, 512, 4, 256, 36][144, 110592, 36, 432, 1]cuda:0" = torch.ops.aten.permute.default(view_5, [2, 0, 3, 1, 4]);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind = torch.ops.aten.unbind.int(permute_3);  permute_3 = None
        getitem_36: "bf16[512, 4, 256, 36][110592, 36, 432, 1]cuda:0" = unbind[0]
        getitem_37: "bf16[512, 4, 256, 36][110592, 36, 432, 1]cuda:0" = unbind[1]
        getitem_38: "bf16[512, 4, 256, 36][110592, 36, 432, 1]cuda:0" = unbind[2];  unbind = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        constant_pad_nd: "bf16[512, 4, 256, 40][40960, 10240, 40, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_36, [0, 4], 0.0);  getitem_36 = None
        constant_pad_nd_1: "bf16[512, 4, 256, 40][40960, 10240, 40, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_37, [0, 4], 0.0);  getitem_37 = None
        constant_pad_nd_2: "bf16[512, 4, 256, 40][40960, 10240, 40, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_38, [0, 4], 0.0);  getitem_38 = None
        _scaled_dot_product_flash_attention = torch.ops.aten._scaled_dot_product_flash_attention.default(constant_pad_nd, constant_pad_nd_1, constant_pad_nd_2, scale = 0.16666666666666666)
        getitem_39: "bf16[512, 4, 256, 40][40960, 10240, 40, 1]cuda:0" = _scaled_dot_product_flash_attention[0]
        getitem_40: "f32[512, 4, 256][1024, 256, 1]cuda:0" = _scaled_dot_product_flash_attention[1]
        getitem_45: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention[6]
        getitem_46: "u64[][]cuda:0" = _scaled_dot_product_flash_attention[7];  _scaled_dot_product_flash_attention = None
        slice_1: "bf16[512, 4, 256, 36][40960, 10240, 40, 1]cuda:0" = torch.ops.aten.slice.Tensor(getitem_39, -1, 0, 36)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_4: "bf16[512, 256, 4, 36][40960, 40, 10240, 1]cuda:0" = torch.ops.aten.permute.default(slice_1, [0, 2, 1, 3]);  slice_1 = None
        clone_15: "bf16[512, 256, 4, 36][36864, 144, 36, 1]cuda:0" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        view_6: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.reshape.default(clone_15, [512, 256, 144]);  clone_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        convert_element_type_84: "bf16[144][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_110, torch.bfloat16);  primals_110 = None
        convert_element_type_85: "bf16[144, 144][144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_109, torch.bfloat16);  primals_109 = None
        view_7: "bf16[131072, 144][144, 1]cuda:0" = torch.ops.aten.reshape.default(view_6, [131072, 144]);  view_6 = None
        permute_5: "bf16[144, 144][1, 144]cuda:0" = torch.ops.aten.permute.default(convert_element_type_85, [1, 0]);  convert_element_type_85 = None
        addmm_1: "bf16[131072, 144][144, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_84, view_7, permute_5);  convert_element_type_84 = None
        view_8: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [512, 256, 144]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_101: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.add.Tensor(view_2, view_8);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_89: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_101, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_89, [2], correction = 0, keepdim = True)
        getitem_48: "f32[512, 256, 1][256, 1, 1]cuda:0" = var_mean_18[0]
        getitem_49: "f32[512, 256, 1][256, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        add_102: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_18: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        sub_18: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_89, getitem_49);  convert_element_type_89 = None
        mul_121: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        mul_122: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_121, primals_111);  mul_121 = None
        add_103: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_122, primals_112);  mul_122 = primals_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_90: "bf16[288][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_114, torch.bfloat16);  primals_114 = None
        convert_element_type_91: "bf16[288, 144][144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_113, torch.bfloat16);  primals_113 = None
        convert_element_type_92: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_103, torch.bfloat16);  add_103 = None
        view_9: "bf16[131072, 144][144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_92, [131072, 144]);  convert_element_type_92 = None
        permute_6: "bf16[144, 288][1, 144]cuda:0" = torch.ops.aten.permute.default(convert_element_type_91, [1, 0]);  convert_element_type_91 = None
        addmm_2: "bf16[131072, 288][288, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_90, view_9, permute_6);  convert_element_type_90 = None
        view_10: "bf16[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [512, 256, 288])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_96: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_10, torch.float32);  view_10 = None
        neg_12: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_96)
        exp_12: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.exp.default(neg_12);  neg_12 = None
        add_104: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_12, 1);  exp_12 = None
        div_12: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_96, add_104);  convert_element_type_96 = add_104 = None
        convert_element_type_97: "bf16[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_12, torch.bfloat16);  div_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_98: "bf16[144][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_116, torch.bfloat16);  primals_116 = None
        convert_element_type_99: "bf16[144, 288][288, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_115, torch.bfloat16);  primals_115 = None
        view_11: "bf16[131072, 288][288, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_97, [131072, 288]);  convert_element_type_97 = None
        permute_7: "bf16[288, 144][1, 288]cuda:0" = torch.ops.aten.permute.default(convert_element_type_99, [1, 0]);  convert_element_type_99 = None
        addmm_3: "bf16[131072, 144][144, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_98, view_11, permute_7);  convert_element_type_98 = None
        view_12: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [512, 256, 144]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_105: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.add.Tensor(add_101, view_12);  view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        convert_element_type_103: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_105, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_103, [2], correction = 0, keepdim = True)
        getitem_50: "f32[512, 256, 1][256, 1, 1]cuda:0" = var_mean_19[0]
        getitem_51: "f32[512, 256, 1][256, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        add_106: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
        rsqrt_19: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        sub_19: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_103, getitem_51);  convert_element_type_103 = None
        mul_123: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        mul_124: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_123, primals_117);  mul_123 = None
        add_107: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_124, primals_118);  mul_124 = primals_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        convert_element_type_104: "bf16[432][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_120, torch.bfloat16);  primals_120 = None
        convert_element_type_105: "bf16[432, 144][144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_119, torch.bfloat16);  primals_119 = None
        convert_element_type_106: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_107, torch.bfloat16);  add_107 = None
        view_13: "bf16[131072, 144][144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_106, [131072, 144]);  convert_element_type_106 = None
        permute_8: "bf16[144, 432][1, 144]cuda:0" = torch.ops.aten.permute.default(convert_element_type_105, [1, 0]);  convert_element_type_105 = None
        addmm_4: "bf16[131072, 432][432, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_104, view_13, permute_8);  convert_element_type_104 = None
        view_14: "bf16[512, 256, 432][110592, 432, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [512, 256, 432]);  addmm_4 = None
        view_15: "bf16[512, 256, 3, 4, 36][110592, 432, 144, 36, 1]cuda:0" = torch.ops.aten.reshape.default(view_14, [512, 256, 3, 4, 36]);  view_14 = None
        permute_9: "bf16[3, 512, 4, 256, 36][144, 110592, 36, 432, 1]cuda:0" = torch.ops.aten.permute.default(view_15, [2, 0, 3, 1, 4]);  view_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_1 = torch.ops.aten.unbind.int(permute_9);  permute_9 = None
        getitem_52: "bf16[512, 4, 256, 36][110592, 36, 432, 1]cuda:0" = unbind_1[0]
        getitem_53: "bf16[512, 4, 256, 36][110592, 36, 432, 1]cuda:0" = unbind_1[1]
        getitem_54: "bf16[512, 4, 256, 36][110592, 36, 432, 1]cuda:0" = unbind_1[2];  unbind_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        constant_pad_nd_3: "bf16[512, 4, 256, 40][40960, 10240, 40, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_52, [0, 4], 0.0);  getitem_52 = None
        constant_pad_nd_4: "bf16[512, 4, 256, 40][40960, 10240, 40, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_53, [0, 4], 0.0);  getitem_53 = None
        constant_pad_nd_5: "bf16[512, 4, 256, 40][40960, 10240, 40, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_54, [0, 4], 0.0);  getitem_54 = None
        _scaled_dot_product_flash_attention_1 = torch.ops.aten._scaled_dot_product_flash_attention.default(constant_pad_nd_3, constant_pad_nd_4, constant_pad_nd_5, scale = 0.16666666666666666)
        getitem_55: "bf16[512, 4, 256, 40][40960, 10240, 40, 1]cuda:0" = _scaled_dot_product_flash_attention_1[0]
        getitem_56: "f32[512, 4, 256][1024, 256, 1]cuda:0" = _scaled_dot_product_flash_attention_1[1]
        getitem_61: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_1[6]
        getitem_62: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_1[7];  _scaled_dot_product_flash_attention_1 = None
        slice_2: "bf16[512, 4, 256, 36][40960, 10240, 40, 1]cuda:0" = torch.ops.aten.slice.Tensor(getitem_55, -1, 0, 36)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_10: "bf16[512, 256, 4, 36][40960, 40, 10240, 1]cuda:0" = torch.ops.aten.permute.default(slice_2, [0, 2, 1, 3]);  slice_2 = None
        clone_19: "bf16[512, 256, 4, 36][36864, 144, 36, 1]cuda:0" = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format);  permute_10 = None
        view_16: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.reshape.default(clone_19, [512, 256, 144]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        convert_element_type_110: "bf16[144][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_122, torch.bfloat16);  primals_122 = None
        convert_element_type_111: "bf16[144, 144][144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_121, torch.bfloat16);  primals_121 = None
        view_17: "bf16[131072, 144][144, 1]cuda:0" = torch.ops.aten.reshape.default(view_16, [131072, 144]);  view_16 = None
        permute_11: "bf16[144, 144][1, 144]cuda:0" = torch.ops.aten.permute.default(convert_element_type_111, [1, 0]);  convert_element_type_111 = None
        addmm_5: "bf16[131072, 144][144, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_110, view_17, permute_11);  convert_element_type_110 = None
        view_18: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [512, 256, 144]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_108: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.add.Tensor(add_105, view_18);  view_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_115: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_108, torch.float32)
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_115, [2], correction = 0, keepdim = True)
        getitem_64: "f32[512, 256, 1][256, 1, 1]cuda:0" = var_mean_20[0]
        getitem_65: "f32[512, 256, 1][256, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None
        add_109: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_20: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
        sub_20: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_115, getitem_65);  convert_element_type_115 = None
        mul_125: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
        mul_126: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_125, primals_123);  mul_125 = None
        add_110: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_126, primals_124);  mul_126 = primals_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_116: "bf16[288][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_126, torch.bfloat16);  primals_126 = None
        convert_element_type_117: "bf16[288, 144][144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_125, torch.bfloat16);  primals_125 = None
        convert_element_type_118: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_110, torch.bfloat16);  add_110 = None
        view_19: "bf16[131072, 144][144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_118, [131072, 144]);  convert_element_type_118 = None
        permute_12: "bf16[144, 288][1, 144]cuda:0" = torch.ops.aten.permute.default(convert_element_type_117, [1, 0]);  convert_element_type_117 = None
        addmm_6: "bf16[131072, 288][288, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_116, view_19, permute_12);  convert_element_type_116 = None
        view_20: "bf16[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [512, 256, 288])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_122: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_20, torch.float32);  view_20 = None
        neg_13: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_122)
        exp_13: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.exp.default(neg_13);  neg_13 = None
        add_111: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_13, 1);  exp_13 = None
        div_13: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_122, add_111);  convert_element_type_122 = add_111 = None
        convert_element_type_123: "bf16[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_124: "bf16[144][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_128, torch.bfloat16);  primals_128 = None
        convert_element_type_125: "bf16[144, 288][288, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_127, torch.bfloat16);  primals_127 = None
        view_21: "bf16[131072, 288][288, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_123, [131072, 288]);  convert_element_type_123 = None
        permute_13: "bf16[288, 144][1, 288]cuda:0" = torch.ops.aten.permute.default(convert_element_type_125, [1, 0]);  convert_element_type_125 = None
        addmm_7: "bf16[131072, 144][144, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_124, view_21, permute_13);  convert_element_type_124 = None
        view_22: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [512, 256, 144]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_112: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.add.Tensor(add_108, view_22);  view_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:264 in forward, code: x = self.norm(x)
        convert_element_type_129: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_112, torch.float32)
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_129, [2], correction = 0, keepdim = True)
        getitem_66: "f32[512, 256, 1][256, 1, 1]cuda:0" = var_mean_21[0]
        getitem_67: "f32[512, 256, 1][256, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        add_113: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_21: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
        sub_21: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_129, getitem_67);  convert_element_type_129 = None
        mul_127: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
        mul_128: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_127, primals_129);  mul_127 = None
        add_114: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_128, primals_130);  mul_128 = primals_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:268 in forward, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
        view_23: "f32[128, 4, 256, 144][147456, 36864, 144, 1]cuda:0" = torch.ops.aten.reshape.default(add_114, [128, 4, 256, -1]);  add_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:269 in forward, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
        permute_14: "f32[128, 144, 256, 4][147456, 1, 144, 36864]cuda:0" = torch.ops.aten.permute.default(view_23, [0, 3, 2, 1]);  view_23 = None
        clone_23: "f32[128, 144, 256, 4][147456, 1024, 4, 1]cuda:0" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
        view_24: "f32[294912, 16, 2, 2][64, 4, 2, 1]cuda:0" = torch.ops.aten.reshape.default(clone_23, [294912, 16, 2, 2]);  clone_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:271 in forward, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
        permute_15: "f32[294912, 2, 16, 2][64, 2, 4, 1]cuda:0" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        clone_24: "f32[294912, 2, 16, 2][64, 32, 2, 1]cuda:0" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
        view_25: "f32[128, 144, 32, 32][147456, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_24, [128, 144, 32, 32]);  clone_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_130: "bf16[96, 144, 1, 1][144, 1, 144, 144]cuda:0" = torch.ops.prims.convert_element_type.default(primals_131, torch.bfloat16);  primals_131 = None
        convert_element_type_131: "bf16[128, 144, 32, 32][147456, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_25, torch.bfloat16);  view_25 = None
        convolution_18: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_131, convert_element_type_130, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_115: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_132, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_132: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_18, torch.float32)
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_132, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_132 = None
        getitem_68: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = var_mean_22[0]
        getitem_69: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None
        add_116: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_68, 1e-05)
        rsqrt_22: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        sub_22: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_18, getitem_69)
        mul_129: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        squeeze_51: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3])
        mul_130: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1);  squeeze_51 = None
        mul_131: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_133, 0.9)
        add_117: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
        squeeze_53: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
        mul_132: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0000076294527394);  squeeze_53 = None
        mul_133: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_132, 0.1);  mul_132 = None
        mul_134: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_134, 0.9)
        add_118: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(mul_133, mul_134);  mul_133 = mul_134 = None
        unsqueeze_68: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_135, -1)
        unsqueeze_69: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_135: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_129, unsqueeze_69);  mul_129 = unsqueeze_69 = None
        unsqueeze_70: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_136, -1)
        unsqueeze_71: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_119: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_135, unsqueeze_71);  mul_135 = unsqueeze_71 = None
        convert_element_type_133: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_119, torch.bfloat16);  add_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_134: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_133, torch.float32);  convert_element_type_133 = None
        neg_14: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.neg.default(convert_element_type_134)
        exp_14: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.exp.default(neg_14);  neg_14 = None
        add_120: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.add.Tensor(exp_14, 1);  exp_14 = None
        div_14: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_134, add_120);  convert_element_type_134 = add_120 = None
        convert_element_type_135: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(div_14, torch.bfloat16);  div_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:277 in forward, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
        cat: "bf16[128, 192, 32, 32][196608, 1, 6144, 192]cuda:0" = torch.ops.aten.cat.default([convert_element_type_70, convert_element_type_135], 1);  convert_element_type_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_136: "bf16[96, 192, 3, 3][1728, 1, 576, 192]cuda:0" = torch.ops.prims.convert_element_type.default(primals_137, torch.bfloat16);  primals_137 = None
        convolution_19: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.convolution.default(cat, convert_element_type_136, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_121: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_138, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_137: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_19, torch.float32)
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_137, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_137 = None
        getitem_70: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = var_mean_23[0]
        getitem_71: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        add_122: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_70, 1e-05)
        rsqrt_23: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        sub_23: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_19, getitem_71)
        mul_136: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
        squeeze_54: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3])
        mul_137: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1);  squeeze_54 = None
        mul_138: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_139, 0.9)
        add_123: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
        squeeze_56: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
        mul_139: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0000076294527394);  squeeze_56 = None
        mul_140: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_139, 0.1);  mul_139 = None
        mul_141: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_140, 0.9)
        add_124: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(mul_140, mul_141);  mul_140 = mul_141 = None
        unsqueeze_72: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_141, -1)
        unsqueeze_73: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        mul_142: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_73);  mul_136 = unsqueeze_73 = None
        unsqueeze_74: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_142, -1)
        unsqueeze_75: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        add_125: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_142, unsqueeze_75);  mul_142 = unsqueeze_75 = None
        convert_element_type_138: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_125, torch.bfloat16);  add_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_139: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_138, torch.float32);  convert_element_type_138 = None
        neg_15: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.neg.default(convert_element_type_139)
        exp_15: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.exp.default(neg_15);  neg_15 = None
        add_126: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.add.Tensor(exp_15, 1);  exp_15 = None
        div_15: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_139, add_126);  convert_element_type_139 = add_126 = None
        convert_element_type_140: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_141: "bf16[384, 96, 1, 1][96, 1, 96, 96]cuda:0" = torch.ops.prims.convert_element_type.default(primals_143, torch.bfloat16);  primals_143 = None
        convolution_20: "bf16[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_140, convert_element_type_141, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_127: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_144, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_142: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_20, torch.float32)
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_142, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_142 = None
        getitem_72: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_24[0]
        getitem_73: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None
        add_128: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_72, 1e-05)
        rsqrt_24: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
        sub_24: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_20, getitem_73)
        mul_143: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        squeeze_57: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3])
        mul_144: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1);  squeeze_57 = None
        mul_145: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_145, 0.9)
        add_129: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
        squeeze_59: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
        mul_146: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0000076294527394);  squeeze_59 = None
        mul_147: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_146, 0.1);  mul_146 = None
        mul_148: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_146, 0.9)
        add_130: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_147, mul_148);  mul_147 = mul_148 = None
        unsqueeze_76: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_147, -1)
        unsqueeze_77: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_149: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_143, unsqueeze_77);  mul_143 = unsqueeze_77 = None
        unsqueeze_78: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_148, -1)
        unsqueeze_79: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_131: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_149, unsqueeze_79);  mul_149 = unsqueeze_79 = None
        convert_element_type_143: "bf16[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_131, torch.bfloat16);  add_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_144: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_143, torch.float32);  convert_element_type_143 = None
        neg_16: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.neg.default(convert_element_type_144)
        exp_16: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.exp.default(neg_16);  neg_16 = None
        add_132: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_16, 1);  exp_16 = None
        div_16: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_144, add_132);  convert_element_type_144 = add_132 = None
        convert_element_type_145: "bf16[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.prims.convert_element_type.default(div_16, torch.bfloat16);  div_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_146: "bf16[384, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_149, torch.bfloat16);  primals_149 = None
        convolution_21: "bf16[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_145, convert_element_type_146, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 384)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_133: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_150, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_147: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_21, torch.float32)
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_147, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_147 = None
        getitem_74: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_25[0]
        getitem_75: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_25[1];  var_mean_25 = None
        add_134: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_74, 1e-05)
        rsqrt_25: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        sub_25: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_21, getitem_75)
        mul_150: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        squeeze_60: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3])
        mul_151: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1);  squeeze_60 = None
        mul_152: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_151, 0.9)
        add_135: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
        squeeze_62: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
        mul_153: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_62, 1.000030518509476);  squeeze_62 = None
        mul_154: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_153, 0.1);  mul_153 = None
        mul_155: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_152, 0.9)
        add_136: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_154, mul_155);  mul_154 = mul_155 = None
        unsqueeze_80: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_153, -1)
        unsqueeze_81: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        mul_156: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_150, unsqueeze_81);  mul_150 = unsqueeze_81 = None
        unsqueeze_82: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_154, -1)
        unsqueeze_83: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        add_137: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_156, unsqueeze_83);  mul_156 = unsqueeze_83 = None
        convert_element_type_148: "bf16[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_137, torch.bfloat16);  add_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_149: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_148, torch.float32);  convert_element_type_148 = None
        neg_17: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.neg.default(convert_element_type_149)
        exp_17: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.exp.default(neg_17);  neg_17 = None
        add_138: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_17, 1);  exp_17 = None
        div_17: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_149, add_138);  convert_element_type_149 = add_138 = None
        convert_element_type_150: "bf16[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.prims.convert_element_type.default(div_17, torch.bfloat16);  div_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_151: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_155, torch.bfloat16);  primals_155 = None
        convolution_22: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_150, convert_element_type_151, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_139: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_156, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_152: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_22, torch.float32)
        var_mean_26 = torch.ops.aten.var_mean.correction(convert_element_type_152, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_152 = None
        getitem_76: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_26[0]
        getitem_77: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_26[1];  var_mean_26 = None
        add_140: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_76, 1e-05)
        rsqrt_26: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
        sub_26: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(convolution_22, getitem_77)
        mul_157: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
        squeeze_63: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
        squeeze_64: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
        mul_158: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
        mul_159: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_157, 0.9)
        add_141: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None
        squeeze_65: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
        mul_160: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_65, 1.000030518509476);  squeeze_65 = None
        mul_161: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_160, 0.1);  mul_160 = None
        mul_162: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_158, 0.9)
        add_142: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_161, mul_162);  mul_161 = mul_162 = None
        unsqueeze_84: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_159, -1)
        unsqueeze_85: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_163: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_157, unsqueeze_85);  mul_157 = unsqueeze_85 = None
        unsqueeze_86: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_160, -1);  primals_160 = None
        unsqueeze_87: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_143: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_163, unsqueeze_87);  mul_163 = unsqueeze_87 = None
        convert_element_type_153: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(add_143, torch.bfloat16);  add_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_154: "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = torch.ops.prims.convert_element_type.default(primals_161, torch.bfloat16);  primals_161 = None
        convolution_23: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_153, convert_element_type_154, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_144: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_162, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_155: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_23, torch.float32)
        var_mean_27 = torch.ops.aten.var_mean.correction(convert_element_type_155, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_155 = None
        getitem_78: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_27[0]
        getitem_79: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_27[1];  var_mean_27 = None
        add_145: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_78, 1e-05)
        rsqrt_27: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
        sub_27: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(convolution_23, getitem_79)
        mul_164: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
        squeeze_66: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3])
        mul_165: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1);  squeeze_66 = None
        mul_166: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_163, 0.9)
        add_146: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
        squeeze_68: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
        mul_167: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_68, 1.000030518509476);  squeeze_68 = None
        mul_168: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_167, 0.1);  mul_167 = None
        mul_169: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_164, 0.9)
        add_147: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_168, mul_169);  mul_168 = mul_169 = None
        unsqueeze_88: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_165, -1)
        unsqueeze_89: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        mul_170: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_164, unsqueeze_89);  mul_164 = unsqueeze_89 = None
        unsqueeze_90: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_166, -1)
        unsqueeze_91: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        add_148: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_170, unsqueeze_91);  mul_170 = unsqueeze_91 = None
        convert_element_type_156: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(add_148, torch.bfloat16);  add_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_157: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_156, torch.float32);  convert_element_type_156 = None
        neg_18: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.neg.default(convert_element_type_157)
        exp_18: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.exp.default(neg_18);  neg_18 = None
        add_149: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.add.Tensor(exp_18, 1);  exp_18 = None
        div_18: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_157, add_149);  convert_element_type_157 = add_149 = None
        convert_element_type_158: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(div_18, torch.bfloat16);  div_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:243 in forward, code: x = self.conv_1x1(x)
        convert_element_type_159: "bf16[192, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.prims.convert_element_type.default(primals_167, torch.bfloat16);  primals_167 = None
        convolution_24: "bf16[128, 192, 16, 16][49152, 1, 3072, 192]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_158, convert_element_type_159, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:258 in forward, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
        clone_30: "bf16[128, 192, 16, 16][49152, 256, 16, 1]cuda:0" = torch.ops.aten.clone.default(convolution_24, memory_format = torch.contiguous_format);  convolution_24 = None
        view_26: "bf16[196608, 2, 8, 2][32, 16, 2, 1]cuda:0" = torch.ops.aten.reshape.default(clone_30, [196608, 2, 8, 2]);  clone_30 = None
        permute_16: "bf16[196608, 8, 2, 2][32, 2, 16, 1]cuda:0" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:260 in forward, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
        clone_31: "bf16[196608, 8, 2, 2][32, 4, 2, 1]cuda:0" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        view_27: "bf16[128, 192, 64, 4][49152, 256, 4, 1]cuda:0" = torch.ops.aten.reshape.default(clone_31, [128, 192, 64, 4]);  clone_31 = None
        permute_17: "bf16[128, 4, 64, 192][49152, 1, 4, 256]cuda:0" = torch.ops.aten.permute.default(view_27, [0, 3, 2, 1]);  view_27 = None
        clone_32: "bf16[128, 4, 64, 192][49152, 12288, 192, 1]cuda:0" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
        view_28: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(clone_32, [512, 64, 192]);  clone_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        convert_element_type_160: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_28, torch.float32)
        var_mean_28 = torch.ops.aten.var_mean.correction(convert_element_type_160, [2], correction = 0, keepdim = True)
        getitem_80: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_28[0]
        getitem_81: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_28[1];  var_mean_28 = None
        add_150: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
        rsqrt_28: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
        sub_28: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_160, getitem_81);  convert_element_type_160 = None
        mul_171: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
        mul_172: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_171, primals_168);  mul_171 = None
        add_151: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_172, primals_169);  mul_172 = primals_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        convert_element_type_161: "bf16[576][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_171, torch.bfloat16);  primals_171 = None
        convert_element_type_162: "bf16[576, 192][192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_170, torch.bfloat16);  primals_170 = None
        convert_element_type_163: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_151, torch.bfloat16);  add_151 = None
        view_29: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_163, [32768, 192]);  convert_element_type_163 = None
        permute_18: "bf16[192, 576][1, 192]cuda:0" = torch.ops.aten.permute.default(convert_element_type_162, [1, 0]);  convert_element_type_162 = None
        addmm_8: "bf16[32768, 576][576, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_161, view_29, permute_18);  convert_element_type_161 = None
        view_30: "bf16[512, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [512, 64, 576]);  addmm_8 = None
        view_31: "bf16[512, 64, 3, 4, 48][36864, 576, 192, 48, 1]cuda:0" = torch.ops.aten.reshape.default(view_30, [512, 64, 3, 4, 48]);  view_30 = None
        permute_19: "bf16[3, 512, 4, 64, 48][192, 36864, 48, 576, 1]cuda:0" = torch.ops.aten.permute.default(view_31, [2, 0, 3, 1, 4]);  view_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_2 = torch.ops.aten.unbind.int(permute_19);  permute_19 = None
        getitem_82: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0" = unbind_2[0]
        getitem_83: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0" = unbind_2[1]
        getitem_84: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0" = unbind_2[2];  unbind_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_82, getitem_83, getitem_84, None, True)
        getitem_85: "bf16[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = _scaled_dot_product_cudnn_attention[0]
        getitem_86: "f32[512, 4, 64, 1][256, 64, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention[1]
        getitem_91: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention[6]
        getitem_92: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention[7];  _scaled_dot_product_cudnn_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_20: "bf16[512, 64, 4, 48][12288, 192, 48, 1]cuda:0" = torch.ops.aten.permute.default(getitem_85, [0, 2, 1, 3])
        view_32: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(permute_20, [512, 64, 192]);  permute_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        convert_element_type_167: "bf16[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_173, torch.bfloat16);  primals_173 = None
        convert_element_type_168: "bf16[192, 192][192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_172, torch.bfloat16);  primals_172 = None
        view_33: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(view_32, [32768, 192]);  view_32 = None
        permute_21: "bf16[192, 192][1, 192]cuda:0" = torch.ops.aten.permute.default(convert_element_type_168, [1, 0]);  convert_element_type_168 = None
        addmm_9: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_167, view_33, permute_21);  convert_element_type_167 = view_33 = None
        view_34: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [512, 64, 192]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_152: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_28, view_34);  view_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_172: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_152, torch.float32)
        var_mean_29 = torch.ops.aten.var_mean.correction(convert_element_type_172, [2], correction = 0, keepdim = True)
        getitem_94: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_29[0]
        getitem_95: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_29[1];  var_mean_29 = None
        add_153: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
        rsqrt_29: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
        sub_29: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_172, getitem_95);  convert_element_type_172 = None
        mul_173: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
        mul_174: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_173, primals_174);  mul_173 = None
        add_154: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_174, primals_175);  mul_174 = primals_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_173: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_177, torch.bfloat16);  primals_177 = None
        convert_element_type_174: "bf16[384, 192][192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_176, torch.bfloat16);  primals_176 = None
        convert_element_type_175: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_154, torch.bfloat16);  add_154 = None
        view_35: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_175, [32768, 192]);  convert_element_type_175 = None
        permute_22: "bf16[192, 384][1, 192]cuda:0" = torch.ops.aten.permute.default(convert_element_type_174, [1, 0]);  convert_element_type_174 = None
        addmm_10: "bf16[32768, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_173, view_35, permute_22);  convert_element_type_173 = None
        view_36: "bf16[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [512, 64, 384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_179: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_36, torch.float32);  view_36 = None
        neg_19: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_179)
        exp_19: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.exp.default(neg_19);  neg_19 = None
        add_155: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_19, 1);  exp_19 = None
        div_19: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_179, add_155);  convert_element_type_179 = add_155 = None
        convert_element_type_180: "bf16[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16);  div_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_181: "bf16[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_179, torch.bfloat16);  primals_179 = None
        convert_element_type_182: "bf16[192, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_178, torch.bfloat16);  primals_178 = None
        view_37: "bf16[32768, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_180, [32768, 384]);  convert_element_type_180 = None
        permute_23: "bf16[384, 192][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_182, [1, 0]);  convert_element_type_182 = None
        addmm_11: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_181, view_37, permute_23);  convert_element_type_181 = None
        view_38: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [512, 64, 192]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_156: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_152, view_38);  view_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        convert_element_type_186: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_156, torch.float32)
        var_mean_30 = torch.ops.aten.var_mean.correction(convert_element_type_186, [2], correction = 0, keepdim = True)
        getitem_96: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_30[0]
        getitem_97: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_30[1];  var_mean_30 = None
        add_157: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
        rsqrt_30: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
        sub_30: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_186, getitem_97);  convert_element_type_186 = None
        mul_175: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
        mul_176: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_175, primals_180);  mul_175 = None
        add_158: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_176, primals_181);  mul_176 = primals_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        convert_element_type_187: "bf16[576][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_183, torch.bfloat16);  primals_183 = None
        convert_element_type_188: "bf16[576, 192][192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_182, torch.bfloat16);  primals_182 = None
        convert_element_type_189: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_158, torch.bfloat16);  add_158 = None
        view_39: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_189, [32768, 192]);  convert_element_type_189 = None
        permute_24: "bf16[192, 576][1, 192]cuda:0" = torch.ops.aten.permute.default(convert_element_type_188, [1, 0]);  convert_element_type_188 = None
        addmm_12: "bf16[32768, 576][576, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_187, view_39, permute_24);  convert_element_type_187 = None
        view_40: "bf16[512, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [512, 64, 576]);  addmm_12 = None
        view_41: "bf16[512, 64, 3, 4, 48][36864, 576, 192, 48, 1]cuda:0" = torch.ops.aten.reshape.default(view_40, [512, 64, 3, 4, 48]);  view_40 = None
        permute_25: "bf16[3, 512, 4, 64, 48][192, 36864, 48, 576, 1]cuda:0" = torch.ops.aten.permute.default(view_41, [2, 0, 3, 1, 4]);  view_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_3 = torch.ops.aten.unbind.int(permute_25);  permute_25 = None
        getitem_98: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0" = unbind_3[0]
        getitem_99: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0" = unbind_3[1]
        getitem_100: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0" = unbind_3[2];  unbind_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_1 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_98, getitem_99, getitem_100, None, True)
        getitem_101: "bf16[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = _scaled_dot_product_cudnn_attention_1[0]
        getitem_102: "f32[512, 4, 64, 1][256, 64, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_1[1]
        getitem_107: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_1[6]
        getitem_108: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_1[7];  _scaled_dot_product_cudnn_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_26: "bf16[512, 64, 4, 48][12288, 192, 48, 1]cuda:0" = torch.ops.aten.permute.default(getitem_101, [0, 2, 1, 3])
        view_42: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(permute_26, [512, 64, 192]);  permute_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        convert_element_type_193: "bf16[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_185, torch.bfloat16);  primals_185 = None
        convert_element_type_194: "bf16[192, 192][192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_184, torch.bfloat16);  primals_184 = None
        view_43: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(view_42, [32768, 192]);  view_42 = None
        permute_27: "bf16[192, 192][1, 192]cuda:0" = torch.ops.aten.permute.default(convert_element_type_194, [1, 0]);  convert_element_type_194 = None
        addmm_13: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_193, view_43, permute_27);  convert_element_type_193 = view_43 = None
        view_44: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [512, 64, 192]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_159: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_156, view_44);  view_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_198: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_159, torch.float32)
        var_mean_31 = torch.ops.aten.var_mean.correction(convert_element_type_198, [2], correction = 0, keepdim = True)
        getitem_110: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_31[0]
        getitem_111: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_31[1];  var_mean_31 = None
        add_160: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
        rsqrt_31: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
        sub_31: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_198, getitem_111);  convert_element_type_198 = None
        mul_177: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
        mul_178: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_177, primals_186);  mul_177 = None
        add_161: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_178, primals_187);  mul_178 = primals_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_199: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_189, torch.bfloat16);  primals_189 = None
        convert_element_type_200: "bf16[384, 192][192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_188, torch.bfloat16);  primals_188 = None
        convert_element_type_201: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_161, torch.bfloat16);  add_161 = None
        view_45: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_201, [32768, 192]);  convert_element_type_201 = None
        permute_28: "bf16[192, 384][1, 192]cuda:0" = torch.ops.aten.permute.default(convert_element_type_200, [1, 0]);  convert_element_type_200 = None
        addmm_14: "bf16[32768, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_199, view_45, permute_28);  convert_element_type_199 = None
        view_46: "bf16[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [512, 64, 384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_205: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_46, torch.float32);  view_46 = None
        neg_20: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_205)
        exp_20: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.exp.default(neg_20);  neg_20 = None
        add_162: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_20, 1);  exp_20 = None
        div_20: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_205, add_162);  convert_element_type_205 = add_162 = None
        convert_element_type_206: "bf16[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_20, torch.bfloat16);  div_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_207: "bf16[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_191, torch.bfloat16);  primals_191 = None
        convert_element_type_208: "bf16[192, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_190, torch.bfloat16);  primals_190 = None
        view_47: "bf16[32768, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_206, [32768, 384]);  convert_element_type_206 = None
        permute_29: "bf16[384, 192][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_208, [1, 0]);  convert_element_type_208 = None
        addmm_15: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_207, view_47, permute_29);  convert_element_type_207 = None
        view_48: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [512, 64, 192]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_163: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_159, view_48);  view_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        convert_element_type_212: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_163, torch.float32)
        var_mean_32 = torch.ops.aten.var_mean.correction(convert_element_type_212, [2], correction = 0, keepdim = True)
        getitem_112: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_32[0]
        getitem_113: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_32[1];  var_mean_32 = None
        add_164: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
        rsqrt_32: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
        sub_32: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_212, getitem_113);  convert_element_type_212 = None
        mul_179: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
        mul_180: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_179, primals_192);  mul_179 = None
        add_165: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_180, primals_193);  mul_180 = primals_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        convert_element_type_213: "bf16[576][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_195, torch.bfloat16);  primals_195 = None
        convert_element_type_214: "bf16[576, 192][192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_194, torch.bfloat16);  primals_194 = None
        convert_element_type_215: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_165, torch.bfloat16);  add_165 = None
        view_49: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_215, [32768, 192]);  convert_element_type_215 = None
        permute_30: "bf16[192, 576][1, 192]cuda:0" = torch.ops.aten.permute.default(convert_element_type_214, [1, 0]);  convert_element_type_214 = None
        addmm_16: "bf16[32768, 576][576, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_213, view_49, permute_30);  convert_element_type_213 = None
        view_50: "bf16[512, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [512, 64, 576]);  addmm_16 = None
        view_51: "bf16[512, 64, 3, 4, 48][36864, 576, 192, 48, 1]cuda:0" = torch.ops.aten.reshape.default(view_50, [512, 64, 3, 4, 48]);  view_50 = None
        permute_31: "bf16[3, 512, 4, 64, 48][192, 36864, 48, 576, 1]cuda:0" = torch.ops.aten.permute.default(view_51, [2, 0, 3, 1, 4]);  view_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_4 = torch.ops.aten.unbind.int(permute_31);  permute_31 = None
        getitem_114: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0" = unbind_4[0]
        getitem_115: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0" = unbind_4[1]
        getitem_116: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0" = unbind_4[2];  unbind_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_2 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_114, getitem_115, getitem_116, None, True)
        getitem_117: "bf16[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = _scaled_dot_product_cudnn_attention_2[0]
        getitem_118: "f32[512, 4, 64, 1][256, 64, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_2[1]
        getitem_123: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_2[6]
        getitem_124: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_2[7];  _scaled_dot_product_cudnn_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_32: "bf16[512, 64, 4, 48][12288, 192, 48, 1]cuda:0" = torch.ops.aten.permute.default(getitem_117, [0, 2, 1, 3])
        view_52: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(permute_32, [512, 64, 192]);  permute_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        convert_element_type_219: "bf16[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_197, torch.bfloat16);  primals_197 = None
        convert_element_type_220: "bf16[192, 192][192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_196, torch.bfloat16);  primals_196 = None
        view_53: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(view_52, [32768, 192]);  view_52 = None
        permute_33: "bf16[192, 192][1, 192]cuda:0" = torch.ops.aten.permute.default(convert_element_type_220, [1, 0]);  convert_element_type_220 = None
        addmm_17: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_219, view_53, permute_33);  convert_element_type_219 = view_53 = None
        view_54: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [512, 64, 192]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_166: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_163, view_54);  view_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_224: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_166, torch.float32)
        var_mean_33 = torch.ops.aten.var_mean.correction(convert_element_type_224, [2], correction = 0, keepdim = True)
        getitem_126: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_33[0]
        getitem_127: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_33[1];  var_mean_33 = None
        add_167: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_126, 1e-05);  getitem_126 = None
        rsqrt_33: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_167);  add_167 = None
        sub_33: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_224, getitem_127);  convert_element_type_224 = None
        mul_181: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
        mul_182: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_181, primals_198);  mul_181 = None
        add_168: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_182, primals_199);  mul_182 = primals_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_225: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_201, torch.bfloat16);  primals_201 = None
        convert_element_type_226: "bf16[384, 192][192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_200, torch.bfloat16);  primals_200 = None
        convert_element_type_227: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_168, torch.bfloat16);  add_168 = None
        view_55: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_227, [32768, 192]);  convert_element_type_227 = None
        permute_34: "bf16[192, 384][1, 192]cuda:0" = torch.ops.aten.permute.default(convert_element_type_226, [1, 0]);  convert_element_type_226 = None
        addmm_18: "bf16[32768, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_225, view_55, permute_34);  convert_element_type_225 = None
        view_56: "bf16[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [512, 64, 384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_231: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_56, torch.float32);  view_56 = None
        neg_21: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_231)
        exp_21: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.exp.default(neg_21);  neg_21 = None
        add_169: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_21, 1);  exp_21 = None
        div_21: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_231, add_169);  convert_element_type_231 = add_169 = None
        convert_element_type_232: "bf16[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16);  div_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_233: "bf16[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_203, torch.bfloat16);  primals_203 = None
        convert_element_type_234: "bf16[192, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_202, torch.bfloat16);  primals_202 = None
        view_57: "bf16[32768, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_232, [32768, 384]);  convert_element_type_232 = None
        permute_35: "bf16[384, 192][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_234, [1, 0]);  convert_element_type_234 = None
        addmm_19: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_233, view_57, permute_35);  convert_element_type_233 = None
        view_58: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [512, 64, 192]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_170: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_166, view_58);  view_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        convert_element_type_238: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_170, torch.float32)
        var_mean_34 = torch.ops.aten.var_mean.correction(convert_element_type_238, [2], correction = 0, keepdim = True)
        getitem_128: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_34[0]
        getitem_129: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_34[1];  var_mean_34 = None
        add_171: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_128, 1e-05);  getitem_128 = None
        rsqrt_34: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
        sub_34: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_238, getitem_129);  convert_element_type_238 = None
        mul_183: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
        mul_184: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_183, primals_204);  mul_183 = None
        add_172: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_184, primals_205);  mul_184 = primals_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        convert_element_type_239: "bf16[576][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_207, torch.bfloat16);  primals_207 = None
        convert_element_type_240: "bf16[576, 192][192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_206, torch.bfloat16);  primals_206 = None
        convert_element_type_241: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_172, torch.bfloat16);  add_172 = None
        view_59: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_241, [32768, 192]);  convert_element_type_241 = None
        permute_36: "bf16[192, 576][1, 192]cuda:0" = torch.ops.aten.permute.default(convert_element_type_240, [1, 0]);  convert_element_type_240 = None
        addmm_20: "bf16[32768, 576][576, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_239, view_59, permute_36);  convert_element_type_239 = None
        view_60: "bf16[512, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [512, 64, 576]);  addmm_20 = None
        view_61: "bf16[512, 64, 3, 4, 48][36864, 576, 192, 48, 1]cuda:0" = torch.ops.aten.reshape.default(view_60, [512, 64, 3, 4, 48]);  view_60 = None
        permute_37: "bf16[3, 512, 4, 64, 48][192, 36864, 48, 576, 1]cuda:0" = torch.ops.aten.permute.default(view_61, [2, 0, 3, 1, 4]);  view_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_5 = torch.ops.aten.unbind.int(permute_37);  permute_37 = None
        getitem_130: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0" = unbind_5[0]
        getitem_131: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0" = unbind_5[1]
        getitem_132: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0" = unbind_5[2];  unbind_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_3 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_130, getitem_131, getitem_132, None, True)
        getitem_133: "bf16[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = _scaled_dot_product_cudnn_attention_3[0]
        getitem_134: "f32[512, 4, 64, 1][256, 64, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_3[1]
        getitem_139: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_3[6]
        getitem_140: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_3[7];  _scaled_dot_product_cudnn_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_38: "bf16[512, 64, 4, 48][12288, 192, 48, 1]cuda:0" = torch.ops.aten.permute.default(getitem_133, [0, 2, 1, 3])
        view_62: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(permute_38, [512, 64, 192]);  permute_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        convert_element_type_245: "bf16[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_209, torch.bfloat16);  primals_209 = None
        convert_element_type_246: "bf16[192, 192][192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_208, torch.bfloat16);  primals_208 = None
        view_63: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(view_62, [32768, 192]);  view_62 = None
        permute_39: "bf16[192, 192][1, 192]cuda:0" = torch.ops.aten.permute.default(convert_element_type_246, [1, 0]);  convert_element_type_246 = None
        addmm_21: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_245, view_63, permute_39);  convert_element_type_245 = view_63 = None
        view_64: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [512, 64, 192]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_173: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_170, view_64);  view_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_250: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_173, torch.float32)
        var_mean_35 = torch.ops.aten.var_mean.correction(convert_element_type_250, [2], correction = 0, keepdim = True)
        getitem_142: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_35[0]
        getitem_143: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_35[1];  var_mean_35 = None
        add_174: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_142, 1e-05);  getitem_142 = None
        rsqrt_35: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        sub_35: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_250, getitem_143);  convert_element_type_250 = None
        mul_185: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
        mul_186: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_185, primals_210);  mul_185 = None
        add_175: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_186, primals_211);  mul_186 = primals_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_251: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_213, torch.bfloat16);  primals_213 = None
        convert_element_type_252: "bf16[384, 192][192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_212, torch.bfloat16);  primals_212 = None
        convert_element_type_253: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_175, torch.bfloat16);  add_175 = None
        view_65: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_253, [32768, 192]);  convert_element_type_253 = None
        permute_40: "bf16[192, 384][1, 192]cuda:0" = torch.ops.aten.permute.default(convert_element_type_252, [1, 0]);  convert_element_type_252 = None
        addmm_22: "bf16[32768, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_251, view_65, permute_40);  convert_element_type_251 = None
        view_66: "bf16[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [512, 64, 384])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_257: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_66, torch.float32);  view_66 = None
        neg_22: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_257)
        exp_22: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.exp.default(neg_22);  neg_22 = None
        add_176: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_22, 1);  exp_22 = None
        div_22: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_257, add_176);  convert_element_type_257 = add_176 = None
        convert_element_type_258: "bf16[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_22, torch.bfloat16);  div_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_259: "bf16[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_215, torch.bfloat16);  primals_215 = None
        convert_element_type_260: "bf16[192, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_214, torch.bfloat16);  primals_214 = None
        view_67: "bf16[32768, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_258, [32768, 384]);  convert_element_type_258 = None
        permute_41: "bf16[384, 192][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_260, [1, 0]);  convert_element_type_260 = None
        addmm_23: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_259, view_67, permute_41);  convert_element_type_259 = None
        view_68: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [512, 64, 192]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_177: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_173, view_68);  view_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:264 in forward, code: x = self.norm(x)
        convert_element_type_264: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_177, torch.float32)
        var_mean_36 = torch.ops.aten.var_mean.correction(convert_element_type_264, [2], correction = 0, keepdim = True)
        getitem_144: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_36[0]
        getitem_145: "f32[512, 64, 1][64, 1, 1]cuda:0" = var_mean_36[1];  var_mean_36 = None
        add_178: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_144, 1e-05);  getitem_144 = None
        rsqrt_36: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
        sub_36: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_264, getitem_145);  convert_element_type_264 = None
        mul_187: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
        mul_188: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_187, primals_216);  mul_187 = None
        add_179: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_188, primals_217);  mul_188 = primals_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:268 in forward, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
        view_69: "f32[128, 4, 64, 192][49152, 12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(add_179, [128, 4, 64, -1]);  add_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:269 in forward, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
        permute_42: "f32[128, 192, 64, 4][49152, 1, 192, 12288]cuda:0" = torch.ops.aten.permute.default(view_69, [0, 3, 2, 1]);  view_69 = None
        clone_45: "f32[128, 192, 64, 4][49152, 256, 4, 1]cuda:0" = torch.ops.aten.clone.default(permute_42, memory_format = torch.contiguous_format);  permute_42 = None
        view_70: "f32[196608, 8, 2, 2][32, 4, 2, 1]cuda:0" = torch.ops.aten.reshape.default(clone_45, [196608, 8, 2, 2]);  clone_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:271 in forward, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
        permute_43: "f32[196608, 2, 8, 2][32, 2, 4, 1]cuda:0" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
        clone_46: "f32[196608, 2, 8, 2][32, 16, 2, 1]cuda:0" = torch.ops.aten.clone.default(permute_43, memory_format = torch.contiguous_format);  permute_43 = None
        view_71: "f32[128, 192, 16, 16][49152, 256, 16, 1]cuda:0" = torch.ops.aten.reshape.default(clone_46, [128, 192, 16, 16]);  clone_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_265: "bf16[128, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(primals_218, torch.bfloat16);  primals_218 = None
        convert_element_type_266: "bf16[128, 192, 16, 16][49152, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_71, torch.bfloat16);  view_71 = None
        convolution_25: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_266, convert_element_type_265, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_180: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_219, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_267: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_25, torch.float32)
        var_mean_37 = torch.ops.aten.var_mean.correction(convert_element_type_267, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_267 = None
        getitem_146: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_37[0]
        getitem_147: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_37[1];  var_mean_37 = None
        add_181: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_146, 1e-05)
        rsqrt_37: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
        sub_37: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(convolution_25, getitem_147)
        mul_189: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
        squeeze_69: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_147, [0, 2, 3])
        mul_190: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1);  squeeze_69 = None
        mul_191: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_220, 0.9)
        add_182: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
        squeeze_71: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_146, [0, 2, 3]);  getitem_146 = None
        mul_192: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_71, 1.000030518509476);  squeeze_71 = None
        mul_193: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_192, 0.1);  mul_192 = None
        mul_194: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_221, 0.9)
        add_183: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_193, mul_194);  mul_193 = mul_194 = None
        unsqueeze_92: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_222, -1)
        unsqueeze_93: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        mul_195: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_189, unsqueeze_93);  mul_189 = unsqueeze_93 = None
        unsqueeze_94: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_223, -1)
        unsqueeze_95: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        add_184: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_195, unsqueeze_95);  mul_195 = unsqueeze_95 = None
        convert_element_type_268: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(add_184, torch.bfloat16);  add_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_269: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_268, torch.float32);  convert_element_type_268 = None
        neg_23: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.neg.default(convert_element_type_269)
        exp_23: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.exp.default(neg_23);  neg_23 = None
        add_185: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.add.Tensor(exp_23, 1);  exp_23 = None
        div_23: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_269, add_185);  convert_element_type_269 = add_185 = None
        convert_element_type_270: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(div_23, torch.bfloat16);  div_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:277 in forward, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
        cat_1: "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = torch.ops.aten.cat.default([convert_element_type_153, convert_element_type_270], 1);  convert_element_type_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_271: "bf16[128, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(primals_224, torch.bfloat16);  primals_224 = None
        convolution_26: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.convolution.default(cat_1, convert_element_type_271, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_186: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_225, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_272: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_26, torch.float32)
        var_mean_38 = torch.ops.aten.var_mean.correction(convert_element_type_272, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_272 = None
        getitem_148: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_38[0]
        getitem_149: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_38[1];  var_mean_38 = None
        add_187: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_148, 1e-05)
        rsqrt_38: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_187);  add_187 = None
        sub_38: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(convolution_26, getitem_149)
        mul_196: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
        squeeze_72: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_149, [0, 2, 3])
        mul_197: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1);  squeeze_72 = None
        mul_198: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_226, 0.9)
        add_188: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
        squeeze_74: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_148, [0, 2, 3]);  getitem_148 = None
        mul_199: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_74, 1.000030518509476);  squeeze_74 = None
        mul_200: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_199, 0.1);  mul_199 = None
        mul_201: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_227, 0.9)
        add_189: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_200, mul_201);  mul_200 = mul_201 = None
        unsqueeze_96: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_228, -1)
        unsqueeze_97: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        mul_202: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_97);  mul_196 = unsqueeze_97 = None
        unsqueeze_98: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_229, -1)
        unsqueeze_99: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        add_190: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_99);  mul_202 = unsqueeze_99 = None
        convert_element_type_273: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(add_190, torch.bfloat16);  add_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_274: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_273, torch.float32);  convert_element_type_273 = None
        neg_24: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.neg.default(convert_element_type_274)
        exp_24: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.exp.default(neg_24);  neg_24 = None
        add_191: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.add.Tensor(exp_24, 1);  exp_24 = None
        div_24: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_274, add_191);  convert_element_type_274 = add_191 = None
        convert_element_type_275: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(div_24, torch.bfloat16);  div_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_276: "bf16[512, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.prims.convert_element_type.default(primals_230, torch.bfloat16);  primals_230 = None
        convolution_27: "bf16[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_275, convert_element_type_276, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_192: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_231, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_277: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_27, torch.float32)
        var_mean_39 = torch.ops.aten.var_mean.correction(convert_element_type_277, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_277 = None
        getitem_150: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_39[0]
        getitem_151: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_39[1];  var_mean_39 = None
        add_193: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_150, 1e-05)
        rsqrt_39: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_193);  add_193 = None
        sub_39: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.sub.Tensor(convolution_27, getitem_151)
        mul_203: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
        squeeze_75: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_151, [0, 2, 3])
        mul_204: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1);  squeeze_75 = None
        mul_205: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_232, 0.9)
        add_194: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None
        squeeze_77: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_150, [0, 2, 3]);  getitem_150 = None
        mul_206: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_77, 1.000030518509476);  squeeze_77 = None
        mul_207: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_206, 0.1);  mul_206 = None
        mul_208: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_233, 0.9)
        add_195: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_207, mul_208);  mul_207 = mul_208 = None
        unsqueeze_100: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_234, -1)
        unsqueeze_101: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_209: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_101);  mul_203 = unsqueeze_101 = None
        unsqueeze_102: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_235, -1)
        unsqueeze_103: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_196: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_103);  mul_209 = unsqueeze_103 = None
        convert_element_type_278: "bf16[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.prims.convert_element_type.default(add_196, torch.bfloat16);  add_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_279: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_278, torch.float32);  convert_element_type_278 = None
        neg_25: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.neg.default(convert_element_type_279)
        exp_25: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.exp.default(neg_25);  neg_25 = None
        add_197: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.add.Tensor(exp_25, 1);  exp_25 = None
        div_25: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_279, add_197);  convert_element_type_279 = add_197 = None
        convert_element_type_280: "bf16[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.prims.convert_element_type.default(div_25, torch.bfloat16);  div_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_281: "bf16[512, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_236, torch.bfloat16);  primals_236 = None
        convolution_28: "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_280, convert_element_type_281, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 512)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_198: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_237, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_282: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_28, torch.float32)
        var_mean_40 = torch.ops.aten.var_mean.correction(convert_element_type_282, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_282 = None
        getitem_152: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_40[0]
        getitem_153: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_40[1];  var_mean_40 = None
        add_199: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_152, 1e-05)
        rsqrt_40: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
        sub_40: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.sub.Tensor(convolution_28, getitem_153)
        mul_210: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
        squeeze_78: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_153, [0, 2, 3])
        mul_211: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1);  squeeze_78 = None
        mul_212: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_238, 0.9)
        add_200: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_211, mul_212);  mul_211 = mul_212 = None
        squeeze_80: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_152, [0, 2, 3]);  getitem_152 = None
        mul_213: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0001220852154804);  squeeze_80 = None
        mul_214: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_213, 0.1);  mul_213 = None
        mul_215: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_239, 0.9)
        add_201: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
        unsqueeze_104: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_240, -1)
        unsqueeze_105: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        mul_216: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_105);  mul_210 = unsqueeze_105 = None
        unsqueeze_106: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_241, -1)
        unsqueeze_107: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
        add_202: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.add.Tensor(mul_216, unsqueeze_107);  mul_216 = unsqueeze_107 = None
        convert_element_type_283: "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.prims.convert_element_type.default(add_202, torch.bfloat16);  add_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_284: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_283, torch.float32);  convert_element_type_283 = None
        neg_26: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.neg.default(convert_element_type_284)
        exp_26: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.exp.default(neg_26);  neg_26 = None
        add_203: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.add.Tensor(exp_26, 1);  exp_26 = None
        div_26: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_284, add_203);  convert_element_type_284 = add_203 = None
        convert_element_type_285: "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.prims.convert_element_type.default(div_26, torch.bfloat16);  div_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_286: "bf16[160, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.prims.convert_element_type.default(primals_242, torch.bfloat16);  primals_242 = None
        convolution_29: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_285, convert_element_type_286, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_204: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_243, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_287: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_29, torch.float32)
        var_mean_41 = torch.ops.aten.var_mean.correction(convert_element_type_287, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_287 = None
        getitem_154: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_41[0]
        getitem_155: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_41[1];  var_mean_41 = None
        add_205: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_154, 1e-05)
        rsqrt_41: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_205);  add_205 = None
        sub_41: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(convolution_29, getitem_155)
        mul_217: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
        squeeze_81: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_155, [0, 2, 3]);  getitem_155 = None
        squeeze_82: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
        mul_218: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
        mul_219: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_244, 0.9)
        add_206: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_218, mul_219);  mul_218 = mul_219 = None
        squeeze_83: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_154, [0, 2, 3]);  getitem_154 = None
        mul_220: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0001220852154804);  squeeze_83 = None
        mul_221: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_220, 0.1);  mul_220 = None
        mul_222: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_245, 0.9)
        add_207: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
        unsqueeze_108: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_246, -1)
        unsqueeze_109: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
        mul_223: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_109);  mul_217 = unsqueeze_109 = None
        unsqueeze_110: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_247, -1);  primals_247 = None
        unsqueeze_111: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
        add_208: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.add.Tensor(mul_223, unsqueeze_111);  mul_223 = unsqueeze_111 = None
        convert_element_type_288: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(add_208, torch.bfloat16);  add_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_289: "bf16[160, 160, 3, 3][1440, 1, 480, 160]cuda:0" = torch.ops.prims.convert_element_type.default(primals_248, torch.bfloat16);  primals_248 = None
        convolution_30: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_288, convert_element_type_289, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_209: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_249, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_290: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_30, torch.float32)
        var_mean_42 = torch.ops.aten.var_mean.correction(convert_element_type_290, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_290 = None
        getitem_156: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_42[0]
        getitem_157: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_42[1];  var_mean_42 = None
        add_210: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_156, 1e-05)
        rsqrt_42: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_210);  add_210 = None
        sub_42: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(convolution_30, getitem_157)
        mul_224: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
        squeeze_84: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_157, [0, 2, 3])
        mul_225: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1);  squeeze_84 = None
        mul_226: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_250, 0.9)
        add_211: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
        squeeze_86: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_156, [0, 2, 3]);  getitem_156 = None
        mul_227: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0001220852154804);  squeeze_86 = None
        mul_228: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_227, 0.1);  mul_227 = None
        mul_229: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_251, 0.9)
        add_212: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_228, mul_229);  mul_228 = mul_229 = None
        unsqueeze_112: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_252, -1)
        unsqueeze_113: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        mul_230: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_224, unsqueeze_113);  mul_224 = unsqueeze_113 = None
        unsqueeze_114: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_253, -1)
        unsqueeze_115: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        add_213: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_115);  mul_230 = unsqueeze_115 = None
        convert_element_type_291: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(add_213, torch.bfloat16);  add_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_292: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_291, torch.float32);  convert_element_type_291 = None
        neg_27: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.neg.default(convert_element_type_292)
        exp_27: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.exp.default(neg_27);  neg_27 = None
        add_214: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.add.Tensor(exp_27, 1);  exp_27 = None
        div_27: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_292, add_214);  convert_element_type_292 = add_214 = None
        convert_element_type_293: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(div_27, torch.bfloat16);  div_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:243 in forward, code: x = self.conv_1x1(x)
        convert_element_type_294: "bf16[240, 160, 1, 1][160, 1, 160, 160]cuda:0" = torch.ops.prims.convert_element_type.default(primals_254, torch.bfloat16);  primals_254 = None
        convolution_31: "bf16[128, 240, 8, 8][15360, 1, 1920, 240]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_293, convert_element_type_294, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:258 in forward, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
        clone_52: "bf16[128, 240, 8, 8][15360, 64, 8, 1]cuda:0" = torch.ops.aten.clone.default(convolution_31, memory_format = torch.contiguous_format);  convolution_31 = None
        view_72: "bf16[122880, 2, 4, 2][16, 8, 2, 1]cuda:0" = torch.ops.aten.reshape.default(clone_52, [122880, 2, 4, 2]);  clone_52 = None
        permute_44: "bf16[122880, 4, 2, 2][16, 2, 8, 1]cuda:0" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:260 in forward, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
        clone_53: "bf16[122880, 4, 2, 2][16, 4, 2, 1]cuda:0" = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
        view_73: "bf16[128, 240, 16, 4][15360, 64, 4, 1]cuda:0" = torch.ops.aten.reshape.default(clone_53, [128, 240, 16, 4]);  clone_53 = None
        permute_45: "bf16[128, 4, 16, 240][15360, 1, 4, 64]cuda:0" = torch.ops.aten.permute.default(view_73, [0, 3, 2, 1]);  view_73 = None
        clone_54: "bf16[128, 4, 16, 240][15360, 3840, 240, 1]cuda:0" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
        view_74: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(clone_54, [512, 16, 240]);  clone_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        convert_element_type_295: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_74, torch.float32)
        var_mean_43 = torch.ops.aten.var_mean.correction(convert_element_type_295, [2], correction = 0, keepdim = True)
        getitem_158: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_43[0]
        getitem_159: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_43[1];  var_mean_43 = None
        add_215: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_158, 1e-05);  getitem_158 = None
        rsqrt_43: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_215);  add_215 = None
        sub_43: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_295, getitem_159);  convert_element_type_295 = None
        mul_231: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
        mul_232: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_231, primals_255);  mul_231 = None
        add_216: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_232, primals_256);  mul_232 = primals_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        convert_element_type_296: "bf16[720][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_258, torch.bfloat16);  primals_258 = None
        convert_element_type_297: "bf16[720, 240][240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_257, torch.bfloat16);  primals_257 = None
        convert_element_type_298: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_216, torch.bfloat16);  add_216 = None
        view_75: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_298, [8192, 240]);  convert_element_type_298 = None
        permute_46: "bf16[240, 720][1, 240]cuda:0" = torch.ops.aten.permute.default(convert_element_type_297, [1, 0]);  convert_element_type_297 = None
        addmm_24: "bf16[8192, 720][720, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_296, view_75, permute_46);  convert_element_type_296 = None
        view_76: "bf16[512, 16, 720][11520, 720, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_24, [512, 16, 720]);  addmm_24 = None
        view_77: "bf16[512, 16, 3, 4, 60][11520, 720, 240, 60, 1]cuda:0" = torch.ops.aten.reshape.default(view_76, [512, 16, 3, 4, 60]);  view_76 = None
        permute_47: "bf16[3, 512, 4, 16, 60][240, 11520, 60, 720, 1]cuda:0" = torch.ops.aten.permute.default(view_77, [2, 0, 3, 1, 4]);  view_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_6 = torch.ops.aten.unbind.int(permute_47);  permute_47 = None
        getitem_160: "bf16[512, 4, 16, 60][11520, 60, 720, 1]cuda:0" = unbind_6[0]
        getitem_161: "bf16[512, 4, 16, 60][11520, 60, 720, 1]cuda:0" = unbind_6[1]
        getitem_162: "bf16[512, 4, 16, 60][11520, 60, 720, 1]cuda:0" = unbind_6[2];  unbind_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        constant_pad_nd_6: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_160, [0, 4], 0.0);  getitem_160 = None
        constant_pad_nd_7: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_161, [0, 4], 0.0);  getitem_161 = None
        constant_pad_nd_8: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_162, [0, 4], 0.0);  getitem_162 = None
        _scaled_dot_product_flash_attention_2 = torch.ops.aten._scaled_dot_product_flash_attention.default(constant_pad_nd_6, constant_pad_nd_7, constant_pad_nd_8, scale = 0.12909944487358055)
        getitem_163: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = _scaled_dot_product_flash_attention_2[0]
        getitem_164: "f32[512, 4, 16][64, 16, 1]cuda:0" = _scaled_dot_product_flash_attention_2[1]
        getitem_169: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_2[6]
        getitem_170: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_2[7];  _scaled_dot_product_flash_attention_2 = None
        slice_3: "bf16[512, 4, 16, 60][4096, 1024, 64, 1]cuda:0" = torch.ops.aten.slice.Tensor(getitem_163, -1, 0, 60)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_48: "bf16[512, 16, 4, 60][4096, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(slice_3, [0, 2, 1, 3]);  slice_3 = None
        clone_55: "bf16[512, 16, 4, 60][3840, 240, 60, 1]cuda:0" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
        view_78: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(clone_55, [512, 16, 240]);  clone_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        convert_element_type_302: "bf16[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_260, torch.bfloat16);  primals_260 = None
        convert_element_type_303: "bf16[240, 240][240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_259, torch.bfloat16);  primals_259 = None
        view_79: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.reshape.default(view_78, [8192, 240]);  view_78 = None
        permute_49: "bf16[240, 240][1, 240]cuda:0" = torch.ops.aten.permute.default(convert_element_type_303, [1, 0]);  convert_element_type_303 = None
        addmm_25: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_302, view_79, permute_49);  convert_element_type_302 = None
        view_80: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [512, 16, 240]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_217: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(view_74, view_80);  view_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_307: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_217, torch.float32)
        var_mean_44 = torch.ops.aten.var_mean.correction(convert_element_type_307, [2], correction = 0, keepdim = True)
        getitem_172: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_44[0]
        getitem_173: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_44[1];  var_mean_44 = None
        add_218: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_172, 1e-05);  getitem_172 = None
        rsqrt_44: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
        sub_44: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_307, getitem_173);  convert_element_type_307 = None
        mul_233: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
        mul_234: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_233, primals_261);  mul_233 = None
        add_219: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_234, primals_262);  mul_234 = primals_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_308: "bf16[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_264, torch.bfloat16);  primals_264 = None
        convert_element_type_309: "bf16[480, 240][240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_263, torch.bfloat16);  primals_263 = None
        convert_element_type_310: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_219, torch.bfloat16);  add_219 = None
        view_81: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_310, [8192, 240]);  convert_element_type_310 = None
        permute_50: "bf16[240, 480][1, 240]cuda:0" = torch.ops.aten.permute.default(convert_element_type_309, [1, 0]);  convert_element_type_309 = None
        addmm_26: "bf16[8192, 480][480, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_308, view_81, permute_50);  convert_element_type_308 = None
        view_82: "bf16[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [512, 16, 480])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_314: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_82, torch.float32);  view_82 = None
        neg_28: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_314)
        exp_28: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.exp.default(neg_28);  neg_28 = None
        add_220: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_28, 1);  exp_28 = None
        div_28: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_314, add_220);  convert_element_type_314 = add_220 = None
        convert_element_type_315: "bf16[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_28, torch.bfloat16);  div_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_316: "bf16[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_266, torch.bfloat16);  primals_266 = None
        convert_element_type_317: "bf16[240, 480][480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_265, torch.bfloat16);  primals_265 = None
        view_83: "bf16[8192, 480][480, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_315, [8192, 480]);  convert_element_type_315 = None
        permute_51: "bf16[480, 240][1, 480]cuda:0" = torch.ops.aten.permute.default(convert_element_type_317, [1, 0]);  convert_element_type_317 = None
        addmm_27: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_316, view_83, permute_51);  convert_element_type_316 = None
        view_84: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_27, [512, 16, 240]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_221: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(add_217, view_84);  view_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        convert_element_type_321: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_221, torch.float32)
        var_mean_45 = torch.ops.aten.var_mean.correction(convert_element_type_321, [2], correction = 0, keepdim = True)
        getitem_174: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_45[0]
        getitem_175: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_45[1];  var_mean_45 = None
        add_222: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_174, 1e-05);  getitem_174 = None
        rsqrt_45: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_222);  add_222 = None
        sub_45: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_321, getitem_175);  convert_element_type_321 = None
        mul_235: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
        mul_236: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_235, primals_267);  mul_235 = None
        add_223: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_236, primals_268);  mul_236 = primals_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        convert_element_type_322: "bf16[720][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_270, torch.bfloat16);  primals_270 = None
        convert_element_type_323: "bf16[720, 240][240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_269, torch.bfloat16);  primals_269 = None
        convert_element_type_324: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_223, torch.bfloat16);  add_223 = None
        view_85: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_324, [8192, 240]);  convert_element_type_324 = None
        permute_52: "bf16[240, 720][1, 240]cuda:0" = torch.ops.aten.permute.default(convert_element_type_323, [1, 0]);  convert_element_type_323 = None
        addmm_28: "bf16[8192, 720][720, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_322, view_85, permute_52);  convert_element_type_322 = None
        view_86: "bf16[512, 16, 720][11520, 720, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [512, 16, 720]);  addmm_28 = None
        view_87: "bf16[512, 16, 3, 4, 60][11520, 720, 240, 60, 1]cuda:0" = torch.ops.aten.reshape.default(view_86, [512, 16, 3, 4, 60]);  view_86 = None
        permute_53: "bf16[3, 512, 4, 16, 60][240, 11520, 60, 720, 1]cuda:0" = torch.ops.aten.permute.default(view_87, [2, 0, 3, 1, 4]);  view_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_7 = torch.ops.aten.unbind.int(permute_53);  permute_53 = None
        getitem_176: "bf16[512, 4, 16, 60][11520, 60, 720, 1]cuda:0" = unbind_7[0]
        getitem_177: "bf16[512, 4, 16, 60][11520, 60, 720, 1]cuda:0" = unbind_7[1]
        getitem_178: "bf16[512, 4, 16, 60][11520, 60, 720, 1]cuda:0" = unbind_7[2];  unbind_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        constant_pad_nd_9: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_176, [0, 4], 0.0);  getitem_176 = None
        constant_pad_nd_10: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_177, [0, 4], 0.0);  getitem_177 = None
        constant_pad_nd_11: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_178, [0, 4], 0.0);  getitem_178 = None
        _scaled_dot_product_flash_attention_3 = torch.ops.aten._scaled_dot_product_flash_attention.default(constant_pad_nd_9, constant_pad_nd_10, constant_pad_nd_11, scale = 0.12909944487358055)
        getitem_179: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = _scaled_dot_product_flash_attention_3[0]
        getitem_180: "f32[512, 4, 16][64, 16, 1]cuda:0" = _scaled_dot_product_flash_attention_3[1]
        getitem_185: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_3[6]
        getitem_186: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_3[7];  _scaled_dot_product_flash_attention_3 = None
        slice_4: "bf16[512, 4, 16, 60][4096, 1024, 64, 1]cuda:0" = torch.ops.aten.slice.Tensor(getitem_179, -1, 0, 60)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_54: "bf16[512, 16, 4, 60][4096, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(slice_4, [0, 2, 1, 3]);  slice_4 = None
        clone_59: "bf16[512, 16, 4, 60][3840, 240, 60, 1]cuda:0" = torch.ops.aten.clone.default(permute_54, memory_format = torch.contiguous_format);  permute_54 = None
        view_88: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(clone_59, [512, 16, 240]);  clone_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        convert_element_type_328: "bf16[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_272, torch.bfloat16);  primals_272 = None
        convert_element_type_329: "bf16[240, 240][240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_271, torch.bfloat16);  primals_271 = None
        view_89: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.reshape.default(view_88, [8192, 240]);  view_88 = None
        permute_55: "bf16[240, 240][1, 240]cuda:0" = torch.ops.aten.permute.default(convert_element_type_329, [1, 0]);  convert_element_type_329 = None
        addmm_29: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_328, view_89, permute_55);  convert_element_type_328 = None
        view_90: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_29, [512, 16, 240]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_224: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(add_221, view_90);  view_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_333: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_224, torch.float32)
        var_mean_46 = torch.ops.aten.var_mean.correction(convert_element_type_333, [2], correction = 0, keepdim = True)
        getitem_188: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_46[0]
        getitem_189: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_46[1];  var_mean_46 = None
        add_225: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_188, 1e-05);  getitem_188 = None
        rsqrt_46: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_225);  add_225 = None
        sub_46: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_333, getitem_189);  convert_element_type_333 = None
        mul_237: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
        mul_238: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_237, primals_273);  mul_237 = None
        add_226: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_238, primals_274);  mul_238 = primals_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_334: "bf16[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_276, torch.bfloat16);  primals_276 = None
        convert_element_type_335: "bf16[480, 240][240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_275, torch.bfloat16);  primals_275 = None
        convert_element_type_336: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_226, torch.bfloat16);  add_226 = None
        view_91: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_336, [8192, 240]);  convert_element_type_336 = None
        permute_56: "bf16[240, 480][1, 240]cuda:0" = torch.ops.aten.permute.default(convert_element_type_335, [1, 0]);  convert_element_type_335 = None
        addmm_30: "bf16[8192, 480][480, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_334, view_91, permute_56);  convert_element_type_334 = None
        view_92: "bf16[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [512, 16, 480])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_340: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_92, torch.float32);  view_92 = None
        neg_29: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_340)
        exp_29: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.exp.default(neg_29);  neg_29 = None
        add_227: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_29, 1);  exp_29 = None
        div_29: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_340, add_227);  convert_element_type_340 = add_227 = None
        convert_element_type_341: "bf16[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_29, torch.bfloat16);  div_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_342: "bf16[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_278, torch.bfloat16);  primals_278 = None
        convert_element_type_343: "bf16[240, 480][480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_277, torch.bfloat16);  primals_277 = None
        view_93: "bf16[8192, 480][480, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_341, [8192, 480]);  convert_element_type_341 = None
        permute_57: "bf16[480, 240][1, 480]cuda:0" = torch.ops.aten.permute.default(convert_element_type_343, [1, 0]);  convert_element_type_343 = None
        addmm_31: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_342, view_93, permute_57);  convert_element_type_342 = None
        view_94: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [512, 16, 240]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_228: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(add_224, view_94);  view_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        convert_element_type_347: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_228, torch.float32)
        var_mean_47 = torch.ops.aten.var_mean.correction(convert_element_type_347, [2], correction = 0, keepdim = True)
        getitem_190: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_47[0]
        getitem_191: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_47[1];  var_mean_47 = None
        add_229: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_190, 1e-05);  getitem_190 = None
        rsqrt_47: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
        sub_47: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_347, getitem_191);  convert_element_type_347 = None
        mul_239: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
        mul_240: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_239, primals_279);  mul_239 = None
        add_230: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_240, primals_280);  mul_240 = primals_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        convert_element_type_348: "bf16[720][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_282, torch.bfloat16);  primals_282 = None
        convert_element_type_349: "bf16[720, 240][240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_281, torch.bfloat16);  primals_281 = None
        convert_element_type_350: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_230, torch.bfloat16);  add_230 = None
        view_95: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_350, [8192, 240]);  convert_element_type_350 = None
        permute_58: "bf16[240, 720][1, 240]cuda:0" = torch.ops.aten.permute.default(convert_element_type_349, [1, 0]);  convert_element_type_349 = None
        addmm_32: "bf16[8192, 720][720, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_348, view_95, permute_58);  convert_element_type_348 = None
        view_96: "bf16[512, 16, 720][11520, 720, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_32, [512, 16, 720]);  addmm_32 = None
        view_97: "bf16[512, 16, 3, 4, 60][11520, 720, 240, 60, 1]cuda:0" = torch.ops.aten.reshape.default(view_96, [512, 16, 3, 4, 60]);  view_96 = None
        permute_59: "bf16[3, 512, 4, 16, 60][240, 11520, 60, 720, 1]cuda:0" = torch.ops.aten.permute.default(view_97, [2, 0, 3, 1, 4]);  view_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_8 = torch.ops.aten.unbind.int(permute_59);  permute_59 = None
        getitem_192: "bf16[512, 4, 16, 60][11520, 60, 720, 1]cuda:0" = unbind_8[0]
        getitem_193: "bf16[512, 4, 16, 60][11520, 60, 720, 1]cuda:0" = unbind_8[1]
        getitem_194: "bf16[512, 4, 16, 60][11520, 60, 720, 1]cuda:0" = unbind_8[2];  unbind_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        constant_pad_nd_12: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_192, [0, 4], 0.0);  getitem_192 = None
        constant_pad_nd_13: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_193, [0, 4], 0.0);  getitem_193 = None
        constant_pad_nd_14: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_194, [0, 4], 0.0);  getitem_194 = None
        _scaled_dot_product_flash_attention_4 = torch.ops.aten._scaled_dot_product_flash_attention.default(constant_pad_nd_12, constant_pad_nd_13, constant_pad_nd_14, scale = 0.12909944487358055)
        getitem_195: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = _scaled_dot_product_flash_attention_4[0]
        getitem_196: "f32[512, 4, 16][64, 16, 1]cuda:0" = _scaled_dot_product_flash_attention_4[1]
        getitem_201: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_4[6]
        getitem_202: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_4[7];  _scaled_dot_product_flash_attention_4 = None
        slice_5: "bf16[512, 4, 16, 60][4096, 1024, 64, 1]cuda:0" = torch.ops.aten.slice.Tensor(getitem_195, -1, 0, 60)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_60: "bf16[512, 16, 4, 60][4096, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(slice_5, [0, 2, 1, 3]);  slice_5 = None
        clone_63: "bf16[512, 16, 4, 60][3840, 240, 60, 1]cuda:0" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
        view_98: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(clone_63, [512, 16, 240]);  clone_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        convert_element_type_354: "bf16[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_284, torch.bfloat16);  primals_284 = None
        convert_element_type_355: "bf16[240, 240][240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_283, torch.bfloat16);  primals_283 = None
        view_99: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.reshape.default(view_98, [8192, 240]);  view_98 = None
        permute_61: "bf16[240, 240][1, 240]cuda:0" = torch.ops.aten.permute.default(convert_element_type_355, [1, 0]);  convert_element_type_355 = None
        addmm_33: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_354, view_99, permute_61);  convert_element_type_354 = None
        view_100: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [512, 16, 240]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_231: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(add_228, view_100);  view_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_359: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_231, torch.float32)
        var_mean_48 = torch.ops.aten.var_mean.correction(convert_element_type_359, [2], correction = 0, keepdim = True)
        getitem_204: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_48[0]
        getitem_205: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_48[1];  var_mean_48 = None
        add_232: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_204, 1e-05);  getitem_204 = None
        rsqrt_48: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_232);  add_232 = None
        sub_48: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_359, getitem_205);  convert_element_type_359 = None
        mul_241: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
        mul_242: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_241, primals_285);  mul_241 = None
        add_233: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_242, primals_286);  mul_242 = primals_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_360: "bf16[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_288, torch.bfloat16);  primals_288 = None
        convert_element_type_361: "bf16[480, 240][240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_287, torch.bfloat16);  primals_287 = None
        convert_element_type_362: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_233, torch.bfloat16);  add_233 = None
        view_101: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_362, [8192, 240]);  convert_element_type_362 = None
        permute_62: "bf16[240, 480][1, 240]cuda:0" = torch.ops.aten.permute.default(convert_element_type_361, [1, 0]);  convert_element_type_361 = None
        addmm_34: "bf16[8192, 480][480, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_360, view_101, permute_62);  convert_element_type_360 = None
        view_102: "bf16[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [512, 16, 480])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_366: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_102, torch.float32);  view_102 = None
        neg_30: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_366)
        exp_30: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.exp.default(neg_30);  neg_30 = None
        add_234: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_30, 1);  exp_30 = None
        div_30: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_366, add_234);  convert_element_type_366 = add_234 = None
        convert_element_type_367: "bf16[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_30, torch.bfloat16);  div_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_368: "bf16[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_290, torch.bfloat16);  primals_290 = None
        convert_element_type_369: "bf16[240, 480][480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_289, torch.bfloat16);  primals_289 = None
        view_103: "bf16[8192, 480][480, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_367, [8192, 480]);  convert_element_type_367 = None
        permute_63: "bf16[480, 240][1, 480]cuda:0" = torch.ops.aten.permute.default(convert_element_type_369, [1, 0]);  convert_element_type_369 = None
        addmm_35: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_368, view_103, permute_63);  convert_element_type_368 = None
        view_104: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_35, [512, 16, 240]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_235: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(add_231, view_104);  view_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:264 in forward, code: x = self.norm(x)
        convert_element_type_373: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_235, torch.float32)
        var_mean_49 = torch.ops.aten.var_mean.correction(convert_element_type_373, [2], correction = 0, keepdim = True)
        getitem_206: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_49[0]
        getitem_207: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_49[1];  var_mean_49 = None
        add_236: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_206, 1e-05);  getitem_206 = None
        rsqrt_49: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_236);  add_236 = None
        sub_49: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_373, getitem_207);  convert_element_type_373 = None
        mul_243: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
        mul_244: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_243, primals_291);  mul_243 = None
        add_237: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_244, primals_292);  mul_244 = primals_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:268 in forward, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
        view_105: "f32[128, 4, 16, 240][15360, 3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(add_237, [128, 4, 16, -1]);  add_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:269 in forward, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
        permute_64: "f32[128, 240, 16, 4][15360, 1, 240, 3840]cuda:0" = torch.ops.aten.permute.default(view_105, [0, 3, 2, 1]);  view_105 = None
        clone_67: "f32[128, 240, 16, 4][15360, 64, 4, 1]cuda:0" = torch.ops.aten.clone.default(permute_64, memory_format = torch.contiguous_format);  permute_64 = None
        view_106: "f32[122880, 4, 2, 2][16, 4, 2, 1]cuda:0" = torch.ops.aten.reshape.default(clone_67, [122880, 4, 2, 2]);  clone_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:271 in forward, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
        permute_65: "f32[122880, 2, 4, 2][16, 2, 4, 1]cuda:0" = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None
        clone_68: "f32[122880, 2, 4, 2][16, 8, 2, 1]cuda:0" = torch.ops.aten.clone.default(permute_65, memory_format = torch.contiguous_format);  permute_65 = None
        view_107: "f32[128, 240, 8, 8][15360, 64, 8, 1]cuda:0" = torch.ops.aten.reshape.default(clone_68, [128, 240, 8, 8]);  clone_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_374: "bf16[160, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.prims.convert_element_type.default(primals_293, torch.bfloat16);  primals_293 = None
        convert_element_type_375: "bf16[128, 240, 8, 8][15360, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_107, torch.bfloat16);  view_107 = None
        convolution_32: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_375, convert_element_type_374, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_238: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_294, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_376: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_32, torch.float32)
        var_mean_50 = torch.ops.aten.var_mean.correction(convert_element_type_376, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_376 = None
        getitem_208: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_50[0]
        getitem_209: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_50[1];  var_mean_50 = None
        add_239: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_208, 1e-05)
        rsqrt_50: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_239);  add_239 = None
        sub_50: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(convolution_32, getitem_209)
        mul_245: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
        squeeze_87: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_209, [0, 2, 3])
        mul_246: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1);  squeeze_87 = None
        mul_247: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_295, 0.9)
        add_240: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
        squeeze_89: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_208, [0, 2, 3]);  getitem_208 = None
        mul_248: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0001220852154804);  squeeze_89 = None
        mul_249: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_248, 0.1);  mul_248 = None
        mul_250: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_296, 0.9)
        add_241: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
        unsqueeze_116: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_297, -1)
        unsqueeze_117: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        mul_251: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_117);  mul_245 = unsqueeze_117 = None
        unsqueeze_118: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_298, -1)
        unsqueeze_119: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
        add_242: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.add.Tensor(mul_251, unsqueeze_119);  mul_251 = unsqueeze_119 = None
        convert_element_type_377: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(add_242, torch.bfloat16);  add_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_378: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_377, torch.float32);  convert_element_type_377 = None
        neg_31: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.neg.default(convert_element_type_378)
        exp_31: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.exp.default(neg_31);  neg_31 = None
        add_243: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.add.Tensor(exp_31, 1);  exp_31 = None
        div_31: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_378, add_243);  convert_element_type_378 = add_243 = None
        convert_element_type_379: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(div_31, torch.bfloat16);  div_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:277 in forward, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
        cat_2: "bf16[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.cat.default([convert_element_type_288, convert_element_type_379], 1);  convert_element_type_379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_380: "bf16[160, 320, 3, 3][2880, 1, 960, 320]cuda:0" = torch.ops.prims.convert_element_type.default(primals_299, torch.bfloat16);  primals_299 = None
        convolution_33: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.convolution.default(cat_2, convert_element_type_380, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_244: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_300, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_381: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_33, torch.float32)
        var_mean_51 = torch.ops.aten.var_mean.correction(convert_element_type_381, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_381 = None
        getitem_210: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_51[0]
        getitem_211: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_51[1];  var_mean_51 = None
        add_245: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_210, 1e-05)
        rsqrt_51: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_245);  add_245 = None
        sub_51: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(convolution_33, getitem_211)
        mul_252: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
        squeeze_90: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_211, [0, 2, 3])
        mul_253: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1);  squeeze_90 = None
        mul_254: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_301, 0.9)
        add_246: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_253, mul_254);  mul_253 = mul_254 = None
        squeeze_92: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_210, [0, 2, 3]);  getitem_210 = None
        mul_255: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0001220852154804);  squeeze_92 = None
        mul_256: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_255, 0.1);  mul_255 = None
        mul_257: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_302, 0.9)
        add_247: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
        unsqueeze_120: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_303, -1)
        unsqueeze_121: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        mul_258: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_252, unsqueeze_121);  mul_252 = unsqueeze_121 = None
        unsqueeze_122: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_304, -1)
        unsqueeze_123: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        add_248: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.add.Tensor(mul_258, unsqueeze_123);  mul_258 = unsqueeze_123 = None
        convert_element_type_382: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(add_248, torch.bfloat16);  add_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_383: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_382, torch.float32);  convert_element_type_382 = None
        neg_32: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.neg.default(convert_element_type_383)
        exp_32: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.exp.default(neg_32);  neg_32 = None
        add_249: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.add.Tensor(exp_32, 1);  exp_32 = None
        div_32: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_383, add_249);  convert_element_type_383 = add_249 = None
        convert_element_type_384: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(div_32, torch.bfloat16);  div_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_385: "bf16[640, 160, 1, 1][160, 1, 160, 160]cuda:0" = torch.ops.prims.convert_element_type.default(primals_305, torch.bfloat16);  primals_305 = None
        convolution_34: "bf16[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_384, convert_element_type_385, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_250: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_306, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_386: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_34, torch.float32)
        var_mean_52 = torch.ops.aten.var_mean.correction(convert_element_type_386, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_386 = None
        getitem_212: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = var_mean_52[0]
        getitem_213: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = var_mean_52[1];  var_mean_52 = None
        add_251: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_212, 1e-05)
        rsqrt_52: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_251);  add_251 = None
        sub_52: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.sub.Tensor(convolution_34, getitem_213)
        mul_259: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
        squeeze_93: "f32[640][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_213, [0, 2, 3])
        mul_260: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1);  squeeze_93 = None
        mul_261: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_307, 0.9)
        add_252: "f32[640][1]cuda:0" = torch.ops.aten.add.Tensor(mul_260, mul_261);  mul_260 = mul_261 = None
        squeeze_95: "f32[640][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_212, [0, 2, 3]);  getitem_212 = None
        mul_262: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0001220852154804);  squeeze_95 = None
        mul_263: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_262, 0.1);  mul_262 = None
        mul_264: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_308, 0.9)
        add_253: "f32[640][1]cuda:0" = torch.ops.aten.add.Tensor(mul_263, mul_264);  mul_263 = mul_264 = None
        unsqueeze_124: "f32[640, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_309, -1)
        unsqueeze_125: "f32[640, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        mul_265: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_125);  mul_259 = unsqueeze_125 = None
        unsqueeze_126: "f32[640, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_310, -1)
        unsqueeze_127: "f32[640, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        add_254: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.add.Tensor(mul_265, unsqueeze_127);  mul_265 = unsqueeze_127 = None
        convert_element_type_387: "bf16[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.prims.convert_element_type.default(add_254, torch.bfloat16);  add_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_388: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_387, torch.float32);  convert_element_type_387 = None
        neg_33: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.neg.default(convert_element_type_388)
        exp_33: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.exp.default(neg_33);  neg_33 = None
        add_255: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.add.Tensor(exp_33, 1);  exp_33 = None
        div_33: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_388, add_255);  convert_element_type_388 = add_255 = None
        convert_element_type_389: "bf16[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.prims.convert_element_type.default(div_33, torch.bfloat16);  div_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean: "bf16[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_389, [-1, -2], True);  convert_element_type_389 = None
        as_strided: "bf16[128, 640, 1, 1][640, 1, 640, 640]cuda:0" = torch.ops.aten.as_strided.default(mean, [128, 640, 1, 1], [640, 1, 640, 640]);  mean = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_108: "bf16[128, 640][640, 1]cuda:0" = torch.ops.aten.reshape.default(as_strided, [128, 640]);  as_strided = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:141 in forward, code: x = self.fc(x)
        convert_element_type_390: "bf16[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_312, torch.bfloat16);  primals_312 = None
        convert_element_type_391: "bf16[1000, 640][640, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_311, torch.bfloat16);  primals_311 = None
        permute_66: "bf16[640, 1000][1, 640]cuda:0" = torch.ops.aten.permute.default(convert_element_type_391, [1, 0]);  convert_element_type_391 = None
        addmm_36: "bf16[128, 1000][1000, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_390, view_108, permute_66);  convert_element_type_390 = None
        permute_67: "bf16[1000, 640][640, 1]cuda:0" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_73: "bf16[240, 480][480, 1]cuda:0" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_77: "bf16[480, 240][240, 1]cuda:0" = torch.ops.aten.permute.default(permute_62, [1, 0]);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_81: "bf16[240, 240][240, 1]cuda:0" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_87: "bf16[720, 240][240, 1]cuda:0" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_91: "bf16[240, 480][480, 1]cuda:0" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_95: "bf16[480, 240][240, 1]cuda:0" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_99: "bf16[240, 240][240, 1]cuda:0" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_105: "bf16[720, 240][240, 1]cuda:0" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_109: "bf16[240, 480][480, 1]cuda:0" = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_113: "bf16[480, 240][240, 1]cuda:0" = torch.ops.aten.permute.default(permute_50, [1, 0]);  permute_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_117: "bf16[240, 240][240, 1]cuda:0" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_123: "bf16[720, 240][240, 1]cuda:0" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_176: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
        unsqueeze_177: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_176, 2);  unsqueeze_176 = None
        unsqueeze_178: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_177, 3);  unsqueeze_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_131: "bf16[192, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_135: "bf16[384, 192][192, 1]cuda:0" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_139: "bf16[192, 192][192, 1]cuda:0" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_145: "bf16[576, 192][192, 1]cuda:0" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_149: "bf16[192, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_153: "bf16[384, 192][192, 1]cuda:0" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_157: "bf16[192, 192][192, 1]cuda:0" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_163: "bf16[576, 192][192, 1]cuda:0" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_167: "bf16[192, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_171: "bf16[384, 192][192, 1]cuda:0" = torch.ops.aten.permute.default(permute_28, [1, 0]);  permute_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_175: "bf16[192, 192][192, 1]cuda:0" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_181: "bf16[576, 192][192, 1]cuda:0" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_185: "bf16[192, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_189: "bf16[384, 192][192, 1]cuda:0" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_193: "bf16[192, 192][192, 1]cuda:0" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_199: "bf16[576, 192][192, 1]cuda:0" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_248: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
        unsqueeze_249: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 2);  unsqueeze_248 = None
        unsqueeze_250: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_249, 3);  unsqueeze_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_207: "bf16[144, 288][288, 1]cuda:0" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_211: "bf16[288, 144][144, 1]cuda:0" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_215: "bf16[144, 144][144, 1]cuda:0" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_221: "bf16[432, 144][144, 1]cuda:0" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_225: "bf16[144, 288][288, 1]cuda:0" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_229: "bf16[288, 144][144, 1]cuda:0" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        permute_233: "bf16[144, 144][144, 1]cuda:0" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_239: "bf16[432, 144][144, 1]cuda:0" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_320: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
        unsqueeze_321: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 2);  unsqueeze_320 = None
        unsqueeze_322: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 3);  unsqueeze_321 = None
        unsqueeze_356: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
        unsqueeze_357: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 2);  unsqueeze_356 = None
        unsqueeze_358: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 3);  unsqueeze_357 = None
        unsqueeze_392: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
        unsqueeze_393: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 2);  unsqueeze_392 = None
        unsqueeze_394: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 3);  unsqueeze_393 = None
        unsqueeze_428: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
        unsqueeze_429: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 2);  unsqueeze_428 = None
        unsqueeze_430: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 3);  unsqueeze_429 = None
        unsqueeze_464: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
        unsqueeze_465: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 2);  unsqueeze_464 = None
        unsqueeze_466: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 3);  unsqueeze_465 = None

        # No stacktrace found for following nodes
        copy_: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_3, add);  primals_3 = add = copy_ = None
        copy__1: "f32[16][1]cuda:0" = torch.ops.aten.copy_.default(primals_4, add_2);  primals_4 = add_2 = copy__1 = None
        copy__2: "f32[16][1]cuda:0" = torch.ops.aten.copy_.default(primals_5, add_3);  primals_5 = add_3 = copy__2 = None
        copy__3: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_9, add_6);  primals_9 = add_6 = copy__3 = None
        copy__4: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_10, add_8);  primals_10 = add_8 = copy__4 = None
        copy__5: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_11, add_9);  primals_11 = add_9 = copy__5 = None
        copy__6: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_15, add_12);  primals_15 = add_12 = copy__6 = None
        copy__7: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_16, add_14);  primals_16 = add_14 = copy__7 = None
        copy__8: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_17, add_15);  primals_17 = add_15 = copy__8 = None
        copy__9: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_21, add_18);  primals_21 = add_18 = copy__9 = None
        copy__10: "f32[32][1]cuda:0" = torch.ops.aten.copy_.default(primals_22, add_20);  primals_22 = add_20 = copy__10 = None
        copy__11: "f32[32][1]cuda:0" = torch.ops.aten.copy_.default(primals_23, add_21);  primals_23 = add_21 = copy__11 = None
        copy__12: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_27, add_23);  primals_27 = add_23 = copy__12 = None
        copy__13: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_28, add_25);  primals_28 = add_25 = copy__13 = None
        copy__14: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_29, add_26);  primals_29 = add_26 = copy__14 = None
        copy__15: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_33, add_29);  primals_33 = add_29 = copy__15 = None
        copy__16: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_34, add_31);  primals_34 = add_31 = copy__16 = None
        copy__17: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_35, add_32);  primals_35 = add_32 = copy__17 = None
        copy__18: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_39, add_35);  primals_39 = add_35 = copy__18 = None
        copy__19: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_40, add_37);  primals_40 = add_37 = copy__19 = None
        copy__20: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_41, add_38);  primals_41 = add_38 = copy__20 = None
        copy__21: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_45, add_40);  primals_45 = add_40 = copy__21 = None
        copy__22: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_46, add_42);  primals_46 = add_42 = copy__22 = None
        copy__23: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_47, add_43);  primals_47 = add_43 = copy__23 = None
        copy__24: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_51, add_46);  primals_51 = add_46 = copy__24 = None
        copy__25: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_52, add_48);  primals_52 = add_48 = copy__25 = None
        copy__26: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_53, add_49);  primals_53 = add_49 = copy__26 = None
        copy__27: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_57, add_52);  primals_57 = add_52 = copy__27 = None
        copy__28: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_58, add_54);  primals_58 = add_54 = copy__28 = None
        copy__29: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_59, add_55);  primals_59 = add_55 = copy__29 = None
        copy__30: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_63, add_58);  primals_63 = add_58 = copy__30 = None
        copy__31: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_64, add_60);  primals_64 = add_60 = copy__31 = None
        copy__32: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_65, add_61);  primals_65 = add_61 = copy__32 = None
        copy__33: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_69, add_64);  primals_69 = add_64 = copy__33 = None
        copy__34: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_70, add_66);  primals_70 = add_66 = copy__34 = None
        copy__35: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_71, add_67);  primals_71 = add_67 = copy__35 = None
        copy__36: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_75, add_70);  primals_75 = add_70 = copy__36 = None
        copy__37: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_76, add_72);  primals_76 = add_72 = copy__37 = None
        copy__38: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_77, add_73);  primals_77 = add_73 = copy__38 = None
        copy__39: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_81, add_76);  primals_81 = add_76 = copy__39 = None
        copy__40: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_82, add_78);  primals_82 = add_78 = copy__40 = None
        copy__41: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_83, add_79);  primals_83 = add_79 = copy__41 = None
        copy__42: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_87, add_82);  primals_87 = add_82 = copy__42 = None
        copy__43: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_88, add_84);  primals_88 = add_84 = copy__43 = None
        copy__44: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_89, add_85);  primals_89 = add_85 = copy__44 = None
        copy__45: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_93, add_88);  primals_93 = add_88 = copy__45 = None
        copy__46: "f32[96][1]cuda:0" = torch.ops.aten.copy_.default(primals_94, add_90);  primals_94 = add_90 = copy__46 = None
        copy__47: "f32[96][1]cuda:0" = torch.ops.aten.copy_.default(primals_95, add_91);  primals_95 = add_91 = copy__47 = None
        copy__48: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_99, add_93);  primals_99 = add_93 = copy__48 = None
        copy__49: "f32[96][1]cuda:0" = torch.ops.aten.copy_.default(primals_100, add_95);  primals_100 = add_95 = copy__49 = None
        copy__50: "f32[96][1]cuda:0" = torch.ops.aten.copy_.default(primals_101, add_96);  primals_101 = add_96 = copy__50 = None
        copy__51: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_132, add_115);  primals_132 = add_115 = copy__51 = None
        copy__52: "f32[96][1]cuda:0" = torch.ops.aten.copy_.default(primals_133, add_117);  primals_133 = add_117 = copy__52 = None
        copy__53: "f32[96][1]cuda:0" = torch.ops.aten.copy_.default(primals_134, add_118);  primals_134 = add_118 = copy__53 = None
        copy__54: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_138, add_121);  primals_138 = add_121 = copy__54 = None
        copy__55: "f32[96][1]cuda:0" = torch.ops.aten.copy_.default(primals_139, add_123);  primals_139 = add_123 = copy__55 = None
        copy__56: "f32[96][1]cuda:0" = torch.ops.aten.copy_.default(primals_140, add_124);  primals_140 = add_124 = copy__56 = None
        copy__57: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_144, add_127);  primals_144 = add_127 = copy__57 = None
        copy__58: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_145, add_129);  primals_145 = add_129 = copy__58 = None
        copy__59: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_146, add_130);  primals_146 = add_130 = copy__59 = None
        copy__60: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_150, add_133);  primals_150 = add_133 = copy__60 = None
        copy__61: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_151, add_135);  primals_151 = add_135 = copy__61 = None
        copy__62: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_152, add_136);  primals_152 = add_136 = copy__62 = None
        copy__63: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_156, add_139);  primals_156 = add_139 = copy__63 = None
        copy__64: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_157, add_141);  primals_157 = add_141 = copy__64 = None
        copy__65: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_158, add_142);  primals_158 = add_142 = copy__65 = None
        copy__66: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_162, add_144);  primals_162 = add_144 = copy__66 = None
        copy__67: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_163, add_146);  primals_163 = add_146 = copy__67 = None
        copy__68: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_164, add_147);  primals_164 = add_147 = copy__68 = None
        copy__69: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_219, add_180);  primals_219 = add_180 = copy__69 = None
        copy__70: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_220, add_182);  primals_220 = add_182 = copy__70 = None
        copy__71: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_221, add_183);  primals_221 = add_183 = copy__71 = None
        copy__72: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_225, add_186);  primals_225 = add_186 = copy__72 = None
        copy__73: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_226, add_188);  primals_226 = add_188 = copy__73 = None
        copy__74: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_227, add_189);  primals_227 = add_189 = copy__74 = None
        copy__75: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_231, add_192);  primals_231 = add_192 = copy__75 = None
        copy__76: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_232, add_194);  primals_232 = add_194 = copy__76 = None
        copy__77: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_233, add_195);  primals_233 = add_195 = copy__77 = None
        copy__78: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_237, add_198);  primals_237 = add_198 = copy__78 = None
        copy__79: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_238, add_200);  primals_238 = add_200 = copy__79 = None
        copy__80: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_239, add_201);  primals_239 = add_201 = copy__80 = None
        copy__81: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_243, add_204);  primals_243 = add_204 = copy__81 = None
        copy__82: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_244, add_206);  primals_244 = add_206 = copy__82 = None
        copy__83: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_245, add_207);  primals_245 = add_207 = copy__83 = None
        copy__84: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_249, add_209);  primals_249 = add_209 = copy__84 = None
        copy__85: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_250, add_211);  primals_250 = add_211 = copy__85 = None
        copy__86: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_251, add_212);  primals_251 = add_212 = copy__86 = None
        copy__87: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_294, add_238);  primals_294 = add_238 = copy__87 = None
        copy__88: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_295, add_240);  primals_295 = add_240 = copy__88 = None
        copy__89: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_296, add_241);  primals_296 = add_241 = copy__89 = None
        copy__90: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_300, add_244);  primals_300 = add_244 = copy__90 = None
        copy__91: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_301, add_246);  primals_301 = add_246 = copy__91 = None
        copy__92: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_302, add_247);  primals_302 = add_247 = copy__92 = None
        copy__93: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_306, add_250);  primals_306 = add_250 = copy__93 = None
        copy__94: "f32[640][1]cuda:0" = torch.ops.aten.copy_.default(primals_307, add_252);  primals_307 = add_252 = copy__94 = None
        copy__95: "f32[640][1]cuda:0" = torch.ops.aten.copy_.default(primals_308, add_253);  primals_308 = add_253 = copy__95 = None
        return (addmm_36, primals_6, primals_7, primals_12, primals_13, primals_18, primals_19, primals_24, primals_30, primals_31, primals_36, primals_37, primals_42, primals_48, primals_49, primals_54, primals_55, primals_60, primals_66, primals_67, primals_72, primals_73, primals_78, primals_84, primals_85, primals_90, primals_91, primals_96, primals_102, primals_103, primals_105, primals_111, primals_117, primals_123, primals_129, primals_135, primals_136, primals_141, primals_142, primals_147, primals_148, primals_153, primals_154, primals_159, primals_165, primals_166, primals_168, primals_174, primals_180, primals_186, primals_192, primals_198, primals_204, primals_210, primals_216, primals_222, primals_223, primals_228, primals_229, primals_234, primals_235, primals_240, primals_241, primals_246, primals_252, primals_253, primals_255, primals_261, primals_267, primals_273, primals_279, primals_285, primals_291, primals_297, primals_298, primals_303, primals_304, primals_309, primals_310, convert_element_type, convert_element_type_1, convolution, getitem_1, rsqrt, convert_element_type_5, convert_element_type_6, convolution_1, getitem_3, rsqrt_1, convert_element_type_10, convert_element_type_11, convolution_2, getitem_5, rsqrt_2, convert_element_type_15, convert_element_type_16, convolution_3, squeeze_10, convert_element_type_18, convert_element_type_19, convolution_4, getitem_9, rsqrt_4, convert_element_type_23, convert_element_type_24, convolution_5, getitem_11, rsqrt_5, convert_element_type_28, convert_element_type_29, convolution_6, squeeze_19, convert_element_type_31, convert_element_type_32, convolution_7, getitem_15, rsqrt_7, convert_element_type_36, convert_element_type_37, convolution_8, getitem_17, rsqrt_8, convert_element_type_41, convert_element_type_42, convolution_9, squeeze_28, add_57, convert_element_type_45, convolution_10, getitem_21, rsqrt_10, convert_element_type_49, convert_element_type_50, convolution_11, getitem_23, rsqrt_11, convert_element_type_54, convert_element_type_55, convolution_12, squeeze_37, add_75, convert_element_type_58, convolution_13, getitem_27, rsqrt_13, convert_element_type_62, convert_element_type_63, convolution_14, getitem_29, rsqrt_14, convert_element_type_67, convert_element_type_68, convolution_15, squeeze_46, convert_element_type_70, convert_element_type_71, convolution_16, getitem_33, rsqrt_16, convert_element_type_75, convert_element_type_76, view_2, getitem_35, rsqrt_17, view_3, constant_pad_nd, constant_pad_nd_1, constant_pad_nd_2, getitem_39, getitem_40, getitem_45, getitem_46, view_7, add_101, getitem_49, rsqrt_18, view_9, addmm_2, view_11, add_105, getitem_51, rsqrt_19, view_13, constant_pad_nd_3, constant_pad_nd_4, constant_pad_nd_5, getitem_55, getitem_56, getitem_61, getitem_62, view_17, add_108, getitem_65, rsqrt_20, view_19, addmm_6, view_21, add_112, getitem_67, rsqrt_21, convert_element_type_130, convert_element_type_131, convolution_18, getitem_69, rsqrt_22, cat, convert_element_type_136, convolution_19, getitem_71, rsqrt_23, convert_element_type_140, convert_element_type_141, convolution_20, getitem_73, rsqrt_24, convert_element_type_145, convert_element_type_146, convolution_21, getitem_75, rsqrt_25, convert_element_type_150, convert_element_type_151, convolution_22, squeeze_64, convert_element_type_153, convert_element_type_154, convolution_23, getitem_79, rsqrt_27, convert_element_type_158, convert_element_type_159, view_28, getitem_81, rsqrt_28, view_29, getitem_82, getitem_83, getitem_84, getitem_85, getitem_86, getitem_91, getitem_92, add_152, getitem_95, rsqrt_29, view_35, addmm_10, view_37, add_156, getitem_97, rsqrt_30, view_39, getitem_98, getitem_99, getitem_100, getitem_101, getitem_102, getitem_107, getitem_108, add_159, getitem_111, rsqrt_31, view_45, addmm_14, view_47, add_163, getitem_113, rsqrt_32, view_49, getitem_114, getitem_115, getitem_116, getitem_117, getitem_118, getitem_123, getitem_124, add_166, getitem_127, rsqrt_33, view_55, addmm_18, view_57, add_170, getitem_129, rsqrt_34, view_59, getitem_130, getitem_131, getitem_132, getitem_133, getitem_134, getitem_139, getitem_140, add_173, getitem_143, rsqrt_35, view_65, addmm_22, view_67, add_177, getitem_145, rsqrt_36, convert_element_type_265, convert_element_type_266, convolution_25, getitem_147, rsqrt_37, cat_1, convert_element_type_271, convolution_26, getitem_149, rsqrt_38, convert_element_type_275, convert_element_type_276, convolution_27, getitem_151, rsqrt_39, convert_element_type_280, convert_element_type_281, convolution_28, getitem_153, rsqrt_40, convert_element_type_285, convert_element_type_286, convolution_29, squeeze_82, convert_element_type_288, convert_element_type_289, convolution_30, getitem_157, rsqrt_42, convert_element_type_293, convert_element_type_294, view_74, getitem_159, rsqrt_43, view_75, constant_pad_nd_6, constant_pad_nd_7, constant_pad_nd_8, getitem_163, getitem_164, getitem_169, getitem_170, view_79, add_217, getitem_173, rsqrt_44, view_81, addmm_26, view_83, add_221, getitem_175, rsqrt_45, view_85, constant_pad_nd_9, constant_pad_nd_10, constant_pad_nd_11, getitem_179, getitem_180, getitem_185, getitem_186, view_89, add_224, getitem_189, rsqrt_46, view_91, addmm_30, view_93, add_228, getitem_191, rsqrt_47, view_95, constant_pad_nd_12, constant_pad_nd_13, constant_pad_nd_14, getitem_195, getitem_196, getitem_201, getitem_202, view_99, add_231, getitem_205, rsqrt_48, view_101, addmm_34, view_103, add_235, getitem_207, rsqrt_49, convert_element_type_374, convert_element_type_375, convolution_32, getitem_209, rsqrt_50, cat_2, convert_element_type_380, convolution_33, getitem_211, rsqrt_51, convert_element_type_384, convert_element_type_385, convolution_34, getitem_213, rsqrt_52, view_108, permute_67, permute_73, permute_77, permute_81, permute_87, permute_91, permute_95, permute_99, permute_105, permute_109, permute_113, permute_117, permute_123, unsqueeze_178, permute_131, permute_135, permute_139, permute_145, permute_149, permute_153, permute_157, permute_163, permute_167, permute_171, permute_175, permute_181, permute_185, permute_189, permute_193, permute_199, unsqueeze_250, permute_207, permute_211, permute_215, permute_221, permute_225, permute_229, permute_233, permute_239, unsqueeze_322, unsqueeze_358, unsqueeze_394, unsqueeze_430, unsqueeze_466)
