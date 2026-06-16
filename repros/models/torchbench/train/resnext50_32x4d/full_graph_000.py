class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[64, 3, 7, 7][147, 49, 7, 1]cuda:0", primals_2: "f32[8, 3, 224, 224][150528, 50176, 224, 1]cuda:0", primals_3: "i64[][]cuda:0", primals_4: "f32[64][1]cuda:0", primals_5: "f32[64][1]cuda:0", primals_6: "f32[64][1]cuda:0", primals_7: "f32[64][1]cuda:0", primals_8: "f32[128, 64, 1, 1][64, 1, 1, 1]cuda:0", primals_9: "i64[][]cuda:0", primals_10: "f32[128][1]cuda:0", primals_11: "f32[128][1]cuda:0", primals_12: "f32[128][1]cuda:0", primals_13: "f32[128][1]cuda:0", primals_14: "f32[128, 4, 3, 3][36, 9, 3, 1]cuda:0", primals_15: "i64[][]cuda:0", primals_16: "f32[128][1]cuda:0", primals_17: "f32[128][1]cuda:0", primals_18: "f32[128][1]cuda:0", primals_19: "f32[128][1]cuda:0", primals_20: "f32[256, 128, 1, 1][128, 1, 1, 1]cuda:0", primals_21: "i64[][]cuda:0", primals_22: "f32[256][1]cuda:0", primals_23: "f32[256][1]cuda:0", primals_24: "f32[256][1]cuda:0", primals_25: "f32[256][1]cuda:0", primals_26: "f32[256, 64, 1, 1][64, 1, 1, 1]cuda:0", primals_27: "i64[][]cuda:0", primals_28: "f32[256][1]cuda:0", primals_29: "f32[256][1]cuda:0", primals_30: "f32[256][1]cuda:0", primals_31: "f32[256][1]cuda:0", primals_32: "f32[128, 256, 1, 1][256, 1, 1, 1]cuda:0", primals_33: "i64[][]cuda:0", primals_34: "f32[128][1]cuda:0", primals_35: "f32[128][1]cuda:0", primals_36: "f32[128][1]cuda:0", primals_37: "f32[128][1]cuda:0", primals_38: "f32[128, 4, 3, 3][36, 9, 3, 1]cuda:0", primals_39: "i64[][]cuda:0", primals_40: "f32[128][1]cuda:0", primals_41: "f32[128][1]cuda:0", primals_42: "f32[128][1]cuda:0", primals_43: "f32[128][1]cuda:0", primals_44: "f32[256, 128, 1, 1][128, 1, 1, 1]cuda:0", primals_45: "i64[][]cuda:0", primals_46: "f32[256][1]cuda:0", primals_47: "f32[256][1]cuda:0", primals_48: "f32[256][1]cuda:0", primals_49: "f32[256][1]cuda:0", primals_50: "f32[128, 256, 1, 1][256, 1, 1, 1]cuda:0", primals_51: "i64[][]cuda:0", primals_52: "f32[128][1]cuda:0", primals_53: "f32[128][1]cuda:0", primals_54: "f32[128][1]cuda:0", primals_55: "f32[128][1]cuda:0", primals_56: "f32[128, 4, 3, 3][36, 9, 3, 1]cuda:0", primals_57: "i64[][]cuda:0", primals_58: "f32[128][1]cuda:0", primals_59: "f32[128][1]cuda:0", primals_60: "f32[128][1]cuda:0", primals_61: "f32[128][1]cuda:0", primals_62: "f32[256, 128, 1, 1][128, 1, 1, 1]cuda:0", primals_63: "i64[][]cuda:0", primals_64: "f32[256][1]cuda:0", primals_65: "f32[256][1]cuda:0", primals_66: "f32[256][1]cuda:0", primals_67: "f32[256][1]cuda:0", primals_68: "f32[256, 256, 1, 1][256, 1, 1, 1]cuda:0", primals_69: "i64[][]cuda:0", primals_70: "f32[256][1]cuda:0", primals_71: "f32[256][1]cuda:0", primals_72: "f32[256][1]cuda:0", primals_73: "f32[256][1]cuda:0", primals_74: "f32[256, 8, 3, 3][72, 9, 3, 1]cuda:0", primals_75: "i64[][]cuda:0", primals_76: "f32[256][1]cuda:0", primals_77: "f32[256][1]cuda:0", primals_78: "f32[256][1]cuda:0", primals_79: "f32[256][1]cuda:0", primals_80: "f32[512, 256, 1, 1][256, 1, 1, 1]cuda:0", primals_81: "i64[][]cuda:0", primals_82: "f32[512][1]cuda:0", primals_83: "f32[512][1]cuda:0", primals_84: "f32[512][1]cuda:0", primals_85: "f32[512][1]cuda:0", primals_86: "f32[512, 256, 1, 1][256, 1, 1, 1]cuda:0", primals_87: "i64[][]cuda:0", primals_88: "f32[512][1]cuda:0", primals_89: "f32[512][1]cuda:0", primals_90: "f32[512][1]cuda:0", primals_91: "f32[512][1]cuda:0", primals_92: "f32[256, 512, 1, 1][512, 1, 1, 1]cuda:0", primals_93: "i64[][]cuda:0", primals_94: "f32[256][1]cuda:0", primals_95: "f32[256][1]cuda:0", primals_96: "f32[256][1]cuda:0", primals_97: "f32[256][1]cuda:0", primals_98: "f32[256, 8, 3, 3][72, 9, 3, 1]cuda:0", primals_99: "i64[][]cuda:0", primals_100: "f32[256][1]cuda:0", primals_101: "f32[256][1]cuda:0", primals_102: "f32[256][1]cuda:0", primals_103: "f32[256][1]cuda:0", primals_104: "f32[512, 256, 1, 1][256, 1, 1, 1]cuda:0", primals_105: "i64[][]cuda:0", primals_106: "f32[512][1]cuda:0", primals_107: "f32[512][1]cuda:0", primals_108: "f32[512][1]cuda:0", primals_109: "f32[512][1]cuda:0", primals_110: "f32[256, 512, 1, 1][512, 1, 1, 1]cuda:0", primals_111: "i64[][]cuda:0", primals_112: "f32[256][1]cuda:0", primals_113: "f32[256][1]cuda:0", primals_114: "f32[256][1]cuda:0", primals_115: "f32[256][1]cuda:0", primals_116: "f32[256, 8, 3, 3][72, 9, 3, 1]cuda:0", primals_117: "i64[][]cuda:0", primals_118: "f32[256][1]cuda:0", primals_119: "f32[256][1]cuda:0", primals_120: "f32[256][1]cuda:0", primals_121: "f32[256][1]cuda:0", primals_122: "f32[512, 256, 1, 1][256, 1, 1, 1]cuda:0", primals_123: "i64[][]cuda:0", primals_124: "f32[512][1]cuda:0", primals_125: "f32[512][1]cuda:0", primals_126: "f32[512][1]cuda:0", primals_127: "f32[512][1]cuda:0", primals_128: "f32[256, 512, 1, 1][512, 1, 1, 1]cuda:0", primals_129: "i64[][]cuda:0", primals_130: "f32[256][1]cuda:0", primals_131: "f32[256][1]cuda:0", primals_132: "f32[256][1]cuda:0", primals_133: "f32[256][1]cuda:0", primals_134: "f32[256, 8, 3, 3][72, 9, 3, 1]cuda:0", primals_135: "i64[][]cuda:0", primals_136: "f32[256][1]cuda:0", primals_137: "f32[256][1]cuda:0", primals_138: "f32[256][1]cuda:0", primals_139: "f32[256][1]cuda:0", primals_140: "f32[512, 256, 1, 1][256, 1, 1, 1]cuda:0", primals_141: "i64[][]cuda:0", primals_142: "f32[512][1]cuda:0", primals_143: "f32[512][1]cuda:0", primals_144: "f32[512][1]cuda:0", primals_145: "f32[512][1]cuda:0", primals_146: "f32[512, 512, 1, 1][512, 1, 1, 1]cuda:0", primals_147: "i64[][]cuda:0", primals_148: "f32[512][1]cuda:0", primals_149: "f32[512][1]cuda:0", primals_150: "f32[512][1]cuda:0", primals_151: "f32[512][1]cuda:0", primals_152: "f32[512, 16, 3, 3][144, 9, 3, 1]cuda:0", primals_153: "i64[][]cuda:0", primals_154: "f32[512][1]cuda:0", primals_155: "f32[512][1]cuda:0", primals_156: "f32[512][1]cuda:0", primals_157: "f32[512][1]cuda:0", primals_158: "f32[1024, 512, 1, 1][512, 1, 1, 1]cuda:0", primals_159: "i64[][]cuda:0", primals_160: "f32[1024][1]cuda:0", primals_161: "f32[1024][1]cuda:0", primals_162: "f32[1024][1]cuda:0", primals_163: "f32[1024][1]cuda:0", primals_164: "f32[1024, 512, 1, 1][512, 1, 1, 1]cuda:0", primals_165: "i64[][]cuda:0", primals_166: "f32[1024][1]cuda:0", primals_167: "f32[1024][1]cuda:0", primals_168: "f32[1024][1]cuda:0", primals_169: "f32[1024][1]cuda:0", primals_170: "f32[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0", primals_171: "i64[][]cuda:0", primals_172: "f32[512][1]cuda:0", primals_173: "f32[512][1]cuda:0", primals_174: "f32[512][1]cuda:0", primals_175: "f32[512][1]cuda:0", primals_176: "f32[512, 16, 3, 3][144, 9, 3, 1]cuda:0", primals_177: "i64[][]cuda:0", primals_178: "f32[512][1]cuda:0", primals_179: "f32[512][1]cuda:0", primals_180: "f32[512][1]cuda:0", primals_181: "f32[512][1]cuda:0", primals_182: "f32[1024, 512, 1, 1][512, 1, 1, 1]cuda:0", primals_183: "i64[][]cuda:0", primals_184: "f32[1024][1]cuda:0", primals_185: "f32[1024][1]cuda:0", primals_186: "f32[1024][1]cuda:0", primals_187: "f32[1024][1]cuda:0", primals_188: "f32[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0", primals_189: "i64[][]cuda:0", primals_190: "f32[512][1]cuda:0", primals_191: "f32[512][1]cuda:0", primals_192: "f32[512][1]cuda:0", primals_193: "f32[512][1]cuda:0", primals_194: "f32[512, 16, 3, 3][144, 9, 3, 1]cuda:0", primals_195: "i64[][]cuda:0", primals_196: "f32[512][1]cuda:0", primals_197: "f32[512][1]cuda:0", primals_198: "f32[512][1]cuda:0", primals_199: "f32[512][1]cuda:0", primals_200: "f32[1024, 512, 1, 1][512, 1, 1, 1]cuda:0", primals_201: "i64[][]cuda:0", primals_202: "f32[1024][1]cuda:0", primals_203: "f32[1024][1]cuda:0", primals_204: "f32[1024][1]cuda:0", primals_205: "f32[1024][1]cuda:0", primals_206: "f32[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0", primals_207: "i64[][]cuda:0", primals_208: "f32[512][1]cuda:0", primals_209: "f32[512][1]cuda:0", primals_210: "f32[512][1]cuda:0", primals_211: "f32[512][1]cuda:0", primals_212: "f32[512, 16, 3, 3][144, 9, 3, 1]cuda:0", primals_213: "i64[][]cuda:0", primals_214: "f32[512][1]cuda:0", primals_215: "f32[512][1]cuda:0", primals_216: "f32[512][1]cuda:0", primals_217: "f32[512][1]cuda:0", primals_218: "f32[1024, 512, 1, 1][512, 1, 1, 1]cuda:0", primals_219: "i64[][]cuda:0", primals_220: "f32[1024][1]cuda:0", primals_221: "f32[1024][1]cuda:0", primals_222: "f32[1024][1]cuda:0", primals_223: "f32[1024][1]cuda:0", primals_224: "f32[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0", primals_225: "i64[][]cuda:0", primals_226: "f32[512][1]cuda:0", primals_227: "f32[512][1]cuda:0", primals_228: "f32[512][1]cuda:0", primals_229: "f32[512][1]cuda:0", primals_230: "f32[512, 16, 3, 3][144, 9, 3, 1]cuda:0", primals_231: "i64[][]cuda:0", primals_232: "f32[512][1]cuda:0", primals_233: "f32[512][1]cuda:0", primals_234: "f32[512][1]cuda:0", primals_235: "f32[512][1]cuda:0", primals_236: "f32[1024, 512, 1, 1][512, 1, 1, 1]cuda:0", primals_237: "i64[][]cuda:0", primals_238: "f32[1024][1]cuda:0", primals_239: "f32[1024][1]cuda:0", primals_240: "f32[1024][1]cuda:0", primals_241: "f32[1024][1]cuda:0", primals_242: "f32[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0", primals_243: "i64[][]cuda:0", primals_244: "f32[512][1]cuda:0", primals_245: "f32[512][1]cuda:0", primals_246: "f32[512][1]cuda:0", primals_247: "f32[512][1]cuda:0", primals_248: "f32[512, 16, 3, 3][144, 9, 3, 1]cuda:0", primals_249: "i64[][]cuda:0", primals_250: "f32[512][1]cuda:0", primals_251: "f32[512][1]cuda:0", primals_252: "f32[512][1]cuda:0", primals_253: "f32[512][1]cuda:0", primals_254: "f32[1024, 512, 1, 1][512, 1, 1, 1]cuda:0", primals_255: "i64[][]cuda:0", primals_256: "f32[1024][1]cuda:0", primals_257: "f32[1024][1]cuda:0", primals_258: "f32[1024][1]cuda:0", primals_259: "f32[1024][1]cuda:0", primals_260: "f32[1024, 1024, 1, 1][1024, 1, 1, 1]cuda:0", primals_261: "i64[][]cuda:0", primals_262: "f32[1024][1]cuda:0", primals_263: "f32[1024][1]cuda:0", primals_264: "f32[1024][1]cuda:0", primals_265: "f32[1024][1]cuda:0", primals_266: "f32[1024, 32, 3, 3][288, 9, 3, 1]cuda:0", primals_267: "i64[][]cuda:0", primals_268: "f32[1024][1]cuda:0", primals_269: "f32[1024][1]cuda:0", primals_270: "f32[1024][1]cuda:0", primals_271: "f32[1024][1]cuda:0", primals_272: "f32[2048, 1024, 1, 1][1024, 1, 1, 1]cuda:0", primals_273: "i64[][]cuda:0", primals_274: "f32[2048][1]cuda:0", primals_275: "f32[2048][1]cuda:0", primals_276: "f32[2048][1]cuda:0", primals_277: "f32[2048][1]cuda:0", primals_278: "f32[2048, 1024, 1, 1][1024, 1, 1, 1]cuda:0", primals_279: "i64[][]cuda:0", primals_280: "f32[2048][1]cuda:0", primals_281: "f32[2048][1]cuda:0", primals_282: "f32[2048][1]cuda:0", primals_283: "f32[2048][1]cuda:0", primals_284: "f32[1024, 2048, 1, 1][2048, 1, 1, 1]cuda:0", primals_285: "i64[][]cuda:0", primals_286: "f32[1024][1]cuda:0", primals_287: "f32[1024][1]cuda:0", primals_288: "f32[1024][1]cuda:0", primals_289: "f32[1024][1]cuda:0", primals_290: "f32[1024, 32, 3, 3][288, 9, 3, 1]cuda:0", primals_291: "i64[][]cuda:0", primals_292: "f32[1024][1]cuda:0", primals_293: "f32[1024][1]cuda:0", primals_294: "f32[1024][1]cuda:0", primals_295: "f32[1024][1]cuda:0", primals_296: "f32[2048, 1024, 1, 1][1024, 1, 1, 1]cuda:0", primals_297: "i64[][]cuda:0", primals_298: "f32[2048][1]cuda:0", primals_299: "f32[2048][1]cuda:0", primals_300: "f32[2048][1]cuda:0", primals_301: "f32[2048][1]cuda:0", primals_302: "f32[1024, 2048, 1, 1][2048, 1, 1, 1]cuda:0", primals_303: "i64[][]cuda:0", primals_304: "f32[1024][1]cuda:0", primals_305: "f32[1024][1]cuda:0", primals_306: "f32[1024][1]cuda:0", primals_307: "f32[1024][1]cuda:0", primals_308: "f32[1024, 32, 3, 3][288, 9, 3, 1]cuda:0", primals_309: "i64[][]cuda:0", primals_310: "f32[1024][1]cuda:0", primals_311: "f32[1024][1]cuda:0", primals_312: "f32[1024][1]cuda:0", primals_313: "f32[1024][1]cuda:0", primals_314: "f32[2048, 1024, 1, 1][1024, 1, 1, 1]cuda:0", primals_315: "i64[][]cuda:0", primals_316: "f32[2048][1]cuda:0", primals_317: "f32[2048][1]cuda:0", primals_318: "f32[2048][1]cuda:0", primals_319: "f32[2048][1]cuda:0", primals_320: "f32[1000, 2048][2048, 1]cuda:0", primals_321: "f32[1000][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:268 in _forward_impl, code: x = self.conv1(x)
        convert_element_type: "bf16[64, 3, 7, 7][147, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_1, torch.bfloat16);  primals_1 = None
        convert_element_type_1: "bf16[8, 3, 224, 224][150528, 50176, 224, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16);  primals_2 = None
        convolution: "bf16[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_1, convert_element_type, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:269 in _forward_impl, code: x = self.bn1(x)
        add: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_3, 1)
        convert_element_type_2: "f32[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_2, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_2 = None
        getitem: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        add_1: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul: "f32[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3])
        mul_1: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze, 0.1);  squeeze = None
        mul_2: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_4, 0.9)
        add_2: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_2: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_3: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_2, 1.00000996502277);  squeeze_2 = None
        mul_4: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
        mul_5: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_5, 0.9)
        add_3: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_6, -1)
        unsqueeze_1: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_7, -1)
        unsqueeze_3: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_4: "f32[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        convert_element_type_3: "bf16[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:270 in _forward_impl, code: x = self.relu(x)
        relu: "bf16[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_3);  convert_element_type_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:271 in _forward_impl, code: x = self.maxpool(x)
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu, [3, 3], [2, 2], [1, 1], [1, 1], False);  relu = None
        getitem_2: "bf16[8, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = _low_memory_max_pool_with_offsets[0]
        getitem_3: "i8[8, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = _low_memory_max_pool_with_offsets[1];  _low_memory_max_pool_with_offsets = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convert_element_type_4: "bf16[128, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_8, torch.bfloat16);  primals_8 = None
        convolution_1: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(getitem_2, convert_element_type_4, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_5: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_9, 1)
        convert_element_type_5: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_5, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_5 = None
        getitem_4: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_1[0]
        getitem_5: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        add_6: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
        rsqrt_1: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_1: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_1, getitem_5)
        mul_7: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        squeeze_3: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        squeeze_4: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_8: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
        mul_9: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_10, 0.9)
        add_7: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        squeeze_5: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_10: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000398612827361);  squeeze_5 = None
        mul_11: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
        mul_12: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_11, 0.9)
        add_8: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        unsqueeze_4: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_12, -1)
        unsqueeze_5: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_13: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
        unsqueeze_6: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_13, -1);  primals_13 = None
        unsqueeze_7: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_9: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
        convert_element_type_6: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_9, torch.bfloat16);  add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_1: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_6);  convert_element_type_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convert_element_type_7: "bf16[128, 4, 3, 3][36, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
        convolution_2: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_1, convert_element_type_7, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_10: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_15, 1)
        convert_element_type_8: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_2, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_8, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_8 = None
        getitem_6: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_2[0]
        getitem_7: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        add_11: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
        rsqrt_2: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_2: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_2, getitem_7)
        mul_14: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        squeeze_6: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        squeeze_7: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_15: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
        mul_16: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_16, 0.9)
        add_12: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
        squeeze_8: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_17: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0000398612827361);  squeeze_8 = None
        mul_18: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
        mul_19: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_17, 0.9)
        add_13: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
        unsqueeze_8: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_18, -1)
        unsqueeze_9: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        mul_20: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
        unsqueeze_10: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_19, -1);  primals_19 = None
        unsqueeze_11: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        add_14: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
        convert_element_type_9: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_14, torch.bfloat16);  add_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_2: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_9);  convert_element_type_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convert_element_type_10: "bf16[256, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_20, torch.bfloat16);  primals_20 = None
        convolution_3: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_2, convert_element_type_10, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_15: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_21, 1)
        convert_element_type_11: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_3, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_11, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_11 = None
        getitem_8: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_3[0]
        getitem_9: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        add_16: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
        rsqrt_3: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_3: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_3, getitem_9)
        mul_21: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        squeeze_9: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        squeeze_10: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_22: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
        mul_23: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_22, 0.9)
        add_17: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
        squeeze_11: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_24: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0000398612827361);  squeeze_11 = None
        mul_25: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
        mul_26: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_23, 0.9)
        add_18: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
        unsqueeze_12: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_24, -1)
        unsqueeze_13: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_27: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
        unsqueeze_14: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_25, -1);  primals_25 = None
        unsqueeze_15: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_19: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
        convert_element_type_12: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_19, torch.bfloat16);  add_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        convert_element_type_13: "bf16[256, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_26, torch.bfloat16);  primals_26 = None
        convolution_4: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(getitem_2, convert_element_type_13, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_20: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_27, 1)
        convert_element_type_14: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_4, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_14, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_14 = None
        getitem_10: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_4[0]
        getitem_11: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        add_21: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
        rsqrt_4: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_4: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_4, getitem_11)
        mul_28: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        squeeze_12: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        squeeze_13: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_29: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
        mul_30: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_28, 0.9)
        add_22: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
        squeeze_14: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_31: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000398612827361);  squeeze_14 = None
        mul_32: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
        mul_33: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_29, 0.9)
        add_23: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
        unsqueeze_16: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_30, -1)
        unsqueeze_17: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        mul_34: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
        unsqueeze_18: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_31, -1);  primals_31 = None
        unsqueeze_19: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        add_24: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None
        convert_element_type_15: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_24, torch.bfloat16);  add_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_25: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_12, convert_element_type_15);  convert_element_type_12 = convert_element_type_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_3: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(add_25);  add_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convert_element_type_16: "bf16[128, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_32, torch.bfloat16);  primals_32 = None
        convolution_5: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_3, convert_element_type_16, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_26: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_33, 1)
        convert_element_type_17: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_5, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_17, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_17 = None
        getitem_12: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_5[0]
        getitem_13: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        add_27: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
        rsqrt_5: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        sub_5: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_5, getitem_13)
        mul_35: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        squeeze_15: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        squeeze_16: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_36: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
        mul_37: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_34, 0.9)
        add_28: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
        squeeze_17: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
        mul_38: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0000398612827361);  squeeze_17 = None
        mul_39: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
        mul_40: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_35, 0.9)
        add_29: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
        unsqueeze_20: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_36, -1)
        unsqueeze_21: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_41: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
        unsqueeze_22: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_37, -1);  primals_37 = None
        unsqueeze_23: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_30: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
        convert_element_type_18: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_30, torch.bfloat16);  add_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_4: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_18);  convert_element_type_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convert_element_type_19: "bf16[128, 4, 3, 3][36, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_38, torch.bfloat16);  primals_38 = None
        convolution_6: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_4, convert_element_type_19, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_31: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_39, 1)
        convert_element_type_20: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_6, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_20, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_20 = None
        getitem_14: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_6[0]
        getitem_15: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        add_32: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
        rsqrt_6: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        sub_6: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_6, getitem_15)
        mul_42: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        squeeze_18: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        squeeze_19: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_43: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
        mul_44: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_40, 0.9)
        add_33: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
        squeeze_20: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
        mul_45: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0000398612827361);  squeeze_20 = None
        mul_46: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_45, 0.1);  mul_45 = None
        mul_47: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_41, 0.9)
        add_34: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
        unsqueeze_24: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_42, -1)
        unsqueeze_25: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        mul_48: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_25);  mul_42 = unsqueeze_25 = None
        unsqueeze_26: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_43, -1);  primals_43 = None
        unsqueeze_27: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        add_35: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_48, unsqueeze_27);  mul_48 = unsqueeze_27 = None
        convert_element_type_21: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_35, torch.bfloat16);  add_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_5: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_21);  convert_element_type_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convert_element_type_22: "bf16[256, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16);  primals_44 = None
        convolution_7: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_5, convert_element_type_22, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_36: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_45, 1)
        convert_element_type_23: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_7, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_23, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_23 = None
        getitem_16: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_7[0]
        getitem_17: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        add_37: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
        rsqrt_7: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        sub_7: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_7, getitem_17)
        mul_49: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        squeeze_21: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
        squeeze_22: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
        mul_50: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
        mul_51: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_46, 0.9)
        add_38: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
        squeeze_23: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
        mul_52: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0000398612827361);  squeeze_23 = None
        mul_53: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_52, 0.1);  mul_52 = None
        mul_54: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_47, 0.9)
        add_39: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
        unsqueeze_28: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_48, -1)
        unsqueeze_29: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_55: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
        unsqueeze_30: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_49, -1);  primals_49 = None
        unsqueeze_31: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_40: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None
        convert_element_type_24: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_40, torch.bfloat16);  add_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_41: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_24, relu_3);  convert_element_type_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_6: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(add_41);  add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convert_element_type_25: "bf16[128, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_50, torch.bfloat16);  primals_50 = None
        convolution_8: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_6, convert_element_type_25, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_42: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_51, 1)
        convert_element_type_26: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_8, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_26, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_26 = None
        getitem_18: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_8[0]
        getitem_19: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        add_43: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
        rsqrt_8: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        sub_8: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_8, getitem_19)
        mul_56: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        squeeze_24: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        squeeze_25: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_57: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
        mul_58: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_52, 0.9)
        add_44: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
        squeeze_26: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
        mul_59: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0000398612827361);  squeeze_26 = None
        mul_60: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
        mul_61: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_53, 0.9)
        add_45: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
        unsqueeze_32: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_54, -1)
        unsqueeze_33: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        mul_62: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
        unsqueeze_34: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_55, -1);  primals_55 = None
        unsqueeze_35: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        add_46: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None
        convert_element_type_27: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_46, torch.bfloat16);  add_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_7: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_27);  convert_element_type_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convert_element_type_28: "bf16[128, 4, 3, 3][36, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_56, torch.bfloat16);  primals_56 = None
        convolution_9: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_7, convert_element_type_28, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_47: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_57, 1)
        convert_element_type_29: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_9, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_29, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_29 = None
        getitem_20: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_9[0]
        getitem_21: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        add_48: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
        rsqrt_9: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        sub_9: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_9, getitem_21)
        mul_63: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        squeeze_27: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        squeeze_28: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_64: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
        mul_65: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_58, 0.9)
        add_49: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
        squeeze_29: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        mul_66: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0000398612827361);  squeeze_29 = None
        mul_67: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, 0.1);  mul_66 = None
        mul_68: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_59, 0.9)
        add_50: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
        unsqueeze_36: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_60, -1)
        unsqueeze_37: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_69: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_37);  mul_63 = unsqueeze_37 = None
        unsqueeze_38: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_61, -1);  primals_61 = None
        unsqueeze_39: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_51: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_39);  mul_69 = unsqueeze_39 = None
        convert_element_type_30: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_51, torch.bfloat16);  add_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_8: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_30);  convert_element_type_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convert_element_type_31: "bf16[256, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_62, torch.bfloat16);  primals_62 = None
        convolution_10: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_8, convert_element_type_31, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_52: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_63, 1)
        convert_element_type_32: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_10, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_32, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_32 = None
        getitem_22: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_10[0]
        getitem_23: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        add_53: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
        rsqrt_10: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_10: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_10, getitem_23)
        mul_70: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        squeeze_30: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
        squeeze_31: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_71: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
        mul_72: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_64, 0.9)
        add_54: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_71, mul_72);  mul_71 = mul_72 = None
        squeeze_32: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
        mul_73: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0000398612827361);  squeeze_32 = None
        mul_74: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_73, 0.1);  mul_73 = None
        mul_75: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_65, 0.9)
        add_55: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
        unsqueeze_40: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_66, -1)
        unsqueeze_41: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        mul_76: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_41);  mul_70 = unsqueeze_41 = None
        unsqueeze_42: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_67, -1);  primals_67 = None
        unsqueeze_43: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        add_56: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_76, unsqueeze_43);  mul_76 = unsqueeze_43 = None
        convert_element_type_33: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_56, torch.bfloat16);  add_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_57: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_33, relu_6);  convert_element_type_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_9: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(add_57);  add_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convert_element_type_34: "bf16[256, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_68, torch.bfloat16);  primals_68 = None
        convolution_11: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_9, convert_element_type_34, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_58: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_69, 1)
        convert_element_type_35: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_11, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_35, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_35 = None
        getitem_24: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_11[0]
        getitem_25: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        add_59: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
        rsqrt_11: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
        sub_11: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_11, getitem_25)
        mul_77: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        squeeze_33: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        squeeze_34: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
        mul_78: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
        mul_79: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_70, 0.9)
        add_60: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
        squeeze_35: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
        mul_80: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0000398612827361);  squeeze_35 = None
        mul_81: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_80, 0.1);  mul_80 = None
        mul_82: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_71, 0.9)
        add_61: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
        unsqueeze_44: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_72, -1)
        unsqueeze_45: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_83: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_45);  mul_77 = unsqueeze_45 = None
        unsqueeze_46: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_73, -1);  primals_73 = None
        unsqueeze_47: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_62: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_47);  mul_83 = unsqueeze_47 = None
        convert_element_type_36: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_62, torch.bfloat16);  add_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_10: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_36);  convert_element_type_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convert_element_type_37: "bf16[256, 8, 3, 3][72, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_74, torch.bfloat16);  primals_74 = None
        convolution_12: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_10, convert_element_type_37, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_63: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_75, 1)
        convert_element_type_38: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_12, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_38, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_38 = None
        getitem_26: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_12[0]
        getitem_27: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        add_64: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
        rsqrt_12: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        sub_12: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_12, getitem_27)
        mul_84: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        squeeze_36: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        squeeze_37: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_85: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
        mul_86: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_76, 0.9)
        add_65: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
        squeeze_38: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        mul_87: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0001594642002871);  squeeze_38 = None
        mul_88: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_87, 0.1);  mul_87 = None
        mul_89: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_77, 0.9)
        add_66: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
        unsqueeze_48: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_78, -1)
        unsqueeze_49: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        mul_90: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_49);  mul_84 = unsqueeze_49 = None
        unsqueeze_50: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_79, -1);  primals_79 = None
        unsqueeze_51: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        add_67: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_51);  mul_90 = unsqueeze_51 = None
        convert_element_type_39: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_67, torch.bfloat16);  add_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_11: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_39);  convert_element_type_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convert_element_type_40: "bf16[512, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_80, torch.bfloat16);  primals_80 = None
        convolution_13: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_11, convert_element_type_40, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_68: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_81, 1)
        convert_element_type_41: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_13, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_41, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_41 = None
        getitem_28: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_13[0]
        getitem_29: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        add_69: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
        rsqrt_13: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_13: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_13, getitem_29)
        mul_91: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        squeeze_39: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        squeeze_40: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
        mul_92: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
        mul_93: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_82, 0.9)
        add_70: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
        squeeze_41: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        mul_94: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0001594642002871);  squeeze_41 = None
        mul_95: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_94, 0.1);  mul_94 = None
        mul_96: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_83, 0.9)
        add_71: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None
        unsqueeze_52: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_84, -1)
        unsqueeze_53: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_97: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_53);  mul_91 = unsqueeze_53 = None
        unsqueeze_54: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_85, -1);  primals_85 = None
        unsqueeze_55: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_72: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_55);  mul_97 = unsqueeze_55 = None
        convert_element_type_42: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_72, torch.bfloat16);  add_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        convert_element_type_43: "bf16[512, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_86, torch.bfloat16);  primals_86 = None
        convolution_14: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_9, convert_element_type_43, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        add_73: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_87, 1)
        convert_element_type_44: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_14, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_44, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_44 = None
        getitem_30: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_14[0]
        getitem_31: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        add_74: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
        rsqrt_14: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_14: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_14, getitem_31)
        mul_98: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        squeeze_42: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
        squeeze_43: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_99: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
        mul_100: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_88, 0.9)
        add_75: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
        squeeze_44: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
        mul_101: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0001594642002871);  squeeze_44 = None
        mul_102: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_101, 0.1);  mul_101 = None
        mul_103: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_89, 0.9)
        add_76: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
        unsqueeze_56: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_90, -1)
        unsqueeze_57: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        mul_104: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_57);  mul_98 = unsqueeze_57 = None
        unsqueeze_58: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_91, -1);  primals_91 = None
        unsqueeze_59: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        add_77: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_59);  mul_104 = unsqueeze_59 = None
        convert_element_type_45: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_77, torch.bfloat16);  add_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_78: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_42, convert_element_type_45);  convert_element_type_42 = convert_element_type_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_12: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(add_78);  add_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convert_element_type_46: "bf16[256, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_92, torch.bfloat16);  primals_92 = None
        convolution_15: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_12, convert_element_type_46, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_79: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_93, 1)
        convert_element_type_47: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_15, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_47, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_47 = None
        getitem_32: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_15[0]
        getitem_33: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        add_80: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
        rsqrt_15: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        sub_15: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_15, getitem_33)
        mul_105: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        squeeze_45: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        squeeze_46: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_106: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
        mul_107: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_94, 0.9)
        add_81: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
        squeeze_47: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
        mul_108: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0001594642002871);  squeeze_47 = None
        mul_109: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_108, 0.1);  mul_108 = None
        mul_110: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_95, 0.9)
        add_82: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
        unsqueeze_60: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_96, -1)
        unsqueeze_61: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_111: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_61);  mul_105 = unsqueeze_61 = None
        unsqueeze_62: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_97, -1);  primals_97 = None
        unsqueeze_63: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_83: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_63);  mul_111 = unsqueeze_63 = None
        convert_element_type_48: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_83, torch.bfloat16);  add_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_13: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_48);  convert_element_type_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convert_element_type_49: "bf16[256, 8, 3, 3][72, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_98, torch.bfloat16);  primals_98 = None
        convolution_16: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_13, convert_element_type_49, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_84: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_99, 1)
        convert_element_type_50: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_16, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_50, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_50 = None
        getitem_34: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_16[0]
        getitem_35: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        add_85: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
        rsqrt_16: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        sub_16: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_16, getitem_35)
        mul_112: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        squeeze_48: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
        squeeze_49: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_113: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
        mul_114: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_100, 0.9)
        add_86: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
        squeeze_50: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
        mul_115: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0001594642002871);  squeeze_50 = None
        mul_116: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_115, 0.1);  mul_115 = None
        mul_117: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_101, 0.9)
        add_87: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
        unsqueeze_64: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_102, -1)
        unsqueeze_65: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        mul_118: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_65);  mul_112 = unsqueeze_65 = None
        unsqueeze_66: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_103, -1);  primals_103 = None
        unsqueeze_67: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        add_88: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_67);  mul_118 = unsqueeze_67 = None
        convert_element_type_51: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_88, torch.bfloat16);  add_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_14: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_51);  convert_element_type_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convert_element_type_52: "bf16[512, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_104, torch.bfloat16);  primals_104 = None
        convolution_17: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_14, convert_element_type_52, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_89: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_105, 1)
        convert_element_type_53: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_17, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_53, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_53 = None
        getitem_36: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_17[0]
        getitem_37: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        add_90: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
        rsqrt_17: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        sub_17: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_17, getitem_37)
        mul_119: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        squeeze_51: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
        squeeze_52: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
        mul_120: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
        mul_121: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_106, 0.9)
        add_91: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_120, mul_121);  mul_120 = mul_121 = None
        squeeze_53: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
        mul_122: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0001594642002871);  squeeze_53 = None
        mul_123: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_122, 0.1);  mul_122 = None
        mul_124: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_107, 0.9)
        add_92: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
        unsqueeze_68: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_108, -1)
        unsqueeze_69: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_125: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_69);  mul_119 = unsqueeze_69 = None
        unsqueeze_70: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_109, -1);  primals_109 = None
        unsqueeze_71: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_93: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_71);  mul_125 = unsqueeze_71 = None
        convert_element_type_54: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_93, torch.bfloat16);  add_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_94: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_54, relu_12);  convert_element_type_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_15: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(add_94);  add_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convert_element_type_55: "bf16[256, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_110, torch.bfloat16);  primals_110 = None
        convolution_18: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_15, convert_element_type_55, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_95: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_111, 1)
        convert_element_type_56: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_18, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_56, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_56 = None
        getitem_38: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_18[0]
        getitem_39: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        add_96: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_38, 1e-05)
        rsqrt_18: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        sub_18: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_18, getitem_39)
        mul_126: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        squeeze_54: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
        squeeze_55: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
        mul_127: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
        mul_128: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_112, 0.9)
        add_97: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
        squeeze_56: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
        mul_129: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0001594642002871);  squeeze_56 = None
        mul_130: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_129, 0.1);  mul_129 = None
        mul_131: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_113, 0.9)
        add_98: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
        unsqueeze_72: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_114, -1)
        unsqueeze_73: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        mul_132: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_126, unsqueeze_73);  mul_126 = unsqueeze_73 = None
        unsqueeze_74: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_115, -1);  primals_115 = None
        unsqueeze_75: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        add_99: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_132, unsqueeze_75);  mul_132 = unsqueeze_75 = None
        convert_element_type_57: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_99, torch.bfloat16);  add_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_16: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_57);  convert_element_type_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convert_element_type_58: "bf16[256, 8, 3, 3][72, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_116, torch.bfloat16);  primals_116 = None
        convolution_19: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_16, convert_element_type_58, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_100: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_117, 1)
        convert_element_type_59: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_19, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_59, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_59 = None
        getitem_40: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_19[0]
        getitem_41: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        add_101: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_40, 1e-05)
        rsqrt_19: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
        sub_19: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_19, getitem_41)
        mul_133: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        squeeze_57: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
        squeeze_58: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
        mul_134: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
        mul_135: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_118, 0.9)
        add_102: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
        squeeze_59: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
        mul_136: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0001594642002871);  squeeze_59 = None
        mul_137: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_136, 0.1);  mul_136 = None
        mul_138: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_119, 0.9)
        add_103: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
        unsqueeze_76: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_120, -1)
        unsqueeze_77: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_139: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_77);  mul_133 = unsqueeze_77 = None
        unsqueeze_78: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_121, -1);  primals_121 = None
        unsqueeze_79: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_104: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_139, unsqueeze_79);  mul_139 = unsqueeze_79 = None
        convert_element_type_60: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_104, torch.bfloat16);  add_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_17: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_60);  convert_element_type_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convert_element_type_61: "bf16[512, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_122, torch.bfloat16);  primals_122 = None
        convolution_20: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_17, convert_element_type_61, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_105: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_123, 1)
        convert_element_type_62: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_20, torch.float32)
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_62, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_62 = None
        getitem_42: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_20[0]
        getitem_43: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None
        add_106: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_42, 1e-05)
        rsqrt_20: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        sub_20: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_20, getitem_43)
        mul_140: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
        squeeze_60: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
        squeeze_61: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
        mul_141: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
        mul_142: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_124, 0.9)
        add_107: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None
        squeeze_62: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
        mul_143: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0001594642002871);  squeeze_62 = None
        mul_144: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_143, 0.1);  mul_143 = None
        mul_145: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_125, 0.9)
        add_108: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
        unsqueeze_80: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_126, -1)
        unsqueeze_81: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        mul_146: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_140, unsqueeze_81);  mul_140 = unsqueeze_81 = None
        unsqueeze_82: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_127, -1);  primals_127 = None
        unsqueeze_83: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        add_109: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_83);  mul_146 = unsqueeze_83 = None
        convert_element_type_63: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_109, torch.bfloat16);  add_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_110: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_63, relu_15);  convert_element_type_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_18: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(add_110);  add_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convert_element_type_64: "bf16[256, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_128, torch.bfloat16);  primals_128 = None
        convolution_21: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_18, convert_element_type_64, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_111: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_129, 1)
        convert_element_type_65: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_21, torch.float32)
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_65, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_65 = None
        getitem_44: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_21[0]
        getitem_45: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        add_112: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_44, 1e-05)
        rsqrt_21: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
        sub_21: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_21, getitem_45)
        mul_147: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
        squeeze_63: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
        squeeze_64: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
        mul_148: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
        mul_149: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_130, 0.9)
        add_113: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None
        squeeze_65: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
        mul_150: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0001594642002871);  squeeze_65 = None
        mul_151: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_150, 0.1);  mul_150 = None
        mul_152: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_131, 0.9)
        add_114: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
        unsqueeze_84: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_132, -1)
        unsqueeze_85: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_153: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_147, unsqueeze_85);  mul_147 = unsqueeze_85 = None
        unsqueeze_86: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_133, -1);  primals_133 = None
        unsqueeze_87: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_115: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_153, unsqueeze_87);  mul_153 = unsqueeze_87 = None
        convert_element_type_66: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_115, torch.bfloat16);  add_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_19: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_66);  convert_element_type_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convert_element_type_67: "bf16[256, 8, 3, 3][72, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_134, torch.bfloat16);  primals_134 = None
        convolution_22: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_19, convert_element_type_67, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_116: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_135, 1)
        convert_element_type_68: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_22, torch.float32)
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_68, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_68 = None
        getitem_46: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_22[0]
        getitem_47: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None
        add_117: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
        rsqrt_22: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
        sub_22: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_22, getitem_47)
        mul_154: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        squeeze_66: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
        squeeze_67: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
        mul_155: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
        mul_156: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_136, 0.9)
        add_118: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
        squeeze_68: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
        mul_157: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0001594642002871);  squeeze_68 = None
        mul_158: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_157, 0.1);  mul_157 = None
        mul_159: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_137, 0.9)
        add_119: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None
        unsqueeze_88: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_138, -1)
        unsqueeze_89: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        mul_160: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_89);  mul_154 = unsqueeze_89 = None
        unsqueeze_90: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_139, -1);  primals_139 = None
        unsqueeze_91: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        add_120: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_160, unsqueeze_91);  mul_160 = unsqueeze_91 = None
        convert_element_type_69: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_120, torch.bfloat16);  add_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_20: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_69);  convert_element_type_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convert_element_type_70: "bf16[512, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_140, torch.bfloat16);  primals_140 = None
        convolution_23: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_20, convert_element_type_70, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_121: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_141, 1)
        convert_element_type_71: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_23, torch.float32)
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_71, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_71 = None
        getitem_48: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_23[0]
        getitem_49: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        add_122: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_48, 1e-05)
        rsqrt_23: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        sub_23: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_23, getitem_49)
        mul_161: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
        squeeze_69: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
        squeeze_70: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
        mul_162: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
        mul_163: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_142, 0.9)
        add_123: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
        squeeze_71: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
        mul_164: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0001594642002871);  squeeze_71 = None
        mul_165: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_164, 0.1);  mul_164 = None
        mul_166: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_143, 0.9)
        add_124: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
        unsqueeze_92: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_144, -1)
        unsqueeze_93: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        mul_167: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_93);  mul_161 = unsqueeze_93 = None
        unsqueeze_94: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_145, -1);  primals_145 = None
        unsqueeze_95: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        add_125: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_95);  mul_167 = unsqueeze_95 = None
        convert_element_type_72: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_125, torch.bfloat16);  add_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_126: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_72, relu_18);  convert_element_type_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_21: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(add_126);  add_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convert_element_type_73: "bf16[512, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_146, torch.bfloat16);  primals_146 = None
        convolution_24: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_21, convert_element_type_73, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_127: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_147, 1)
        convert_element_type_74: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_24, torch.float32)
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_74, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_74 = None
        getitem_50: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_24[0]
        getitem_51: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None
        add_128: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_50, 1e-05)
        rsqrt_24: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
        sub_24: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_24, getitem_51)
        mul_168: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        squeeze_72: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
        squeeze_73: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
        mul_169: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
        mul_170: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_148, 0.9)
        add_129: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
        squeeze_74: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
        mul_171: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0001594642002871);  squeeze_74 = None
        mul_172: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_171, 0.1);  mul_171 = None
        mul_173: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_149, 0.9)
        add_130: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
        unsqueeze_96: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_150, -1)
        unsqueeze_97: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        mul_174: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_168, unsqueeze_97);  mul_168 = unsqueeze_97 = None
        unsqueeze_98: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_151, -1);  primals_151 = None
        unsqueeze_99: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        add_131: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_174, unsqueeze_99);  mul_174 = unsqueeze_99 = None
        convert_element_type_75: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_131, torch.bfloat16);  add_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_22: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_75);  convert_element_type_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convert_element_type_76: "bf16[512, 16, 3, 3][144, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_152, torch.bfloat16);  primals_152 = None
        convolution_25: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_22, convert_element_type_76, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_132: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_153, 1)
        convert_element_type_77: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_25, torch.float32)
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_77, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_77 = None
        getitem_52: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_25[0]
        getitem_53: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_25[1];  var_mean_25 = None
        add_133: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_52, 1e-05)
        rsqrt_25: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
        sub_25: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_25, getitem_53)
        mul_175: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        squeeze_75: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
        squeeze_76: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
        mul_176: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
        mul_177: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_154, 0.9)
        add_134: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
        squeeze_77: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
        mul_178: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0006381620931717);  squeeze_77 = None
        mul_179: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_178, 0.1);  mul_178 = None
        mul_180: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_155, 0.9)
        add_135: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
        unsqueeze_100: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_156, -1)
        unsqueeze_101: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_181: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_101);  mul_175 = unsqueeze_101 = None
        unsqueeze_102: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_157, -1);  primals_157 = None
        unsqueeze_103: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_136: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_103);  mul_181 = unsqueeze_103 = None
        convert_element_type_78: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_136, torch.bfloat16);  add_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_23: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_78);  convert_element_type_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convert_element_type_79: "bf16[1024, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_158, torch.bfloat16);  primals_158 = None
        convolution_26: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_23, convert_element_type_79, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_137: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_159, 1)
        convert_element_type_80: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_26, torch.float32)
        var_mean_26 = torch.ops.aten.var_mean.correction(convert_element_type_80, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_80 = None
        getitem_54: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_26[0]
        getitem_55: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_26[1];  var_mean_26 = None
        add_138: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_54, 1e-05)
        rsqrt_26: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
        sub_26: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_26, getitem_55)
        mul_182: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
        squeeze_78: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
        squeeze_79: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
        mul_183: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
        mul_184: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_160, 0.9)
        add_139: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_183, mul_184);  mul_183 = mul_184 = None
        squeeze_80: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
        mul_185: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0006381620931717);  squeeze_80 = None
        mul_186: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_185, 0.1);  mul_185 = None
        mul_187: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_161, 0.9)
        add_140: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_186, mul_187);  mul_186 = mul_187 = None
        unsqueeze_104: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_162, -1)
        unsqueeze_105: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        mul_188: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_182, unsqueeze_105);  mul_182 = unsqueeze_105 = None
        unsqueeze_106: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_163, -1);  primals_163 = None
        unsqueeze_107: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
        add_141: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_107);  mul_188 = unsqueeze_107 = None
        convert_element_type_81: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_141, torch.bfloat16);  add_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        convert_element_type_82: "bf16[1024, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_164, torch.bfloat16);  primals_164 = None
        convolution_27: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_21, convert_element_type_82, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        add_142: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_165, 1)
        convert_element_type_83: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_27, torch.float32)
        var_mean_27 = torch.ops.aten.var_mean.correction(convert_element_type_83, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_83 = None
        getitem_56: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_27[0]
        getitem_57: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_27[1];  var_mean_27 = None
        add_143: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_56, 1e-05)
        rsqrt_27: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
        sub_27: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_27, getitem_57)
        mul_189: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
        squeeze_81: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
        squeeze_82: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
        mul_190: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
        mul_191: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_166, 0.9)
        add_144: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
        squeeze_83: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
        mul_192: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0006381620931717);  squeeze_83 = None
        mul_193: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_192, 0.1);  mul_192 = None
        mul_194: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_167, 0.9)
        add_145: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_193, mul_194);  mul_193 = mul_194 = None
        unsqueeze_108: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_168, -1)
        unsqueeze_109: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
        mul_195: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_189, unsqueeze_109);  mul_189 = unsqueeze_109 = None
        unsqueeze_110: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_169, -1);  primals_169 = None
        unsqueeze_111: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
        add_146: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_195, unsqueeze_111);  mul_195 = unsqueeze_111 = None
        convert_element_type_84: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_146, torch.bfloat16);  add_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_147: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_81, convert_element_type_84);  convert_element_type_81 = convert_element_type_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_24: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(add_147);  add_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convert_element_type_85: "bf16[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_170, torch.bfloat16);  primals_170 = None
        convolution_28: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_24, convert_element_type_85, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_148: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_171, 1)
        convert_element_type_86: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_28, torch.float32)
        var_mean_28 = torch.ops.aten.var_mean.correction(convert_element_type_86, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_86 = None
        getitem_58: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_28[0]
        getitem_59: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_28[1];  var_mean_28 = None
        add_149: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_58, 1e-05)
        rsqrt_28: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
        sub_28: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_28, getitem_59)
        mul_196: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
        squeeze_84: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
        squeeze_85: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
        mul_197: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
        mul_198: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_172, 0.9)
        add_150: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
        squeeze_86: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
        mul_199: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0006381620931717);  squeeze_86 = None
        mul_200: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_199, 0.1);  mul_199 = None
        mul_201: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_173, 0.9)
        add_151: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_200, mul_201);  mul_200 = mul_201 = None
        unsqueeze_112: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_174, -1)
        unsqueeze_113: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        mul_202: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_113);  mul_196 = unsqueeze_113 = None
        unsqueeze_114: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_175, -1);  primals_175 = None
        unsqueeze_115: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        add_152: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_115);  mul_202 = unsqueeze_115 = None
        convert_element_type_87: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_152, torch.bfloat16);  add_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_25: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_87);  convert_element_type_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convert_element_type_88: "bf16[512, 16, 3, 3][144, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_176, torch.bfloat16);  primals_176 = None
        convolution_29: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_25, convert_element_type_88, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_153: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_177, 1)
        convert_element_type_89: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_29, torch.float32)
        var_mean_29 = torch.ops.aten.var_mean.correction(convert_element_type_89, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_89 = None
        getitem_60: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_29[0]
        getitem_61: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_29[1];  var_mean_29 = None
        add_154: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_60, 1e-05)
        rsqrt_29: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
        sub_29: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_29, getitem_61)
        mul_203: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
        squeeze_87: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
        squeeze_88: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
        mul_204: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
        mul_205: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_178, 0.9)
        add_155: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None
        squeeze_89: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
        mul_206: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0006381620931717);  squeeze_89 = None
        mul_207: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_206, 0.1);  mul_206 = None
        mul_208: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_179, 0.9)
        add_156: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_207, mul_208);  mul_207 = mul_208 = None
        unsqueeze_116: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_180, -1)
        unsqueeze_117: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        mul_209: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_117);  mul_203 = unsqueeze_117 = None
        unsqueeze_118: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_181, -1);  primals_181 = None
        unsqueeze_119: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
        add_157: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_119);  mul_209 = unsqueeze_119 = None
        convert_element_type_90: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_157, torch.bfloat16);  add_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_26: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_90);  convert_element_type_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convert_element_type_91: "bf16[1024, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_182, torch.bfloat16);  primals_182 = None
        convolution_30: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_26, convert_element_type_91, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_158: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_183, 1)
        convert_element_type_92: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_30, torch.float32)
        var_mean_30 = torch.ops.aten.var_mean.correction(convert_element_type_92, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_92 = None
        getitem_62: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_30[0]
        getitem_63: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_30[1];  var_mean_30 = None
        add_159: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
        rsqrt_30: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
        sub_30: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_30, getitem_63)
        mul_210: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
        squeeze_90: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
        squeeze_91: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
        mul_211: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
        mul_212: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_184, 0.9)
        add_160: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_211, mul_212);  mul_211 = mul_212 = None
        squeeze_92: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
        mul_213: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0006381620931717);  squeeze_92 = None
        mul_214: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_213, 0.1);  mul_213 = None
        mul_215: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_185, 0.9)
        add_161: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
        unsqueeze_120: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_186, -1)
        unsqueeze_121: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        mul_216: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_121);  mul_210 = unsqueeze_121 = None
        unsqueeze_122: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_187, -1);  primals_187 = None
        unsqueeze_123: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        add_162: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_216, unsqueeze_123);  mul_216 = unsqueeze_123 = None
        convert_element_type_93: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_162, torch.bfloat16);  add_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_163: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_93, relu_24);  convert_element_type_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_27: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(add_163);  add_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convert_element_type_94: "bf16[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_188, torch.bfloat16);  primals_188 = None
        convolution_31: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_27, convert_element_type_94, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_164: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_189, 1)
        convert_element_type_95: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_31, torch.float32)
        var_mean_31 = torch.ops.aten.var_mean.correction(convert_element_type_95, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_95 = None
        getitem_64: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_31[0]
        getitem_65: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_31[1];  var_mean_31 = None
        add_165: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
        rsqrt_31: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
        sub_31: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_31, getitem_65)
        mul_217: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
        squeeze_93: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
        squeeze_94: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
        mul_218: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
        mul_219: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_190, 0.9)
        add_166: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_218, mul_219);  mul_218 = mul_219 = None
        squeeze_95: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
        mul_220: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0006381620931717);  squeeze_95 = None
        mul_221: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_220, 0.1);  mul_220 = None
        mul_222: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_191, 0.9)
        add_167: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
        unsqueeze_124: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_192, -1)
        unsqueeze_125: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        mul_223: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_125);  mul_217 = unsqueeze_125 = None
        unsqueeze_126: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_193, -1);  primals_193 = None
        unsqueeze_127: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        add_168: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_223, unsqueeze_127);  mul_223 = unsqueeze_127 = None
        convert_element_type_96: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_168, torch.bfloat16);  add_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_28: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_96);  convert_element_type_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convert_element_type_97: "bf16[512, 16, 3, 3][144, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_194, torch.bfloat16);  primals_194 = None
        convolution_32: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_28, convert_element_type_97, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_169: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_195, 1)
        convert_element_type_98: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_32, torch.float32)
        var_mean_32 = torch.ops.aten.var_mean.correction(convert_element_type_98, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_98 = None
        getitem_66: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_32[0]
        getitem_67: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_32[1];  var_mean_32 = None
        add_170: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_66, 1e-05)
        rsqrt_32: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        sub_32: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_32, getitem_67)
        mul_224: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
        squeeze_96: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
        squeeze_97: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
        mul_225: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
        mul_226: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_196, 0.9)
        add_171: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
        squeeze_98: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
        mul_227: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0006381620931717);  squeeze_98 = None
        mul_228: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_227, 0.1);  mul_227 = None
        mul_229: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_197, 0.9)
        add_172: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_228, mul_229);  mul_228 = mul_229 = None
        unsqueeze_128: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_198, -1)
        unsqueeze_129: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
        mul_230: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_224, unsqueeze_129);  mul_224 = unsqueeze_129 = None
        unsqueeze_130: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_199, -1);  primals_199 = None
        unsqueeze_131: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
        add_173: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_131);  mul_230 = unsqueeze_131 = None
        convert_element_type_99: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_173, torch.bfloat16);  add_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_29: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_99);  convert_element_type_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convert_element_type_100: "bf16[1024, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_200, torch.bfloat16);  primals_200 = None
        convolution_33: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_29, convert_element_type_100, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_174: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_201, 1)
        convert_element_type_101: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_33, torch.float32)
        var_mean_33 = torch.ops.aten.var_mean.correction(convert_element_type_101, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_101 = None
        getitem_68: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_33[0]
        getitem_69: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_33[1];  var_mean_33 = None
        add_175: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_68, 1e-05)
        rsqrt_33: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
        sub_33: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_33, getitem_69)
        mul_231: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
        squeeze_99: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
        squeeze_100: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
        mul_232: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
        mul_233: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_202, 0.9)
        add_176: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
        squeeze_101: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
        mul_234: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0006381620931717);  squeeze_101 = None
        mul_235: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_234, 0.1);  mul_234 = None
        mul_236: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_203, 0.9)
        add_177: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_235, mul_236);  mul_235 = mul_236 = None
        unsqueeze_132: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_204, -1)
        unsqueeze_133: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
        mul_237: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_231, unsqueeze_133);  mul_231 = unsqueeze_133 = None
        unsqueeze_134: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_205, -1);  primals_205 = None
        unsqueeze_135: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
        add_178: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_237, unsqueeze_135);  mul_237 = unsqueeze_135 = None
        convert_element_type_102: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_178, torch.bfloat16);  add_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_179: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_102, relu_27);  convert_element_type_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_30: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(add_179);  add_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convert_element_type_103: "bf16[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_206, torch.bfloat16);  primals_206 = None
        convolution_34: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_30, convert_element_type_103, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_180: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_207, 1)
        convert_element_type_104: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_34, torch.float32)
        var_mean_34 = torch.ops.aten.var_mean.correction(convert_element_type_104, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_104 = None
        getitem_70: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_34[0]
        getitem_71: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_34[1];  var_mean_34 = None
        add_181: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_70, 1e-05)
        rsqrt_34: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
        sub_34: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_34, getitem_71)
        mul_238: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
        squeeze_102: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
        squeeze_103: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
        mul_239: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
        mul_240: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_208, 0.9)
        add_182: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_239, mul_240);  mul_239 = mul_240 = None
        squeeze_104: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
        mul_241: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0006381620931717);  squeeze_104 = None
        mul_242: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_241, 0.1);  mul_241 = None
        mul_243: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_209, 0.9)
        add_183: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
        unsqueeze_136: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_210, -1)
        unsqueeze_137: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
        mul_244: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_137);  mul_238 = unsqueeze_137 = None
        unsqueeze_138: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_211, -1);  primals_211 = None
        unsqueeze_139: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
        add_184: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_244, unsqueeze_139);  mul_244 = unsqueeze_139 = None
        convert_element_type_105: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_184, torch.bfloat16);  add_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_31: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_105);  convert_element_type_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convert_element_type_106: "bf16[512, 16, 3, 3][144, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_212, torch.bfloat16);  primals_212 = None
        convolution_35: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_31, convert_element_type_106, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_185: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_213, 1)
        convert_element_type_107: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_35, torch.float32)
        var_mean_35 = torch.ops.aten.var_mean.correction(convert_element_type_107, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_107 = None
        getitem_72: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_35[0]
        getitem_73: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_35[1];  var_mean_35 = None
        add_186: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_72, 1e-05)
        rsqrt_35: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
        sub_35: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_35, getitem_73)
        mul_245: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
        squeeze_105: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
        squeeze_106: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
        mul_246: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
        mul_247: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_214, 0.9)
        add_187: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
        squeeze_107: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
        mul_248: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_107, 1.0006381620931717);  squeeze_107 = None
        mul_249: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_248, 0.1);  mul_248 = None
        mul_250: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_215, 0.9)
        add_188: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
        unsqueeze_140: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_216, -1)
        unsqueeze_141: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
        mul_251: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_141);  mul_245 = unsqueeze_141 = None
        unsqueeze_142: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_217, -1);  primals_217 = None
        unsqueeze_143: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
        add_189: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_251, unsqueeze_143);  mul_251 = unsqueeze_143 = None
        convert_element_type_108: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_189, torch.bfloat16);  add_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_32: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_108);  convert_element_type_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convert_element_type_109: "bf16[1024, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_218, torch.bfloat16);  primals_218 = None
        convolution_36: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_32, convert_element_type_109, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_190: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_219, 1)
        convert_element_type_110: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_36, torch.float32)
        var_mean_36 = torch.ops.aten.var_mean.correction(convert_element_type_110, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_110 = None
        getitem_74: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_36[0]
        getitem_75: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_36[1];  var_mean_36 = None
        add_191: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_74, 1e-05)
        rsqrt_36: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
        sub_36: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_36, getitem_75)
        mul_252: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
        squeeze_108: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
        squeeze_109: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
        mul_253: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
        mul_254: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_220, 0.9)
        add_192: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_253, mul_254);  mul_253 = mul_254 = None
        squeeze_110: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
        mul_255: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0006381620931717);  squeeze_110 = None
        mul_256: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_255, 0.1);  mul_255 = None
        mul_257: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_221, 0.9)
        add_193: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
        unsqueeze_144: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_222, -1)
        unsqueeze_145: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
        mul_258: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_252, unsqueeze_145);  mul_252 = unsqueeze_145 = None
        unsqueeze_146: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_223, -1);  primals_223 = None
        unsqueeze_147: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
        add_194: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_258, unsqueeze_147);  mul_258 = unsqueeze_147 = None
        convert_element_type_111: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_194, torch.bfloat16);  add_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_195: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_111, relu_30);  convert_element_type_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_33: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(add_195);  add_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convert_element_type_112: "bf16[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_224, torch.bfloat16);  primals_224 = None
        convolution_37: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_33, convert_element_type_112, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_196: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_225, 1)
        convert_element_type_113: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_37, torch.float32)
        var_mean_37 = torch.ops.aten.var_mean.correction(convert_element_type_113, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_113 = None
        getitem_76: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_37[0]
        getitem_77: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_37[1];  var_mean_37 = None
        add_197: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_76, 1e-05)
        rsqrt_37: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
        sub_37: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_37, getitem_77)
        mul_259: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
        squeeze_111: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
        squeeze_112: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
        mul_260: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
        mul_261: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_226, 0.9)
        add_198: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_260, mul_261);  mul_260 = mul_261 = None
        squeeze_113: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
        mul_262: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0006381620931717);  squeeze_113 = None
        mul_263: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_262, 0.1);  mul_262 = None
        mul_264: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_227, 0.9)
        add_199: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_263, mul_264);  mul_263 = mul_264 = None
        unsqueeze_148: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_228, -1)
        unsqueeze_149: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
        mul_265: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_149);  mul_259 = unsqueeze_149 = None
        unsqueeze_150: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_229, -1);  primals_229 = None
        unsqueeze_151: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
        add_200: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_265, unsqueeze_151);  mul_265 = unsqueeze_151 = None
        convert_element_type_114: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_200, torch.bfloat16);  add_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_34: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_114);  convert_element_type_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convert_element_type_115: "bf16[512, 16, 3, 3][144, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_230, torch.bfloat16);  primals_230 = None
        convolution_38: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_34, convert_element_type_115, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_201: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_231, 1)
        convert_element_type_116: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_38, torch.float32)
        var_mean_38 = torch.ops.aten.var_mean.correction(convert_element_type_116, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_116 = None
        getitem_78: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_38[0]
        getitem_79: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_38[1];  var_mean_38 = None
        add_202: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_78, 1e-05)
        rsqrt_38: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_202);  add_202 = None
        sub_38: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_38, getitem_79)
        mul_266: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
        squeeze_114: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
        squeeze_115: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
        mul_267: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
        mul_268: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_232, 0.9)
        add_203: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_267, mul_268);  mul_267 = mul_268 = None
        squeeze_116: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
        mul_269: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0006381620931717);  squeeze_116 = None
        mul_270: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_269, 0.1);  mul_269 = None
        mul_271: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_233, 0.9)
        add_204: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_270, mul_271);  mul_270 = mul_271 = None
        unsqueeze_152: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_234, -1)
        unsqueeze_153: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
        mul_272: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_266, unsqueeze_153);  mul_266 = unsqueeze_153 = None
        unsqueeze_154: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_235, -1);  primals_235 = None
        unsqueeze_155: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
        add_205: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_272, unsqueeze_155);  mul_272 = unsqueeze_155 = None
        convert_element_type_117: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_205, torch.bfloat16);  add_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_35: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_117);  convert_element_type_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convert_element_type_118: "bf16[1024, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_236, torch.bfloat16);  primals_236 = None
        convolution_39: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_35, convert_element_type_118, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_206: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_237, 1)
        convert_element_type_119: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_39, torch.float32)
        var_mean_39 = torch.ops.aten.var_mean.correction(convert_element_type_119, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_119 = None
        getitem_80: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_39[0]
        getitem_81: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_39[1];  var_mean_39 = None
        add_207: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_80, 1e-05)
        rsqrt_39: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_207);  add_207 = None
        sub_39: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_39, getitem_81)
        mul_273: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
        squeeze_117: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
        squeeze_118: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
        mul_274: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1)
        mul_275: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_238, 0.9)
        add_208: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_274, mul_275);  mul_274 = mul_275 = None
        squeeze_119: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
        mul_276: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_119, 1.0006381620931717);  squeeze_119 = None
        mul_277: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_276, 0.1);  mul_276 = None
        mul_278: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_239, 0.9)
        add_209: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
        unsqueeze_156: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_240, -1)
        unsqueeze_157: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
        mul_279: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_273, unsqueeze_157);  mul_273 = unsqueeze_157 = None
        unsqueeze_158: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_241, -1);  primals_241 = None
        unsqueeze_159: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
        add_210: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_279, unsqueeze_159);  mul_279 = unsqueeze_159 = None
        convert_element_type_120: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_210, torch.bfloat16);  add_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_211: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_120, relu_33);  convert_element_type_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_36: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(add_211);  add_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convert_element_type_121: "bf16[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_242, torch.bfloat16);  primals_242 = None
        convolution_40: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_36, convert_element_type_121, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_212: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_243, 1)
        convert_element_type_122: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_40, torch.float32)
        var_mean_40 = torch.ops.aten.var_mean.correction(convert_element_type_122, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_122 = None
        getitem_82: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_40[0]
        getitem_83: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_40[1];  var_mean_40 = None
        add_213: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_82, 1e-05)
        rsqrt_40: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
        sub_40: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_40, getitem_83)
        mul_280: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
        squeeze_120: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
        squeeze_121: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
        mul_281: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1)
        mul_282: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_244, 0.9)
        add_214: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_281, mul_282);  mul_281 = mul_282 = None
        squeeze_122: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
        mul_283: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_122, 1.0006381620931717);  squeeze_122 = None
        mul_284: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_283, 0.1);  mul_283 = None
        mul_285: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_245, 0.9)
        add_215: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
        unsqueeze_160: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_246, -1)
        unsqueeze_161: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
        mul_286: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_161);  mul_280 = unsqueeze_161 = None
        unsqueeze_162: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_247, -1);  primals_247 = None
        unsqueeze_163: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
        add_216: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_286, unsqueeze_163);  mul_286 = unsqueeze_163 = None
        convert_element_type_123: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_216, torch.bfloat16);  add_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_37: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_123);  convert_element_type_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convert_element_type_124: "bf16[512, 16, 3, 3][144, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_248, torch.bfloat16);  primals_248 = None
        convolution_41: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_37, convert_element_type_124, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_217: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_249, 1)
        convert_element_type_125: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_41, torch.float32)
        var_mean_41 = torch.ops.aten.var_mean.correction(convert_element_type_125, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_125 = None
        getitem_84: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_41[0]
        getitem_85: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_41[1];  var_mean_41 = None
        add_218: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_84, 1e-05)
        rsqrt_41: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
        sub_41: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_41, getitem_85)
        mul_287: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
        squeeze_123: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
        squeeze_124: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
        mul_288: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
        mul_289: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_250, 0.9)
        add_219: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_288, mul_289);  mul_288 = mul_289 = None
        squeeze_125: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
        mul_290: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_125, 1.0006381620931717);  squeeze_125 = None
        mul_291: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_290, 0.1);  mul_290 = None
        mul_292: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_251, 0.9)
        add_220: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_291, mul_292);  mul_291 = mul_292 = None
        unsqueeze_164: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_252, -1)
        unsqueeze_165: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
        mul_293: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_287, unsqueeze_165);  mul_287 = unsqueeze_165 = None
        unsqueeze_166: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_253, -1);  primals_253 = None
        unsqueeze_167: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
        add_221: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_293, unsqueeze_167);  mul_293 = unsqueeze_167 = None
        convert_element_type_126: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_221, torch.bfloat16);  add_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_38: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_126);  convert_element_type_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convert_element_type_127: "bf16[1024, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_254, torch.bfloat16);  primals_254 = None
        convolution_42: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_38, convert_element_type_127, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_222: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_255, 1)
        convert_element_type_128: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_42, torch.float32)
        var_mean_42 = torch.ops.aten.var_mean.correction(convert_element_type_128, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_128 = None
        getitem_86: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_42[0]
        getitem_87: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_42[1];  var_mean_42 = None
        add_223: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_86, 1e-05)
        rsqrt_42: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_223);  add_223 = None
        sub_42: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_42, getitem_87)
        mul_294: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
        squeeze_126: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
        squeeze_127: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
        mul_295: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1)
        mul_296: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_256, 0.9)
        add_224: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_295, mul_296);  mul_295 = mul_296 = None
        squeeze_128: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
        mul_297: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_128, 1.0006381620931717);  squeeze_128 = None
        mul_298: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_297, 0.1);  mul_297 = None
        mul_299: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_257, 0.9)
        add_225: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None
        unsqueeze_168: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_258, -1)
        unsqueeze_169: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
        mul_300: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_294, unsqueeze_169);  mul_294 = unsqueeze_169 = None
        unsqueeze_170: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_259, -1);  primals_259 = None
        unsqueeze_171: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
        add_226: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_300, unsqueeze_171);  mul_300 = unsqueeze_171 = None
        convert_element_type_129: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_226, torch.bfloat16);  add_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_227: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_129, relu_36);  convert_element_type_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_39: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(add_227);  add_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convert_element_type_130: "bf16[1024, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_260, torch.bfloat16);  primals_260 = None
        convolution_43: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_39, convert_element_type_130, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_228: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_261, 1)
        convert_element_type_131: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_43, torch.float32)
        var_mean_43 = torch.ops.aten.var_mean.correction(convert_element_type_131, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_131 = None
        getitem_88: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_43[0]
        getitem_89: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_43[1];  var_mean_43 = None
        add_229: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_88, 1e-05)
        rsqrt_43: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
        sub_43: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_43, getitem_89)
        mul_301: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
        squeeze_129: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
        squeeze_130: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
        mul_302: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1)
        mul_303: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_262, 0.9)
        add_230: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_302, mul_303);  mul_302 = mul_303 = None
        squeeze_131: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
        mul_304: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_131, 1.0006381620931717);  squeeze_131 = None
        mul_305: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_304, 0.1);  mul_304 = None
        mul_306: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_263, 0.9)
        add_231: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_305, mul_306);  mul_305 = mul_306 = None
        unsqueeze_172: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_264, -1)
        unsqueeze_173: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
        mul_307: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_173);  mul_301 = unsqueeze_173 = None
        unsqueeze_174: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_265, -1);  primals_265 = None
        unsqueeze_175: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
        add_232: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_307, unsqueeze_175);  mul_307 = unsqueeze_175 = None
        convert_element_type_132: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_232, torch.bfloat16);  add_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_40: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_132);  convert_element_type_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convert_element_type_133: "bf16[1024, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_266, torch.bfloat16);  primals_266 = None
        convolution_44: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_40, convert_element_type_133, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_233: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_267, 1)
        convert_element_type_134: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_44, torch.float32)
        var_mean_44 = torch.ops.aten.var_mean.correction(convert_element_type_134, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_134 = None
        getitem_90: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_44[0]
        getitem_91: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_44[1];  var_mean_44 = None
        add_234: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_90, 1e-05)
        rsqrt_44: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_234);  add_234 = None
        sub_44: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_44, getitem_91)
        mul_308: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
        squeeze_132: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
        squeeze_133: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
        mul_309: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_132, 0.1)
        mul_310: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_268, 0.9)
        add_235: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_309, mul_310);  mul_309 = mul_310 = None
        squeeze_134: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
        mul_311: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_134, 1.0025575447570332);  squeeze_134 = None
        mul_312: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_311, 0.1);  mul_311 = None
        mul_313: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_269, 0.9)
        add_236: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_312, mul_313);  mul_312 = mul_313 = None
        unsqueeze_176: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_270, -1)
        unsqueeze_177: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
        mul_314: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_308, unsqueeze_177);  mul_308 = unsqueeze_177 = None
        unsqueeze_178: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_271, -1);  primals_271 = None
        unsqueeze_179: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
        add_237: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_314, unsqueeze_179);  mul_314 = unsqueeze_179 = None
        convert_element_type_135: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_237, torch.bfloat16);  add_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_41: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_135);  convert_element_type_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convert_element_type_136: "bf16[2048, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_272, torch.bfloat16);  primals_272 = None
        convolution_45: "bf16[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_41, convert_element_type_136, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_238: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_273, 1)
        convert_element_type_137: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_45, torch.float32)
        var_mean_45 = torch.ops.aten.var_mean.correction(convert_element_type_137, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_137 = None
        getitem_92: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = var_mean_45[0]
        getitem_93: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = var_mean_45[1];  var_mean_45 = None
        add_239: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_92, 1e-05)
        rsqrt_45: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_239);  add_239 = None
        sub_45: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_45, getitem_93)
        mul_315: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
        squeeze_135: "f32[2048][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3]);  getitem_93 = None
        squeeze_136: "f32[2048][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
        mul_316: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_135, 0.1)
        mul_317: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_274, 0.9)
        add_240: "f32[2048][1]cuda:0" = torch.ops.aten.add.Tensor(mul_316, mul_317);  mul_316 = mul_317 = None
        squeeze_137: "f32[2048][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_92, [0, 2, 3]);  getitem_92 = None
        mul_318: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_137, 1.0025575447570332);  squeeze_137 = None
        mul_319: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_318, 0.1);  mul_318 = None
        mul_320: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_275, 0.9)
        add_241: "f32[2048][1]cuda:0" = torch.ops.aten.add.Tensor(mul_319, mul_320);  mul_319 = mul_320 = None
        unsqueeze_180: "f32[2048, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_276, -1)
        unsqueeze_181: "f32[2048, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
        mul_321: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_315, unsqueeze_181);  mul_315 = unsqueeze_181 = None
        unsqueeze_182: "f32[2048, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_277, -1);  primals_277 = None
        unsqueeze_183: "f32[2048, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
        add_242: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_321, unsqueeze_183);  mul_321 = unsqueeze_183 = None
        convert_element_type_138: "bf16[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_242, torch.bfloat16);  add_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        convert_element_type_139: "bf16[2048, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_278, torch.bfloat16);  primals_278 = None
        convolution_46: "bf16[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_39, convert_element_type_139, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        add_243: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_279, 1)
        convert_element_type_140: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_46, torch.float32)
        var_mean_46 = torch.ops.aten.var_mean.correction(convert_element_type_140, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_140 = None
        getitem_94: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = var_mean_46[0]
        getitem_95: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = var_mean_46[1];  var_mean_46 = None
        add_244: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_94, 1e-05)
        rsqrt_46: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_244);  add_244 = None
        sub_46: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_46, getitem_95)
        mul_322: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
        squeeze_138: "f32[2048][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
        squeeze_139: "f32[2048][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
        mul_323: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_138, 0.1)
        mul_324: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_280, 0.9)
        add_245: "f32[2048][1]cuda:0" = torch.ops.aten.add.Tensor(mul_323, mul_324);  mul_323 = mul_324 = None
        squeeze_140: "f32[2048][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_94, [0, 2, 3]);  getitem_94 = None
        mul_325: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_140, 1.0025575447570332);  squeeze_140 = None
        mul_326: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_325, 0.1);  mul_325 = None
        mul_327: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_281, 0.9)
        add_246: "f32[2048][1]cuda:0" = torch.ops.aten.add.Tensor(mul_326, mul_327);  mul_326 = mul_327 = None
        unsqueeze_184: "f32[2048, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_282, -1)
        unsqueeze_185: "f32[2048, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
        mul_328: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_322, unsqueeze_185);  mul_322 = unsqueeze_185 = None
        unsqueeze_186: "f32[2048, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_283, -1);  primals_283 = None
        unsqueeze_187: "f32[2048, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
        add_247: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_328, unsqueeze_187);  mul_328 = unsqueeze_187 = None
        convert_element_type_141: "bf16[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_247, torch.bfloat16);  add_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_248: "bf16[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_138, convert_element_type_141);  convert_element_type_138 = convert_element_type_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_42: "bf16[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(add_248);  add_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convert_element_type_142: "bf16[1024, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_284, torch.bfloat16);  primals_284 = None
        convolution_47: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_42, convert_element_type_142, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_249: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_285, 1)
        convert_element_type_143: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_47, torch.float32)
        var_mean_47 = torch.ops.aten.var_mean.correction(convert_element_type_143, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_143 = None
        getitem_96: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_47[0]
        getitem_97: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_47[1];  var_mean_47 = None
        add_250: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_96, 1e-05)
        rsqrt_47: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_250);  add_250 = None
        sub_47: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_47, getitem_97)
        mul_329: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
        squeeze_141: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
        squeeze_142: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
        mul_330: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_141, 0.1)
        mul_331: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_286, 0.9)
        add_251: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_330, mul_331);  mul_330 = mul_331 = None
        squeeze_143: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
        mul_332: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_143, 1.0025575447570332);  squeeze_143 = None
        mul_333: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_332, 0.1);  mul_332 = None
        mul_334: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_287, 0.9)
        add_252: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_333, mul_334);  mul_333 = mul_334 = None
        unsqueeze_188: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_288, -1)
        unsqueeze_189: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
        mul_335: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_329, unsqueeze_189);  mul_329 = unsqueeze_189 = None
        unsqueeze_190: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_289, -1);  primals_289 = None
        unsqueeze_191: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
        add_253: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_335, unsqueeze_191);  mul_335 = unsqueeze_191 = None
        convert_element_type_144: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_253, torch.bfloat16);  add_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_43: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_144);  convert_element_type_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convert_element_type_145: "bf16[1024, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_290, torch.bfloat16);  primals_290 = None
        convolution_48: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_43, convert_element_type_145, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_254: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_291, 1)
        convert_element_type_146: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_48, torch.float32)
        var_mean_48 = torch.ops.aten.var_mean.correction(convert_element_type_146, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_146 = None
        getitem_98: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_48[0]
        getitem_99: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_48[1];  var_mean_48 = None
        add_255: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_98, 1e-05)
        rsqrt_48: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_255);  add_255 = None
        sub_48: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_48, getitem_99)
        mul_336: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
        squeeze_144: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
        squeeze_145: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
        mul_337: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_144, 0.1)
        mul_338: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_292, 0.9)
        add_256: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_337, mul_338);  mul_337 = mul_338 = None
        squeeze_146: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_98, [0, 2, 3]);  getitem_98 = None
        mul_339: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_146, 1.0025575447570332);  squeeze_146 = None
        mul_340: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_339, 0.1);  mul_339 = None
        mul_341: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_293, 0.9)
        add_257: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_340, mul_341);  mul_340 = mul_341 = None
        unsqueeze_192: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_294, -1)
        unsqueeze_193: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
        mul_342: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_336, unsqueeze_193);  mul_336 = unsqueeze_193 = None
        unsqueeze_194: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_295, -1);  primals_295 = None
        unsqueeze_195: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
        add_258: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_342, unsqueeze_195);  mul_342 = unsqueeze_195 = None
        convert_element_type_147: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_258, torch.bfloat16);  add_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_44: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_147);  convert_element_type_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convert_element_type_148: "bf16[2048, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_296, torch.bfloat16);  primals_296 = None
        convolution_49: "bf16[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_44, convert_element_type_148, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_259: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_297, 1)
        convert_element_type_149: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_49, torch.float32)
        var_mean_49 = torch.ops.aten.var_mean.correction(convert_element_type_149, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_149 = None
        getitem_100: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = var_mean_49[0]
        getitem_101: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = var_mean_49[1];  var_mean_49 = None
        add_260: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_100, 1e-05)
        rsqrt_49: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_260);  add_260 = None
        sub_49: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_49, getitem_101)
        mul_343: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
        squeeze_147: "f32[2048][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2, 3]);  getitem_101 = None
        squeeze_148: "f32[2048][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
        mul_344: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_147, 0.1)
        mul_345: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_298, 0.9)
        add_261: "f32[2048][1]cuda:0" = torch.ops.aten.add.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
        squeeze_149: "f32[2048][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_100, [0, 2, 3]);  getitem_100 = None
        mul_346: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_149, 1.0025575447570332);  squeeze_149 = None
        mul_347: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_346, 0.1);  mul_346 = None
        mul_348: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_299, 0.9)
        add_262: "f32[2048][1]cuda:0" = torch.ops.aten.add.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
        unsqueeze_196: "f32[2048, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_300, -1)
        unsqueeze_197: "f32[2048, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
        mul_349: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_197);  mul_343 = unsqueeze_197 = None
        unsqueeze_198: "f32[2048, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_301, -1);  primals_301 = None
        unsqueeze_199: "f32[2048, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
        add_263: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_349, unsqueeze_199);  mul_349 = unsqueeze_199 = None
        convert_element_type_150: "bf16[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_263, torch.bfloat16);  add_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_264: "bf16[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_150, relu_42);  convert_element_type_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_45: "bf16[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(add_264);  add_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convert_element_type_151: "bf16[1024, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_302, torch.bfloat16);  primals_302 = None
        convolution_50: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_45, convert_element_type_151, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_265: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_303, 1)
        convert_element_type_152: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_50, torch.float32)
        var_mean_50 = torch.ops.aten.var_mean.correction(convert_element_type_152, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_152 = None
        getitem_102: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_50[0]
        getitem_103: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_50[1];  var_mean_50 = None
        add_266: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_102, 1e-05)
        rsqrt_50: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_266);  add_266 = None
        sub_50: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_50, getitem_103)
        mul_350: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
        squeeze_150: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
        squeeze_151: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
        mul_351: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_150, 0.1)
        mul_352: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_304, 0.9)
        add_267: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_351, mul_352);  mul_351 = mul_352 = None
        squeeze_152: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_102, [0, 2, 3]);  getitem_102 = None
        mul_353: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_152, 1.0025575447570332);  squeeze_152 = None
        mul_354: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_353, 0.1);  mul_353 = None
        mul_355: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_305, 0.9)
        add_268: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_354, mul_355);  mul_354 = mul_355 = None
        unsqueeze_200: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_306, -1)
        unsqueeze_201: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
        mul_356: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_350, unsqueeze_201);  mul_350 = unsqueeze_201 = None
        unsqueeze_202: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_307, -1);  primals_307 = None
        unsqueeze_203: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
        add_269: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_356, unsqueeze_203);  mul_356 = unsqueeze_203 = None
        convert_element_type_153: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_269, torch.bfloat16);  add_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_46: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_153);  convert_element_type_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convert_element_type_154: "bf16[1024, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_308, torch.bfloat16);  primals_308 = None
        convolution_51: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_46, convert_element_type_154, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_270: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_309, 1)
        convert_element_type_155: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_51, torch.float32)
        var_mean_51 = torch.ops.aten.var_mean.correction(convert_element_type_155, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_155 = None
        getitem_104: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_51[0]
        getitem_105: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_51[1];  var_mean_51 = None
        add_271: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_104, 1e-05)
        rsqrt_51: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_271);  add_271 = None
        sub_51: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_51, getitem_105)
        mul_357: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
        squeeze_153: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_105, [0, 2, 3]);  getitem_105 = None
        squeeze_154: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
        mul_358: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_153, 0.1)
        mul_359: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_310, 0.9)
        add_272: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_358, mul_359);  mul_358 = mul_359 = None
        squeeze_155: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_104, [0, 2, 3]);  getitem_104 = None
        mul_360: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_155, 1.0025575447570332);  squeeze_155 = None
        mul_361: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_360, 0.1);  mul_360 = None
        mul_362: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_311, 0.9)
        add_273: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_361, mul_362);  mul_361 = mul_362 = None
        unsqueeze_204: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_312, -1)
        unsqueeze_205: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
        mul_363: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_357, unsqueeze_205);  mul_357 = unsqueeze_205 = None
        unsqueeze_206: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_313, -1);  primals_313 = None
        unsqueeze_207: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
        add_274: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_363, unsqueeze_207);  mul_363 = unsqueeze_207 = None
        convert_element_type_156: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_274, torch.bfloat16);  add_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_47: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_156);  convert_element_type_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convert_element_type_157: "bf16[2048, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_314, torch.bfloat16);  primals_314 = None
        convolution_52: "bf16[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_47, convert_element_type_157, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_275: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_315, 1)
        convert_element_type_158: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_52, torch.float32)
        var_mean_52 = torch.ops.aten.var_mean.correction(convert_element_type_158, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_158 = None
        getitem_106: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = var_mean_52[0]
        getitem_107: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = var_mean_52[1];  var_mean_52 = None
        add_276: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_106, 1e-05)
        rsqrt_52: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_276);  add_276 = None
        sub_52: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_52, getitem_107)
        mul_364: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
        squeeze_156: "f32[2048][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
        squeeze_157: "f32[2048][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
        mul_365: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_156, 0.1)
        mul_366: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_316, 0.9)
        add_277: "f32[2048][1]cuda:0" = torch.ops.aten.add.Tensor(mul_365, mul_366);  mul_365 = mul_366 = None
        squeeze_158: "f32[2048][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_106, [0, 2, 3]);  getitem_106 = None
        mul_367: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_158, 1.0025575447570332);  squeeze_158 = None
        mul_368: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_367, 0.1);  mul_367 = None
        mul_369: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_317, 0.9)
        add_278: "f32[2048][1]cuda:0" = torch.ops.aten.add.Tensor(mul_368, mul_369);  mul_368 = mul_369 = None
        unsqueeze_208: "f32[2048, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_318, -1)
        unsqueeze_209: "f32[2048, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
        mul_370: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_364, unsqueeze_209);  mul_364 = unsqueeze_209 = None
        unsqueeze_210: "f32[2048, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_319, -1);  primals_319 = None
        unsqueeze_211: "f32[2048, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
        add_279: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_370, unsqueeze_211);  mul_370 = unsqueeze_211 = None
        convert_element_type_159: "bf16[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_279, torch.bfloat16);  add_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_280: "bf16[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_159, relu_45);  convert_element_type_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_48: "bf16[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(add_280);  add_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:278 in _forward_impl, code: x = self.avgpool(x)
        mean: "bf16[8, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(relu_48, [-1, -2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:279 in _forward_impl, code: x = torch.flatten(x, 1)
        view: "bf16[8, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(mean, [8, 2048]);  mean = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:280 in _forward_impl, code: x = self.fc(x)
        convert_element_type_160: "bf16[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_321, torch.bfloat16);  primals_321 = None
        convert_element_type_161: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_320, torch.bfloat16);  primals_320 = None
        permute: "bf16[2048, 1000][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_161, [1, 0]);  convert_element_type_161 = None
        addmm: "bf16[8, 1000][1000, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_160, view, permute);  convert_element_type_160 = None
        permute_1: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le: "b8[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_48, 0);  relu_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_212: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_156, 0);  squeeze_156 = None
        unsqueeze_213: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 2);  unsqueeze_212 = None
        unsqueeze_214: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 3);  unsqueeze_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_224: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
        unsqueeze_225: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 2);  unsqueeze_224 = None
        unsqueeze_226: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_225, 3);  unsqueeze_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_236: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_150, 0);  squeeze_150 = None
        unsqueeze_237: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 2);  unsqueeze_236 = None
        unsqueeze_238: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_237, 3);  unsqueeze_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_248: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_147, 0);  squeeze_147 = None
        unsqueeze_249: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 2);  unsqueeze_248 = None
        unsqueeze_250: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_249, 3);  unsqueeze_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_260: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
        unsqueeze_261: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 2);  unsqueeze_260 = None
        unsqueeze_262: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 3);  unsqueeze_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_272: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_141, 0);  squeeze_141 = None
        unsqueeze_273: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 2);  unsqueeze_272 = None
        unsqueeze_274: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 3);  unsqueeze_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        unsqueeze_284: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
        unsqueeze_285: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 2);  unsqueeze_284 = None
        unsqueeze_286: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 3);  unsqueeze_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_296: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
        unsqueeze_297: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 2);  unsqueeze_296 = None
        unsqueeze_298: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 3);  unsqueeze_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_308: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
        unsqueeze_309: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 2);  unsqueeze_308 = None
        unsqueeze_310: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 3);  unsqueeze_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_320: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
        unsqueeze_321: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 2);  unsqueeze_320 = None
        unsqueeze_322: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 3);  unsqueeze_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_332: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
        unsqueeze_333: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 2);  unsqueeze_332 = None
        unsqueeze_334: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 3);  unsqueeze_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_344: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
        unsqueeze_345: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 2);  unsqueeze_344 = None
        unsqueeze_346: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 3);  unsqueeze_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_356: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
        unsqueeze_357: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 2);  unsqueeze_356 = None
        unsqueeze_358: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 3);  unsqueeze_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_368: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
        unsqueeze_369: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 2);  unsqueeze_368 = None
        unsqueeze_370: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 3);  unsqueeze_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_380: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
        unsqueeze_381: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 2);  unsqueeze_380 = None
        unsqueeze_382: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 3);  unsqueeze_381 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_392: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
        unsqueeze_393: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 2);  unsqueeze_392 = None
        unsqueeze_394: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 3);  unsqueeze_393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_404: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
        unsqueeze_405: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 2);  unsqueeze_404 = None
        unsqueeze_406: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 3);  unsqueeze_405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_416: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
        unsqueeze_417: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 2);  unsqueeze_416 = None
        unsqueeze_418: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 3);  unsqueeze_417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_428: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
        unsqueeze_429: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 2);  unsqueeze_428 = None
        unsqueeze_430: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 3);  unsqueeze_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_440: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
        unsqueeze_441: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 2);  unsqueeze_440 = None
        unsqueeze_442: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 3);  unsqueeze_441 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_452: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
        unsqueeze_453: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 2);  unsqueeze_452 = None
        unsqueeze_454: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 3);  unsqueeze_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_464: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
        unsqueeze_465: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 2);  unsqueeze_464 = None
        unsqueeze_466: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 3);  unsqueeze_465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_476: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
        unsqueeze_477: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 2);  unsqueeze_476 = None
        unsqueeze_478: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 3);  unsqueeze_477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_488: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
        unsqueeze_489: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 2);  unsqueeze_488 = None
        unsqueeze_490: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 3);  unsqueeze_489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_500: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
        unsqueeze_501: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 2);  unsqueeze_500 = None
        unsqueeze_502: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 3);  unsqueeze_501 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        unsqueeze_512: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
        unsqueeze_513: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 2);  unsqueeze_512 = None
        unsqueeze_514: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_513, 3);  unsqueeze_513 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_524: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
        unsqueeze_525: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 2);  unsqueeze_524 = None
        unsqueeze_526: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_525, 3);  unsqueeze_525 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_536: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
        unsqueeze_537: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 2);  unsqueeze_536 = None
        unsqueeze_538: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_537, 3);  unsqueeze_537 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_548: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
        unsqueeze_549: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 2);  unsqueeze_548 = None
        unsqueeze_550: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_549, 3);  unsqueeze_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_560: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
        unsqueeze_561: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 2);  unsqueeze_560 = None
        unsqueeze_562: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_561, 3);  unsqueeze_561 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_572: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
        unsqueeze_573: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 2);  unsqueeze_572 = None
        unsqueeze_574: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_573, 3);  unsqueeze_573 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_584: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
        unsqueeze_585: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 2);  unsqueeze_584 = None
        unsqueeze_586: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_585, 3);  unsqueeze_585 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_596: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
        unsqueeze_597: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 2);  unsqueeze_596 = None
        unsqueeze_598: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_597, 3);  unsqueeze_597 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_608: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
        unsqueeze_609: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 2);  unsqueeze_608 = None
        unsqueeze_610: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_609, 3);  unsqueeze_609 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_620: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
        unsqueeze_621: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 2);  unsqueeze_620 = None
        unsqueeze_622: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_621, 3);  unsqueeze_621 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_632: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
        unsqueeze_633: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 2);  unsqueeze_632 = None
        unsqueeze_634: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_633, 3);  unsqueeze_633 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_644: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
        unsqueeze_645: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 2);  unsqueeze_644 = None
        unsqueeze_646: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_645, 3);  unsqueeze_645 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_656: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
        unsqueeze_657: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 2);  unsqueeze_656 = None
        unsqueeze_658: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_657, 3);  unsqueeze_657 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        unsqueeze_668: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
        unsqueeze_669: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 2);  unsqueeze_668 = None
        unsqueeze_670: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_669, 3);  unsqueeze_669 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_680: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
        unsqueeze_681: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 2);  unsqueeze_680 = None
        unsqueeze_682: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_681, 3);  unsqueeze_681 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_692: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
        unsqueeze_693: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 2);  unsqueeze_692 = None
        unsqueeze_694: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_693, 3);  unsqueeze_693 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_704: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
        unsqueeze_705: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 2);  unsqueeze_704 = None
        unsqueeze_706: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_705, 3);  unsqueeze_705 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_716: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
        unsqueeze_717: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 2);  unsqueeze_716 = None
        unsqueeze_718: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_717, 3);  unsqueeze_717 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_728: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
        unsqueeze_729: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 2);  unsqueeze_728 = None
        unsqueeze_730: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_729, 3);  unsqueeze_729 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_740: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
        unsqueeze_741: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 2);  unsqueeze_740 = None
        unsqueeze_742: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_741, 3);  unsqueeze_741 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_752: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
        unsqueeze_753: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 2);  unsqueeze_752 = None
        unsqueeze_754: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_753, 3);  unsqueeze_753 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_764: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
        unsqueeze_765: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 2);  unsqueeze_764 = None
        unsqueeze_766: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_765, 3);  unsqueeze_765 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_776: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
        unsqueeze_777: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 2);  unsqueeze_776 = None
        unsqueeze_778: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_777, 3);  unsqueeze_777 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        unsqueeze_788: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
        unsqueeze_789: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_788, 2);  unsqueeze_788 = None
        unsqueeze_790: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_789, 3);  unsqueeze_789 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_800: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
        unsqueeze_801: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 2);  unsqueeze_800 = None
        unsqueeze_802: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_801, 3);  unsqueeze_801 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_812: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
        unsqueeze_813: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_812, 2);  unsqueeze_812 = None
        unsqueeze_814: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_813, 3);  unsqueeze_813 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_824: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
        unsqueeze_825: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_824, 2);  unsqueeze_824 = None
        unsqueeze_826: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_825, 3);  unsqueeze_825 = None

        # No stacktrace found for following nodes
        copy_: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_3, add);  primals_3 = add = copy_ = None
        copy__1: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_4, add_2);  primals_4 = add_2 = copy__1 = None
        copy__2: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_5, add_3);  primals_5 = add_3 = copy__2 = None
        copy__3: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_9, add_5);  primals_9 = add_5 = copy__3 = None
        copy__4: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_10, add_7);  primals_10 = add_7 = copy__4 = None
        copy__5: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_11, add_8);  primals_11 = add_8 = copy__5 = None
        copy__6: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_15, add_10);  primals_15 = add_10 = copy__6 = None
        copy__7: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_16, add_12);  primals_16 = add_12 = copy__7 = None
        copy__8: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_17, add_13);  primals_17 = add_13 = copy__8 = None
        copy__9: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_21, add_15);  primals_21 = add_15 = copy__9 = None
        copy__10: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_22, add_17);  primals_22 = add_17 = copy__10 = None
        copy__11: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_23, add_18);  primals_23 = add_18 = copy__11 = None
        copy__12: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_27, add_20);  primals_27 = add_20 = copy__12 = None
        copy__13: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_28, add_22);  primals_28 = add_22 = copy__13 = None
        copy__14: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_29, add_23);  primals_29 = add_23 = copy__14 = None
        copy__15: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_33, add_26);  primals_33 = add_26 = copy__15 = None
        copy__16: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_34, add_28);  primals_34 = add_28 = copy__16 = None
        copy__17: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_35, add_29);  primals_35 = add_29 = copy__17 = None
        copy__18: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_39, add_31);  primals_39 = add_31 = copy__18 = None
        copy__19: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_40, add_33);  primals_40 = add_33 = copy__19 = None
        copy__20: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_41, add_34);  primals_41 = add_34 = copy__20 = None
        copy__21: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_45, add_36);  primals_45 = add_36 = copy__21 = None
        copy__22: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_46, add_38);  primals_46 = add_38 = copy__22 = None
        copy__23: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_47, add_39);  primals_47 = add_39 = copy__23 = None
        copy__24: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_51, add_42);  primals_51 = add_42 = copy__24 = None
        copy__25: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_52, add_44);  primals_52 = add_44 = copy__25 = None
        copy__26: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_53, add_45);  primals_53 = add_45 = copy__26 = None
        copy__27: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_57, add_47);  primals_57 = add_47 = copy__27 = None
        copy__28: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_58, add_49);  primals_58 = add_49 = copy__28 = None
        copy__29: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_59, add_50);  primals_59 = add_50 = copy__29 = None
        copy__30: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_63, add_52);  primals_63 = add_52 = copy__30 = None
        copy__31: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_64, add_54);  primals_64 = add_54 = copy__31 = None
        copy__32: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_65, add_55);  primals_65 = add_55 = copy__32 = None
        copy__33: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_69, add_58);  primals_69 = add_58 = copy__33 = None
        copy__34: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_70, add_60);  primals_70 = add_60 = copy__34 = None
        copy__35: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_71, add_61);  primals_71 = add_61 = copy__35 = None
        copy__36: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_75, add_63);  primals_75 = add_63 = copy__36 = None
        copy__37: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_76, add_65);  primals_76 = add_65 = copy__37 = None
        copy__38: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_77, add_66);  primals_77 = add_66 = copy__38 = None
        copy__39: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_81, add_68);  primals_81 = add_68 = copy__39 = None
        copy__40: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_82, add_70);  primals_82 = add_70 = copy__40 = None
        copy__41: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_83, add_71);  primals_83 = add_71 = copy__41 = None
        copy__42: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_87, add_73);  primals_87 = add_73 = copy__42 = None
        copy__43: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_88, add_75);  primals_88 = add_75 = copy__43 = None
        copy__44: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_89, add_76);  primals_89 = add_76 = copy__44 = None
        copy__45: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_93, add_79);  primals_93 = add_79 = copy__45 = None
        copy__46: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_94, add_81);  primals_94 = add_81 = copy__46 = None
        copy__47: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_95, add_82);  primals_95 = add_82 = copy__47 = None
        copy__48: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_99, add_84);  primals_99 = add_84 = copy__48 = None
        copy__49: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_100, add_86);  primals_100 = add_86 = copy__49 = None
        copy__50: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_101, add_87);  primals_101 = add_87 = copy__50 = None
        copy__51: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_105, add_89);  primals_105 = add_89 = copy__51 = None
        copy__52: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_106, add_91);  primals_106 = add_91 = copy__52 = None
        copy__53: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_107, add_92);  primals_107 = add_92 = copy__53 = None
        copy__54: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_111, add_95);  primals_111 = add_95 = copy__54 = None
        copy__55: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_112, add_97);  primals_112 = add_97 = copy__55 = None
        copy__56: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_113, add_98);  primals_113 = add_98 = copy__56 = None
        copy__57: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_117, add_100);  primals_117 = add_100 = copy__57 = None
        copy__58: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_118, add_102);  primals_118 = add_102 = copy__58 = None
        copy__59: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_119, add_103);  primals_119 = add_103 = copy__59 = None
        copy__60: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_123, add_105);  primals_123 = add_105 = copy__60 = None
        copy__61: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_124, add_107);  primals_124 = add_107 = copy__61 = None
        copy__62: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_125, add_108);  primals_125 = add_108 = copy__62 = None
        copy__63: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_129, add_111);  primals_129 = add_111 = copy__63 = None
        copy__64: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_130, add_113);  primals_130 = add_113 = copy__64 = None
        copy__65: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_131, add_114);  primals_131 = add_114 = copy__65 = None
        copy__66: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_135, add_116);  primals_135 = add_116 = copy__66 = None
        copy__67: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_136, add_118);  primals_136 = add_118 = copy__67 = None
        copy__68: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_137, add_119);  primals_137 = add_119 = copy__68 = None
        copy__69: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_141, add_121);  primals_141 = add_121 = copy__69 = None
        copy__70: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_142, add_123);  primals_142 = add_123 = copy__70 = None
        copy__71: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_143, add_124);  primals_143 = add_124 = copy__71 = None
        copy__72: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_147, add_127);  primals_147 = add_127 = copy__72 = None
        copy__73: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_148, add_129);  primals_148 = add_129 = copy__73 = None
        copy__74: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_149, add_130);  primals_149 = add_130 = copy__74 = None
        copy__75: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_153, add_132);  primals_153 = add_132 = copy__75 = None
        copy__76: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_154, add_134);  primals_154 = add_134 = copy__76 = None
        copy__77: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_155, add_135);  primals_155 = add_135 = copy__77 = None
        copy__78: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_159, add_137);  primals_159 = add_137 = copy__78 = None
        copy__79: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_160, add_139);  primals_160 = add_139 = copy__79 = None
        copy__80: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_161, add_140);  primals_161 = add_140 = copy__80 = None
        copy__81: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_165, add_142);  primals_165 = add_142 = copy__81 = None
        copy__82: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_166, add_144);  primals_166 = add_144 = copy__82 = None
        copy__83: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_167, add_145);  primals_167 = add_145 = copy__83 = None
        copy__84: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_171, add_148);  primals_171 = add_148 = copy__84 = None
        copy__85: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_172, add_150);  primals_172 = add_150 = copy__85 = None
        copy__86: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_173, add_151);  primals_173 = add_151 = copy__86 = None
        copy__87: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_177, add_153);  primals_177 = add_153 = copy__87 = None
        copy__88: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_178, add_155);  primals_178 = add_155 = copy__88 = None
        copy__89: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_179, add_156);  primals_179 = add_156 = copy__89 = None
        copy__90: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_183, add_158);  primals_183 = add_158 = copy__90 = None
        copy__91: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_184, add_160);  primals_184 = add_160 = copy__91 = None
        copy__92: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_185, add_161);  primals_185 = add_161 = copy__92 = None
        copy__93: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_189, add_164);  primals_189 = add_164 = copy__93 = None
        copy__94: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_190, add_166);  primals_190 = add_166 = copy__94 = None
        copy__95: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_191, add_167);  primals_191 = add_167 = copy__95 = None
        copy__96: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_195, add_169);  primals_195 = add_169 = copy__96 = None
        copy__97: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_196, add_171);  primals_196 = add_171 = copy__97 = None
        copy__98: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_197, add_172);  primals_197 = add_172 = copy__98 = None
        copy__99: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_201, add_174);  primals_201 = add_174 = copy__99 = None
        copy__100: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_202, add_176);  primals_202 = add_176 = copy__100 = None
        copy__101: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_203, add_177);  primals_203 = add_177 = copy__101 = None
        copy__102: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_207, add_180);  primals_207 = add_180 = copy__102 = None
        copy__103: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_208, add_182);  primals_208 = add_182 = copy__103 = None
        copy__104: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_209, add_183);  primals_209 = add_183 = copy__104 = None
        copy__105: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_213, add_185);  primals_213 = add_185 = copy__105 = None
        copy__106: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_214, add_187);  primals_214 = add_187 = copy__106 = None
        copy__107: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_215, add_188);  primals_215 = add_188 = copy__107 = None
        copy__108: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_219, add_190);  primals_219 = add_190 = copy__108 = None
        copy__109: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_220, add_192);  primals_220 = add_192 = copy__109 = None
        copy__110: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_221, add_193);  primals_221 = add_193 = copy__110 = None
        copy__111: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_225, add_196);  primals_225 = add_196 = copy__111 = None
        copy__112: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_226, add_198);  primals_226 = add_198 = copy__112 = None
        copy__113: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_227, add_199);  primals_227 = add_199 = copy__113 = None
        copy__114: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_231, add_201);  primals_231 = add_201 = copy__114 = None
        copy__115: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_232, add_203);  primals_232 = add_203 = copy__115 = None
        copy__116: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_233, add_204);  primals_233 = add_204 = copy__116 = None
        copy__117: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_237, add_206);  primals_237 = add_206 = copy__117 = None
        copy__118: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_238, add_208);  primals_238 = add_208 = copy__118 = None
        copy__119: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_239, add_209);  primals_239 = add_209 = copy__119 = None
        copy__120: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_243, add_212);  primals_243 = add_212 = copy__120 = None
        copy__121: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_244, add_214);  primals_244 = add_214 = copy__121 = None
        copy__122: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_245, add_215);  primals_245 = add_215 = copy__122 = None
        copy__123: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_249, add_217);  primals_249 = add_217 = copy__123 = None
        copy__124: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_250, add_219);  primals_250 = add_219 = copy__124 = None
        copy__125: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_251, add_220);  primals_251 = add_220 = copy__125 = None
        copy__126: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_255, add_222);  primals_255 = add_222 = copy__126 = None
        copy__127: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_256, add_224);  primals_256 = add_224 = copy__127 = None
        copy__128: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_257, add_225);  primals_257 = add_225 = copy__128 = None
        copy__129: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_261, add_228);  primals_261 = add_228 = copy__129 = None
        copy__130: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_262, add_230);  primals_262 = add_230 = copy__130 = None
        copy__131: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_263, add_231);  primals_263 = add_231 = copy__131 = None
        copy__132: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_267, add_233);  primals_267 = add_233 = copy__132 = None
        copy__133: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_268, add_235);  primals_268 = add_235 = copy__133 = None
        copy__134: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_269, add_236);  primals_269 = add_236 = copy__134 = None
        copy__135: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_273, add_238);  primals_273 = add_238 = copy__135 = None
        copy__136: "f32[2048][1]cuda:0" = torch.ops.aten.copy_.default(primals_274, add_240);  primals_274 = add_240 = copy__136 = None
        copy__137: "f32[2048][1]cuda:0" = torch.ops.aten.copy_.default(primals_275, add_241);  primals_275 = add_241 = copy__137 = None
        copy__138: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_279, add_243);  primals_279 = add_243 = copy__138 = None
        copy__139: "f32[2048][1]cuda:0" = torch.ops.aten.copy_.default(primals_280, add_245);  primals_280 = add_245 = copy__139 = None
        copy__140: "f32[2048][1]cuda:0" = torch.ops.aten.copy_.default(primals_281, add_246);  primals_281 = add_246 = copy__140 = None
        copy__141: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_285, add_249);  primals_285 = add_249 = copy__141 = None
        copy__142: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_286, add_251);  primals_286 = add_251 = copy__142 = None
        copy__143: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_287, add_252);  primals_287 = add_252 = copy__143 = None
        copy__144: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_291, add_254);  primals_291 = add_254 = copy__144 = None
        copy__145: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_292, add_256);  primals_292 = add_256 = copy__145 = None
        copy__146: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_293, add_257);  primals_293 = add_257 = copy__146 = None
        copy__147: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_297, add_259);  primals_297 = add_259 = copy__147 = None
        copy__148: "f32[2048][1]cuda:0" = torch.ops.aten.copy_.default(primals_298, add_261);  primals_298 = add_261 = copy__148 = None
        copy__149: "f32[2048][1]cuda:0" = torch.ops.aten.copy_.default(primals_299, add_262);  primals_299 = add_262 = copy__149 = None
        copy__150: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_303, add_265);  primals_303 = add_265 = copy__150 = None
        copy__151: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_304, add_267);  primals_304 = add_267 = copy__151 = None
        copy__152: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_305, add_268);  primals_305 = add_268 = copy__152 = None
        copy__153: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_309, add_270);  primals_309 = add_270 = copy__153 = None
        copy__154: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_310, add_272);  primals_310 = add_272 = copy__154 = None
        copy__155: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_311, add_273);  primals_311 = add_273 = copy__155 = None
        copy__156: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_315, add_275);  primals_315 = add_275 = copy__156 = None
        copy__157: "f32[2048][1]cuda:0" = torch.ops.aten.copy_.default(primals_316, add_277);  primals_316 = add_277 = copy__157 = None
        copy__158: "f32[2048][1]cuda:0" = torch.ops.aten.copy_.default(primals_317, add_278);  primals_317 = add_278 = copy__158 = None
        return (addmm, primals_6, primals_7, primals_12, primals_18, primals_24, primals_30, primals_36, primals_42, primals_48, primals_54, primals_60, primals_66, primals_72, primals_78, primals_84, primals_90, primals_96, primals_102, primals_108, primals_114, primals_120, primals_126, primals_132, primals_138, primals_144, primals_150, primals_156, primals_162, primals_168, primals_174, primals_180, primals_186, primals_192, primals_198, primals_204, primals_210, primals_216, primals_222, primals_228, primals_234, primals_240, primals_246, primals_252, primals_258, primals_264, primals_270, primals_276, primals_282, primals_288, primals_294, primals_300, primals_306, primals_312, primals_318, convert_element_type, convert_element_type_1, convolution, getitem_1, rsqrt, getitem_2, getitem_3, convert_element_type_4, convolution_1, squeeze_4, relu_1, convert_element_type_7, convolution_2, squeeze_7, relu_2, convert_element_type_10, convolution_3, squeeze_10, convert_element_type_13, convolution_4, squeeze_13, relu_3, convert_element_type_16, convolution_5, squeeze_16, relu_4, convert_element_type_19, convolution_6, squeeze_19, relu_5, convert_element_type_22, convolution_7, squeeze_22, relu_6, convert_element_type_25, convolution_8, squeeze_25, relu_7, convert_element_type_28, convolution_9, squeeze_28, relu_8, convert_element_type_31, convolution_10, squeeze_31, relu_9, convert_element_type_34, convolution_11, squeeze_34, relu_10, convert_element_type_37, convolution_12, squeeze_37, relu_11, convert_element_type_40, convolution_13, squeeze_40, convert_element_type_43, convolution_14, squeeze_43, relu_12, convert_element_type_46, convolution_15, squeeze_46, relu_13, convert_element_type_49, convolution_16, squeeze_49, relu_14, convert_element_type_52, convolution_17, squeeze_52, relu_15, convert_element_type_55, convolution_18, squeeze_55, relu_16, convert_element_type_58, convolution_19, squeeze_58, relu_17, convert_element_type_61, convolution_20, squeeze_61, relu_18, convert_element_type_64, convolution_21, squeeze_64, relu_19, convert_element_type_67, convolution_22, squeeze_67, relu_20, convert_element_type_70, convolution_23, squeeze_70, relu_21, convert_element_type_73, convolution_24, squeeze_73, relu_22, convert_element_type_76, convolution_25, squeeze_76, relu_23, convert_element_type_79, convolution_26, squeeze_79, convert_element_type_82, convolution_27, squeeze_82, relu_24, convert_element_type_85, convolution_28, squeeze_85, relu_25, convert_element_type_88, convolution_29, squeeze_88, relu_26, convert_element_type_91, convolution_30, squeeze_91, relu_27, convert_element_type_94, convolution_31, squeeze_94, relu_28, convert_element_type_97, convolution_32, squeeze_97, relu_29, convert_element_type_100, convolution_33, squeeze_100, relu_30, convert_element_type_103, convolution_34, squeeze_103, relu_31, convert_element_type_106, convolution_35, squeeze_106, relu_32, convert_element_type_109, convolution_36, squeeze_109, relu_33, convert_element_type_112, convolution_37, squeeze_112, relu_34, convert_element_type_115, convolution_38, squeeze_115, relu_35, convert_element_type_118, convolution_39, squeeze_118, relu_36, convert_element_type_121, convolution_40, squeeze_121, relu_37, convert_element_type_124, convolution_41, squeeze_124, relu_38, convert_element_type_127, convolution_42, squeeze_127, relu_39, convert_element_type_130, convolution_43, squeeze_130, relu_40, convert_element_type_133, convolution_44, squeeze_133, relu_41, convert_element_type_136, convolution_45, squeeze_136, convert_element_type_139, convolution_46, squeeze_139, relu_42, convert_element_type_142, convolution_47, squeeze_142, relu_43, convert_element_type_145, convolution_48, squeeze_145, relu_44, convert_element_type_148, convolution_49, squeeze_148, relu_45, convert_element_type_151, convolution_50, squeeze_151, relu_46, convert_element_type_154, convolution_51, squeeze_154, relu_47, convert_element_type_157, convolution_52, squeeze_157, view, permute_1, le, unsqueeze_214, unsqueeze_226, unsqueeze_238, unsqueeze_250, unsqueeze_262, unsqueeze_274, unsqueeze_286, unsqueeze_298, unsqueeze_310, unsqueeze_322, unsqueeze_334, unsqueeze_346, unsqueeze_358, unsqueeze_370, unsqueeze_382, unsqueeze_394, unsqueeze_406, unsqueeze_418, unsqueeze_430, unsqueeze_442, unsqueeze_454, unsqueeze_466, unsqueeze_478, unsqueeze_490, unsqueeze_502, unsqueeze_514, unsqueeze_526, unsqueeze_538, unsqueeze_550, unsqueeze_562, unsqueeze_574, unsqueeze_586, unsqueeze_598, unsqueeze_610, unsqueeze_622, unsqueeze_634, unsqueeze_646, unsqueeze_658, unsqueeze_670, unsqueeze_682, unsqueeze_694, unsqueeze_706, unsqueeze_718, unsqueeze_730, unsqueeze_742, unsqueeze_754, unsqueeze_766, unsqueeze_778, unsqueeze_790, unsqueeze_802, unsqueeze_814, unsqueeze_826)
