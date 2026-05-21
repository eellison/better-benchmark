class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[32, 3, 224, 224][150528, 50176, 224, 1]cuda:0", primals_2: "f32[768, 3, 32, 32][3072, 1024, 32, 1]cuda:0", primals_3: "f32[768][1]cuda:0", primals_4: "f32[50, 768][768, 1]cuda:0", primals_5: "f32[768][1]cuda:0", primals_6: "f32[768][1]cuda:0", primals_7: "f32[768][1]cuda:0", primals_8: "f32[768][1]cuda:0", primals_9: "f32[2304][1]cuda:0", primals_10: "f32[2304, 768][768, 1]cuda:0", primals_11: "f32[768, 768][768, 1]cuda:0", primals_12: "f32[768][1]cuda:0", primals_13: "f32[768][1]cuda:0", primals_14: "f32[768][1]cuda:0", primals_15: "f32[3072, 768][768, 1]cuda:0", primals_16: "f32[3072][1]cuda:0", primals_17: "f32[768, 3072][3072, 1]cuda:0", primals_18: "f32[768][1]cuda:0", primals_19: "f32[768][1]cuda:0", primals_20: "f32[768][1]cuda:0", primals_21: "f32[2304][1]cuda:0", primals_22: "f32[2304, 768][768, 1]cuda:0", primals_23: "f32[768, 768][768, 1]cuda:0", primals_24: "f32[768][1]cuda:0", primals_25: "f32[768][1]cuda:0", primals_26: "f32[768][1]cuda:0", primals_27: "f32[3072, 768][768, 1]cuda:0", primals_28: "f32[3072][1]cuda:0", primals_29: "f32[768, 3072][3072, 1]cuda:0", primals_30: "f32[768][1]cuda:0", primals_31: "f32[768][1]cuda:0", primals_32: "f32[768][1]cuda:0", primals_33: "f32[2304][1]cuda:0", primals_34: "f32[2304, 768][768, 1]cuda:0", primals_35: "f32[768, 768][768, 1]cuda:0", primals_36: "f32[768][1]cuda:0", primals_37: "f32[768][1]cuda:0", primals_38: "f32[768][1]cuda:0", primals_39: "f32[3072, 768][768, 1]cuda:0", primals_40: "f32[3072][1]cuda:0", primals_41: "f32[768, 3072][3072, 1]cuda:0", primals_42: "f32[768][1]cuda:0", primals_43: "f32[768][1]cuda:0", primals_44: "f32[768][1]cuda:0", primals_45: "f32[2304][1]cuda:0", primals_46: "f32[2304, 768][768, 1]cuda:0", primals_47: "f32[768, 768][768, 1]cuda:0", primals_48: "f32[768][1]cuda:0", primals_49: "f32[768][1]cuda:0", primals_50: "f32[768][1]cuda:0", primals_51: "f32[3072, 768][768, 1]cuda:0", primals_52: "f32[3072][1]cuda:0", primals_53: "f32[768, 3072][3072, 1]cuda:0", primals_54: "f32[768][1]cuda:0", primals_55: "f32[768][1]cuda:0", primals_56: "f32[768][1]cuda:0", primals_57: "f32[2304][1]cuda:0", primals_58: "f32[2304, 768][768, 1]cuda:0", primals_59: "f32[768, 768][768, 1]cuda:0", primals_60: "f32[768][1]cuda:0", primals_61: "f32[768][1]cuda:0", primals_62: "f32[768][1]cuda:0", primals_63: "f32[3072, 768][768, 1]cuda:0", primals_64: "f32[3072][1]cuda:0", primals_65: "f32[768, 3072][3072, 1]cuda:0", primals_66: "f32[768][1]cuda:0", primals_67: "f32[768][1]cuda:0", primals_68: "f32[768][1]cuda:0", primals_69: "f32[2304][1]cuda:0", primals_70: "f32[2304, 768][768, 1]cuda:0", primals_71: "f32[768, 768][768, 1]cuda:0", primals_72: "f32[768][1]cuda:0", primals_73: "f32[768][1]cuda:0", primals_74: "f32[768][1]cuda:0", primals_75: "f32[3072, 768][768, 1]cuda:0", primals_76: "f32[3072][1]cuda:0", primals_77: "f32[768, 3072][3072, 1]cuda:0", primals_78: "f32[768][1]cuda:0", primals_79: "f32[768][1]cuda:0", primals_80: "f32[768][1]cuda:0", primals_81: "f32[2304][1]cuda:0", primals_82: "f32[2304, 768][768, 1]cuda:0", primals_83: "f32[768, 768][768, 1]cuda:0", primals_84: "f32[768][1]cuda:0", primals_85: "f32[768][1]cuda:0", primals_86: "f32[768][1]cuda:0", primals_87: "f32[3072, 768][768, 1]cuda:0", primals_88: "f32[3072][1]cuda:0", primals_89: "f32[768, 3072][3072, 1]cuda:0", primals_90: "f32[768][1]cuda:0", primals_91: "f32[768][1]cuda:0", primals_92: "f32[768][1]cuda:0", primals_93: "f32[2304][1]cuda:0", primals_94: "f32[2304, 768][768, 1]cuda:0", primals_95: "f32[768, 768][768, 1]cuda:0", primals_96: "f32[768][1]cuda:0", primals_97: "f32[768][1]cuda:0", primals_98: "f32[768][1]cuda:0", primals_99: "f32[3072, 768][768, 1]cuda:0", primals_100: "f32[3072][1]cuda:0", primals_101: "f32[768, 3072][3072, 1]cuda:0", primals_102: "f32[768][1]cuda:0", primals_103: "f32[768][1]cuda:0", primals_104: "f32[768][1]cuda:0", primals_105: "f32[2304][1]cuda:0", primals_106: "f32[2304, 768][768, 1]cuda:0", primals_107: "f32[768, 768][768, 1]cuda:0", primals_108: "f32[768][1]cuda:0", primals_109: "f32[768][1]cuda:0", primals_110: "f32[768][1]cuda:0", primals_111: "f32[3072, 768][768, 1]cuda:0", primals_112: "f32[3072][1]cuda:0", primals_113: "f32[768, 3072][3072, 1]cuda:0", primals_114: "f32[768][1]cuda:0", primals_115: "f32[768][1]cuda:0", primals_116: "f32[768][1]cuda:0", primals_117: "f32[2304][1]cuda:0", primals_118: "f32[2304, 768][768, 1]cuda:0", primals_119: "f32[768, 768][768, 1]cuda:0", primals_120: "f32[768][1]cuda:0", primals_121: "f32[768][1]cuda:0", primals_122: "f32[768][1]cuda:0", primals_123: "f32[3072, 768][768, 1]cuda:0", primals_124: "f32[3072][1]cuda:0", primals_125: "f32[768, 3072][3072, 1]cuda:0", primals_126: "f32[768][1]cuda:0", primals_127: "f32[768][1]cuda:0", primals_128: "f32[768][1]cuda:0", primals_129: "f32[2304][1]cuda:0", primals_130: "f32[2304, 768][768, 1]cuda:0", primals_131: "f32[768, 768][768, 1]cuda:0", primals_132: "f32[768][1]cuda:0", primals_133: "f32[768][1]cuda:0", primals_134: "f32[768][1]cuda:0", primals_135: "f32[3072, 768][768, 1]cuda:0", primals_136: "f32[3072][1]cuda:0", primals_137: "f32[768, 3072][3072, 1]cuda:0", primals_138: "f32[768][1]cuda:0", primals_139: "f32[768][1]cuda:0", primals_140: "f32[768][1]cuda:0", primals_141: "f32[2304][1]cuda:0", primals_142: "f32[2304, 768][768, 1]cuda:0", primals_143: "f32[768, 768][768, 1]cuda:0", primals_144: "f32[768][1]cuda:0", primals_145: "f32[768][1]cuda:0", primals_146: "f32[768][1]cuda:0", primals_147: "f32[3072, 768][768, 1]cuda:0", primals_148: "f32[3072][1]cuda:0", primals_149: "f32[768, 3072][3072, 1]cuda:0", primals_150: "f32[768][1]cuda:0", primals_151: "f32[768][1]cuda:0", primals_152: "f32[768][1]cuda:0", primals_153: "f32[768, 512][512, 1]cuda:0", primals_154: "i64[32, 77][77, 1]cuda:0", primals_155: "f32[49408, 512][512, 1]cuda:0", primals_156: "f32[77, 512][512, 1]cuda:0", primals_157: "f32[77, 77][77, 1]cpu", primals_158: "f32[512][1]cuda:0", primals_159: "f32[512][1]cuda:0", primals_160: "f32[1536][1]cuda:0", primals_161: "f32[1536, 512][512, 1]cuda:0", primals_162: "f32[512, 512][512, 1]cuda:0", primals_163: "f32[512][1]cuda:0", primals_164: "f32[512][1]cuda:0", primals_165: "f32[512][1]cuda:0", primals_166: "f32[2048, 512][512, 1]cuda:0", primals_167: "f32[2048][1]cuda:0", primals_168: "f32[512, 2048][2048, 1]cuda:0", primals_169: "f32[512][1]cuda:0", primals_170: "f32[512][1]cuda:0", primals_171: "f32[512][1]cuda:0", primals_172: "f32[1536][1]cuda:0", primals_173: "f32[1536, 512][512, 1]cuda:0", primals_174: "f32[512, 512][512, 1]cuda:0", primals_175: "f32[512][1]cuda:0", primals_176: "f32[512][1]cuda:0", primals_177: "f32[512][1]cuda:0", primals_178: "f32[2048, 512][512, 1]cuda:0", primals_179: "f32[2048][1]cuda:0", primals_180: "f32[512, 2048][2048, 1]cuda:0", primals_181: "f32[512][1]cuda:0", primals_182: "f32[512][1]cuda:0", primals_183: "f32[512][1]cuda:0", primals_184: "f32[1536][1]cuda:0", primals_185: "f32[1536, 512][512, 1]cuda:0", primals_186: "f32[512, 512][512, 1]cuda:0", primals_187: "f32[512][1]cuda:0", primals_188: "f32[512][1]cuda:0", primals_189: "f32[512][1]cuda:0", primals_190: "f32[2048, 512][512, 1]cuda:0", primals_191: "f32[2048][1]cuda:0", primals_192: "f32[512, 2048][2048, 1]cuda:0", primals_193: "f32[512][1]cuda:0", primals_194: "f32[512][1]cuda:0", primals_195: "f32[512][1]cuda:0", primals_196: "f32[1536][1]cuda:0", primals_197: "f32[1536, 512][512, 1]cuda:0", primals_198: "f32[512, 512][512, 1]cuda:0", primals_199: "f32[512][1]cuda:0", primals_200: "f32[512][1]cuda:0", primals_201: "f32[512][1]cuda:0", primals_202: "f32[2048, 512][512, 1]cuda:0", primals_203: "f32[2048][1]cuda:0", primals_204: "f32[512, 2048][2048, 1]cuda:0", primals_205: "f32[512][1]cuda:0", primals_206: "f32[512][1]cuda:0", primals_207: "f32[512][1]cuda:0", primals_208: "f32[1536][1]cuda:0", primals_209: "f32[1536, 512][512, 1]cuda:0", primals_210: "f32[512, 512][512, 1]cuda:0", primals_211: "f32[512][1]cuda:0", primals_212: "f32[512][1]cuda:0", primals_213: "f32[512][1]cuda:0", primals_214: "f32[2048, 512][512, 1]cuda:0", primals_215: "f32[2048][1]cuda:0", primals_216: "f32[512, 2048][2048, 1]cuda:0", primals_217: "f32[512][1]cuda:0", primals_218: "f32[512][1]cuda:0", primals_219: "f32[512][1]cuda:0", primals_220: "f32[1536][1]cuda:0", primals_221: "f32[1536, 512][512, 1]cuda:0", primals_222: "f32[512, 512][512, 1]cuda:0", primals_223: "f32[512][1]cuda:0", primals_224: "f32[512][1]cuda:0", primals_225: "f32[512][1]cuda:0", primals_226: "f32[2048, 512][512, 1]cuda:0", primals_227: "f32[2048][1]cuda:0", primals_228: "f32[512, 2048][2048, 1]cuda:0", primals_229: "f32[512][1]cuda:0", primals_230: "f32[512][1]cuda:0", primals_231: "f32[512][1]cuda:0", primals_232: "f32[1536][1]cuda:0", primals_233: "f32[1536, 512][512, 1]cuda:0", primals_234: "f32[512, 512][512, 1]cuda:0", primals_235: "f32[512][1]cuda:0", primals_236: "f32[512][1]cuda:0", primals_237: "f32[512][1]cuda:0", primals_238: "f32[2048, 512][512, 1]cuda:0", primals_239: "f32[2048][1]cuda:0", primals_240: "f32[512, 2048][2048, 1]cuda:0", primals_241: "f32[512][1]cuda:0", primals_242: "f32[512][1]cuda:0", primals_243: "f32[512][1]cuda:0", primals_244: "f32[1536][1]cuda:0", primals_245: "f32[1536, 512][512, 1]cuda:0", primals_246: "f32[512, 512][512, 1]cuda:0", primals_247: "f32[512][1]cuda:0", primals_248: "f32[512][1]cuda:0", primals_249: "f32[512][1]cuda:0", primals_250: "f32[2048, 512][512, 1]cuda:0", primals_251: "f32[2048][1]cuda:0", primals_252: "f32[512, 2048][2048, 1]cuda:0", primals_253: "f32[512][1]cuda:0", primals_254: "f32[512][1]cuda:0", primals_255: "f32[512][1]cuda:0", primals_256: "f32[1536][1]cuda:0", primals_257: "f32[1536, 512][512, 1]cuda:0", primals_258: "f32[512, 512][512, 1]cuda:0", primals_259: "f32[512][1]cuda:0", primals_260: "f32[512][1]cuda:0", primals_261: "f32[512][1]cuda:0", primals_262: "f32[2048, 512][512, 1]cuda:0", primals_263: "f32[2048][1]cuda:0", primals_264: "f32[512, 2048][2048, 1]cuda:0", primals_265: "f32[512][1]cuda:0", primals_266: "f32[512][1]cuda:0", primals_267: "f32[512][1]cuda:0", primals_268: "f32[1536][1]cuda:0", primals_269: "f32[1536, 512][512, 1]cuda:0", primals_270: "f32[512, 512][512, 1]cuda:0", primals_271: "f32[512][1]cuda:0", primals_272: "f32[512][1]cuda:0", primals_273: "f32[512][1]cuda:0", primals_274: "f32[2048, 512][512, 1]cuda:0", primals_275: "f32[2048][1]cuda:0", primals_276: "f32[512, 2048][2048, 1]cuda:0", primals_277: "f32[512][1]cuda:0", primals_278: "f32[512][1]cuda:0", primals_279: "f32[512][1]cuda:0", primals_280: "f32[1536][1]cuda:0", primals_281: "f32[1536, 512][512, 1]cuda:0", primals_282: "f32[512, 512][512, 1]cuda:0", primals_283: "f32[512][1]cuda:0", primals_284: "f32[512][1]cuda:0", primals_285: "f32[512][1]cuda:0", primals_286: "f32[2048, 512][512, 1]cuda:0", primals_287: "f32[2048][1]cuda:0", primals_288: "f32[512, 2048][2048, 1]cuda:0", primals_289: "f32[512][1]cuda:0", primals_290: "f32[512][1]cuda:0", primals_291: "f32[512][1]cuda:0", primals_292: "f32[1536][1]cuda:0", primals_293: "f32[1536, 512][512, 1]cuda:0", primals_294: "f32[512, 512][512, 1]cuda:0", primals_295: "f32[512][1]cuda:0", primals_296: "f32[512][1]cuda:0", primals_297: "f32[512][1]cuda:0", primals_298: "f32[2048, 512][512, 1]cuda:0", primals_299: "f32[2048][1]cuda:0", primals_300: "f32[512, 2048][2048, 1]cuda:0", primals_301: "f32[512][1]cuda:0", primals_302: "f32[512][1]cuda:0", primals_303: "f32[512][1]cuda:0", primals_304: "f32[512, 512][512, 1]cuda:0"):
        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:91 in forward, code: x = self.conv(x)
        convert_element_type: "f16[768, 3, 32, 32][3072, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_2, torch.float16);  primals_2 = None
        convert_element_type_1: "f16[32, 3, 224, 224][150528, 50176, 224, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_1, torch.float16);  primals_1 = None
        convolution: "f16[32, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_1, convert_element_type, None, [32, 32], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:94 in forward, code: x = torch.flatten(x, start_dim=2)
        view: "f16[32, 768, 49][37632, 49, 1]cuda:0" = torch.ops.aten.reshape.default(convolution, [32, 768, 49]);  convolution = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:97 in forward, code: x = x.permute(0, 2, 1)
        permute: "f16[32, 49, 768][37632, 1, 49]cuda:0" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:100 in forward, code: self.cls_token_embedding.unsqueeze(0).expand(x.shape[0], -1, -1),
        unsqueeze: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_3, 0);  primals_3 = None
        expand: "f32[32, 1, 768][0, 768, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze, [32, -1, -1]);  unsqueeze = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:98 in forward, code: x = torch.cat(
        cat: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.cat.default([expand, permute], 1);  expand = permute = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:105 in forward, code: x = x + self.positional_embedding
        add: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(cat, primals_4)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/normalizations.py:18 in forward, code: output = nn.functional.layer_norm(
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        add_1: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add, getitem_1);  add = None
        mul: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, primals_5);  mul = None
        add_2: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, primals_6);  mul_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_2, [2], correction = 0, keepdim = True)
        getitem_2: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_1[0]
        getitem_3: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        add_3: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        sub_1: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_2, getitem_3)
        mul_2: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_3: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2, primals_7);  mul_2 = None
        add_4: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_3, primals_8);  mul_3 = primals_8 = None
        permute_1: "f32[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.aten.permute.default(add_4, [1, 0, 2]);  add_4 = None
        convert_element_type_2: "f16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_9, torch.float16);  primals_9 = None
        convert_element_type_3: "f16[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_10, torch.float16);  primals_10 = None
        convert_element_type_4: "f16[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_1, torch.float16);  permute_1 = None
        permute_2: "f16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_3, [1, 0]);  convert_element_type_3 = None
        clone: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_4, memory_format = torch.contiguous_format);  convert_element_type_4 = None
        view_1: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone, [1600, 768]);  clone = None
        mm: "f16[1600, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_1, permute_2)
        view_2: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [50, 32, 2304]);  mm = None
        add_5: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(view_2, convert_element_type_2);  view_2 = convert_element_type_2 = None
        view_3: "f16[50, 32, 3, 768][73728, 2304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(add_5, [50, 32, 3, 768]);  add_5 = None
        unsqueeze_1: "f16[1, 50, 32, 3, 768][3686400, 73728, 2304, 768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_3, 0);  view_3 = None
        permute_3: "f16[3, 50, 32, 1, 768][768, 73728, 2304, 3686400, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_1, [3, 1, 2, 0, 4]);  unsqueeze_1 = None
        squeeze: "f16[3, 50, 32, 768][768, 73728, 2304, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_3, -2);  permute_3 = None
        clone_1: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
        select: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_1, 0, 0)
        select_1: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_1, 0, 1)
        select_2: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_1, 0, 2);  clone_1 = None
        view_4: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select, [50, 384, 64]);  select = None
        permute_4: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_4, [1, 0, 2]);  view_4 = None
        view_5: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_1, [50, 384, 64]);  select_1 = None
        permute_5: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_5, [1, 0, 2]);  view_5 = None
        view_6: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_2, [50, 384, 64]);  select_2 = None
        permute_6: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_6, [1, 0, 2]);  view_6 = None
        view_7: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_4, [32, 12, 50, 64]);  permute_4 = None
        view_8: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_5, [32, 12, 50, 64]);  permute_5 = None
        view_9: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_6, [32, 12, 50, 64]);  permute_6 = None
        _scaled_dot_product_cudnn_attention = torch.ops.aten._scaled_dot_product_cudnn_attention.default(view_7, view_8, view_9, None, True)
        getitem_4: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention[0]
        getitem_5: "f32[32, 12, 50, 1][600, 50, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention[1]
        getitem_10: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention[6]
        getitem_11: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention[7];  _scaled_dot_product_cudnn_attention = None
        permute_7: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_4, [2, 0, 1, 3])
        view_10: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_7, [1600, 768]);  permute_7 = None
        convert_element_type_7: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_12, torch.float16);  primals_12 = None
        convert_element_type_8: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_11, torch.float16);  primals_11 = None
        permute_8: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_8, [1, 0]);  convert_element_type_8 = None
        addmm: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_7, view_10, permute_8);  convert_element_type_7 = view_10 = None
        view_11: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [50, 32, 768]);  addmm = None
        permute_9: "f16[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_11, [1, 0, 2]);  view_11 = None
        add_6: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_2, permute_9);  add_2 = permute_9 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_6, [2], correction = 0, keepdim = True)
        getitem_13: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_2[0]
        getitem_14: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        add_7: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_13, 1e-05);  getitem_13 = None
        rsqrt_2: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        sub_2: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_6, getitem_14);  getitem_14 = None
        mul_4: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        mul_5: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4, primals_13)
        add_8: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_5, primals_14);  mul_5 = primals_14 = None
        convert_element_type_12: "f16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_16, torch.float16);  primals_16 = None
        convert_element_type_13: "f16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_15, torch.float16);  primals_15 = None
        convert_element_type_14: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_8, torch.float16);  add_8 = None
        view_12: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_14, [1600, 768]);  convert_element_type_14 = None
        permute_10: "f16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_13, [1, 0]);  convert_element_type_13 = None
        addmm_1: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_12, view_12, permute_10);  convert_element_type_12 = None
        view_13: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [32, 50, 3072])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_6: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_13, 1.702)
        sigmoid: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_6);  mul_6 = None
        mul_7: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid, view_13);  sigmoid = view_13 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        convert_element_type_18: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_18, torch.float16);  primals_18 = None
        convert_element_type_19: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_17, torch.float16);  primals_17 = None
        view_14: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_7, [1600, 3072]);  mul_7 = None
        permute_11: "f16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_19, [1, 0]);  convert_element_type_19 = None
        addmm_2: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_18, view_14, permute_11);  convert_element_type_18 = None
        view_15: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [32, 50, 768]);  addmm_2 = None
        add_9: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_6, view_15);  add_6 = view_15 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
        getitem_15: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_3[0]
        getitem_16: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        add_10: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_15, 1e-05);  getitem_15 = None
        rsqrt_3: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_3: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_9, getitem_16);  getitem_16 = None
        mul_8: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        mul_9: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_8, primals_19)
        add_11: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_9, primals_20);  mul_9 = primals_20 = None
        permute_12: "f32[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.aten.permute.default(add_11, [1, 0, 2]);  add_11 = None
        convert_element_type_23: "f16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_21, torch.float16);  primals_21 = None
        convert_element_type_24: "f16[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_22, torch.float16);  primals_22 = None
        convert_element_type_25: "f16[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_12, torch.float16);  permute_12 = None
        permute_13: "f16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_24, [1, 0]);  convert_element_type_24 = None
        clone_5: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_25, memory_format = torch.contiguous_format);  convert_element_type_25 = None
        view_16: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_5, [1600, 768]);  clone_5 = None
        mm_1: "f16[1600, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_16, permute_13)
        view_17: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_1, [50, 32, 2304]);  mm_1 = None
        add_12: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(view_17, convert_element_type_23);  view_17 = convert_element_type_23 = None
        view_18: "f16[50, 32, 3, 768][73728, 2304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(add_12, [50, 32, 3, 768]);  add_12 = None
        unsqueeze_2: "f16[1, 50, 32, 3, 768][3686400, 73728, 2304, 768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_18, 0);  view_18 = None
        permute_14: "f16[3, 50, 32, 1, 768][768, 73728, 2304, 3686400, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_2, [3, 1, 2, 0, 4]);  unsqueeze_2 = None
        squeeze_1: "f16[3, 50, 32, 768][768, 73728, 2304, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_14, -2);  permute_14 = None
        clone_6: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
        select_3: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_6, 0, 0)
        select_4: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_6, 0, 1)
        select_5: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_6, 0, 2);  clone_6 = None
        view_19: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_3, [50, 384, 64]);  select_3 = None
        permute_15: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_19, [1, 0, 2]);  view_19 = None
        view_20: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_4, [50, 384, 64]);  select_4 = None
        permute_16: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_20, [1, 0, 2]);  view_20 = None
        view_21: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_5, [50, 384, 64]);  select_5 = None
        permute_17: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_21, [1, 0, 2]);  view_21 = None
        view_22: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_15, [32, 12, 50, 64]);  permute_15 = None
        view_23: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_16, [32, 12, 50, 64]);  permute_16 = None
        view_24: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_17, [32, 12, 50, 64]);  permute_17 = None
        _scaled_dot_product_cudnn_attention_1 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(view_22, view_23, view_24, None, True)
        getitem_17: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_1[0]
        getitem_18: "f32[32, 12, 50, 1][600, 50, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_1[1]
        getitem_23: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_1[6]
        getitem_24: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_1[7];  _scaled_dot_product_cudnn_attention_1 = None
        permute_18: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_17, [2, 0, 1, 3])
        view_25: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_18, [1600, 768]);  permute_18 = None
        convert_element_type_28: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_24, torch.float16);  primals_24 = None
        convert_element_type_29: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_23, torch.float16);  primals_23 = None
        permute_19: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_29, [1, 0]);  convert_element_type_29 = None
        addmm_3: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_28, view_25, permute_19);  convert_element_type_28 = view_25 = None
        view_26: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [50, 32, 768]);  addmm_3 = None
        permute_20: "f16[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_26, [1, 0, 2]);  view_26 = None
        add_13: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_9, permute_20);  add_9 = permute_20 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_13, [2], correction = 0, keepdim = True)
        getitem_26: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_4[0]
        getitem_27: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        add_14: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_4: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        sub_4: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_13, getitem_27);  getitem_27 = None
        mul_10: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        mul_11: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, primals_25)
        add_15: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_11, primals_26);  mul_11 = primals_26 = None
        convert_element_type_33: "f16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_28, torch.float16);  primals_28 = None
        convert_element_type_34: "f16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_27, torch.float16);  primals_27 = None
        convert_element_type_35: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_15, torch.float16);  add_15 = None
        view_27: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_35, [1600, 768]);  convert_element_type_35 = None
        permute_21: "f16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_34, [1, 0]);  convert_element_type_34 = None
        addmm_4: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_33, view_27, permute_21);  convert_element_type_33 = None
        view_28: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [32, 50, 3072])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_12: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_28, 1.702)
        sigmoid_1: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_12);  mul_12 = None
        mul_13: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_1, view_28);  sigmoid_1 = view_28 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        convert_element_type_39: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_30, torch.float16);  primals_30 = None
        convert_element_type_40: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_29, torch.float16);  primals_29 = None
        view_29: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_13, [1600, 3072]);  mul_13 = None
        permute_22: "f16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_40, [1, 0]);  convert_element_type_40 = None
        addmm_5: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_39, view_29, permute_22);  convert_element_type_39 = None
        view_30: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [32, 50, 768]);  addmm_5 = None
        add_16: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_13, view_30);  add_13 = view_30 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_16, [2], correction = 0, keepdim = True)
        getitem_28: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_5[0]
        getitem_29: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        add_17: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_5: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        sub_5: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_16, getitem_29);  getitem_29 = None
        mul_14: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        mul_15: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, primals_31)
        add_18: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_15, primals_32);  mul_15 = primals_32 = None
        permute_23: "f32[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.aten.permute.default(add_18, [1, 0, 2]);  add_18 = None
        convert_element_type_44: "f16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_33, torch.float16);  primals_33 = None
        convert_element_type_45: "f16[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_34, torch.float16);  primals_34 = None
        convert_element_type_46: "f16[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_23, torch.float16);  permute_23 = None
        permute_24: "f16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_45, [1, 0]);  convert_element_type_45 = None
        clone_10: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_46, memory_format = torch.contiguous_format);  convert_element_type_46 = None
        view_31: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_10, [1600, 768]);  clone_10 = None
        mm_2: "f16[1600, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_31, permute_24)
        view_32: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [50, 32, 2304]);  mm_2 = None
        add_19: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(view_32, convert_element_type_44);  view_32 = convert_element_type_44 = None
        view_33: "f16[50, 32, 3, 768][73728, 2304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(add_19, [50, 32, 3, 768]);  add_19 = None
        unsqueeze_3: "f16[1, 50, 32, 3, 768][3686400, 73728, 2304, 768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_33, 0);  view_33 = None
        permute_25: "f16[3, 50, 32, 1, 768][768, 73728, 2304, 3686400, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_3, [3, 1, 2, 0, 4]);  unsqueeze_3 = None
        squeeze_2: "f16[3, 50, 32, 768][768, 73728, 2304, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_25, -2);  permute_25 = None
        clone_11: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_2, memory_format = torch.contiguous_format);  squeeze_2 = None
        select_6: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_11, 0, 0)
        select_7: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_11, 0, 1)
        select_8: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_11, 0, 2);  clone_11 = None
        view_34: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_6, [50, 384, 64]);  select_6 = None
        permute_26: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_34, [1, 0, 2]);  view_34 = None
        view_35: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_7, [50, 384, 64]);  select_7 = None
        permute_27: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_35, [1, 0, 2]);  view_35 = None
        view_36: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_8, [50, 384, 64]);  select_8 = None
        permute_28: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_36, [1, 0, 2]);  view_36 = None
        view_37: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_26, [32, 12, 50, 64]);  permute_26 = None
        view_38: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_27, [32, 12, 50, 64]);  permute_27 = None
        view_39: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_28, [32, 12, 50, 64]);  permute_28 = None
        _scaled_dot_product_cudnn_attention_2 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(view_37, view_38, view_39, None, True)
        getitem_30: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_2[0]
        getitem_31: "f32[32, 12, 50, 1][600, 50, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_2[1]
        getitem_36: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_2[6]
        getitem_37: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_2[7];  _scaled_dot_product_cudnn_attention_2 = None
        permute_29: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_30, [2, 0, 1, 3])
        view_40: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_29, [1600, 768]);  permute_29 = None
        convert_element_type_49: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_36, torch.float16);  primals_36 = None
        convert_element_type_50: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_35, torch.float16);  primals_35 = None
        permute_30: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_50, [1, 0]);  convert_element_type_50 = None
        addmm_6: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_49, view_40, permute_30);  convert_element_type_49 = view_40 = None
        view_41: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [50, 32, 768]);  addmm_6 = None
        permute_31: "f16[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_41, [1, 0, 2]);  view_41 = None
        add_20: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_16, permute_31);  add_16 = permute_31 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_20, [2], correction = 0, keepdim = True)
        getitem_39: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_6[0]
        getitem_40: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        add_21: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_39, 1e-05);  getitem_39 = None
        rsqrt_6: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_6: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_20, getitem_40);  getitem_40 = None
        mul_16: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        mul_17: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_16, primals_37)
        add_22: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_17, primals_38);  mul_17 = primals_38 = None
        convert_element_type_54: "f16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_40, torch.float16);  primals_40 = None
        convert_element_type_55: "f16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_39, torch.float16);  primals_39 = None
        convert_element_type_56: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_22, torch.float16);  add_22 = None
        view_42: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_56, [1600, 768]);  convert_element_type_56 = None
        permute_32: "f16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_55, [1, 0]);  convert_element_type_55 = None
        addmm_7: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_54, view_42, permute_32);  convert_element_type_54 = None
        view_43: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [32, 50, 3072])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_18: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_43, 1.702)
        sigmoid_2: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_18);  mul_18 = None
        mul_19: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_2, view_43);  sigmoid_2 = view_43 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        convert_element_type_60: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_42, torch.float16);  primals_42 = None
        convert_element_type_61: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_41, torch.float16);  primals_41 = None
        view_44: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_19, [1600, 3072]);  mul_19 = None
        permute_33: "f16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_61, [1, 0]);  convert_element_type_61 = None
        addmm_8: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_60, view_44, permute_33);  convert_element_type_60 = None
        view_45: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [32, 50, 768]);  addmm_8 = None
        add_23: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_20, view_45);  add_20 = view_45 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_23, [2], correction = 0, keepdim = True)
        getitem_41: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_7[0]
        getitem_42: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        add_24: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_41, 1e-05);  getitem_41 = None
        rsqrt_7: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        sub_7: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_23, getitem_42);  getitem_42 = None
        mul_20: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        mul_21: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_20, primals_43)
        add_25: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_21, primals_44);  mul_21 = primals_44 = None
        permute_34: "f32[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.aten.permute.default(add_25, [1, 0, 2]);  add_25 = None
        convert_element_type_65: "f16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_45, torch.float16);  primals_45 = None
        convert_element_type_66: "f16[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_46, torch.float16);  primals_46 = None
        convert_element_type_67: "f16[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_34, torch.float16);  permute_34 = None
        permute_35: "f16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_66, [1, 0]);  convert_element_type_66 = None
        clone_15: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_67, memory_format = torch.contiguous_format);  convert_element_type_67 = None
        view_46: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_15, [1600, 768]);  clone_15 = None
        mm_3: "f16[1600, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_46, permute_35)
        view_47: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_3, [50, 32, 2304]);  mm_3 = None
        add_26: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(view_47, convert_element_type_65);  view_47 = convert_element_type_65 = None
        view_48: "f16[50, 32, 3, 768][73728, 2304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(add_26, [50, 32, 3, 768]);  add_26 = None
        unsqueeze_4: "f16[1, 50, 32, 3, 768][3686400, 73728, 2304, 768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_48, 0);  view_48 = None
        permute_36: "f16[3, 50, 32, 1, 768][768, 73728, 2304, 3686400, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_4, [3, 1, 2, 0, 4]);  unsqueeze_4 = None
        squeeze_3: "f16[3, 50, 32, 768][768, 73728, 2304, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_36, -2);  permute_36 = None
        clone_16: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_3, memory_format = torch.contiguous_format);  squeeze_3 = None
        select_9: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_16, 0, 0)
        select_10: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_16, 0, 1)
        select_11: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_16, 0, 2);  clone_16 = None
        view_49: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_9, [50, 384, 64]);  select_9 = None
        permute_37: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_49, [1, 0, 2]);  view_49 = None
        view_50: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_10, [50, 384, 64]);  select_10 = None
        permute_38: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_50, [1, 0, 2]);  view_50 = None
        view_51: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_11, [50, 384, 64]);  select_11 = None
        permute_39: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_51, [1, 0, 2]);  view_51 = None
        view_52: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_37, [32, 12, 50, 64]);  permute_37 = None
        view_53: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_38, [32, 12, 50, 64]);  permute_38 = None
        view_54: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_39, [32, 12, 50, 64]);  permute_39 = None
        _scaled_dot_product_cudnn_attention_3 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(view_52, view_53, view_54, None, True)
        getitem_43: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_3[0]
        getitem_44: "f32[32, 12, 50, 1][600, 50, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_3[1]
        getitem_49: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_3[6]
        getitem_50: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_3[7];  _scaled_dot_product_cudnn_attention_3 = None
        permute_40: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_43, [2, 0, 1, 3])
        view_55: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_40, [1600, 768]);  permute_40 = None
        convert_element_type_70: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_48, torch.float16);  primals_48 = None
        convert_element_type_71: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_47, torch.float16);  primals_47 = None
        permute_41: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_71, [1, 0]);  convert_element_type_71 = None
        addmm_9: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_70, view_55, permute_41);  convert_element_type_70 = view_55 = None
        view_56: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [50, 32, 768]);  addmm_9 = None
        permute_42: "f16[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_56, [1, 0, 2]);  view_56 = None
        add_27: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_23, permute_42);  add_23 = permute_42 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_52: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_8[0]
        getitem_53: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        add_28: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
        rsqrt_8: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_8: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_27, getitem_53);  getitem_53 = None
        mul_22: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        mul_23: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_22, primals_49)
        add_29: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_23, primals_50);  mul_23 = primals_50 = None
        convert_element_type_75: "f16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_52, torch.float16);  primals_52 = None
        convert_element_type_76: "f16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_51, torch.float16);  primals_51 = None
        convert_element_type_77: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_29, torch.float16);  add_29 = None
        view_57: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_77, [1600, 768]);  convert_element_type_77 = None
        permute_43: "f16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_76, [1, 0]);  convert_element_type_76 = None
        addmm_10: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_75, view_57, permute_43);  convert_element_type_75 = None
        view_58: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [32, 50, 3072])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_24: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_58, 1.702)
        sigmoid_3: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_24);  mul_24 = None
        mul_25: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_3, view_58);  sigmoid_3 = view_58 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        convert_element_type_81: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_54, torch.float16);  primals_54 = None
        convert_element_type_82: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_53, torch.float16);  primals_53 = None
        view_59: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_25, [1600, 3072]);  mul_25 = None
        permute_44: "f16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_82, [1, 0]);  convert_element_type_82 = None
        addmm_11: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_81, view_59, permute_44);  convert_element_type_81 = None
        view_60: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [32, 50, 768]);  addmm_11 = None
        add_30: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_27, view_60);  add_27 = view_60 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_30, [2], correction = 0, keepdim = True)
        getitem_54: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_9[0]
        getitem_55: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        add_31: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
        rsqrt_9: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        sub_9: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_30, getitem_55);  getitem_55 = None
        mul_26: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        mul_27: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_26, primals_55)
        add_32: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_27, primals_56);  mul_27 = primals_56 = None
        permute_45: "f32[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.aten.permute.default(add_32, [1, 0, 2]);  add_32 = None
        convert_element_type_86: "f16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_57, torch.float16);  primals_57 = None
        convert_element_type_87: "f16[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_58, torch.float16);  primals_58 = None
        convert_element_type_88: "f16[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_45, torch.float16);  permute_45 = None
        permute_46: "f16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_87, [1, 0]);  convert_element_type_87 = None
        clone_20: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_88, memory_format = torch.contiguous_format);  convert_element_type_88 = None
        view_61: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_20, [1600, 768]);  clone_20 = None
        mm_4: "f16[1600, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_61, permute_46)
        view_62: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [50, 32, 2304]);  mm_4 = None
        add_33: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(view_62, convert_element_type_86);  view_62 = convert_element_type_86 = None
        view_63: "f16[50, 32, 3, 768][73728, 2304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(add_33, [50, 32, 3, 768]);  add_33 = None
        unsqueeze_5: "f16[1, 50, 32, 3, 768][3686400, 73728, 2304, 768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_63, 0);  view_63 = None
        permute_47: "f16[3, 50, 32, 1, 768][768, 73728, 2304, 3686400, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_5, [3, 1, 2, 0, 4]);  unsqueeze_5 = None
        squeeze_4: "f16[3, 50, 32, 768][768, 73728, 2304, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_47, -2);  permute_47 = None
        clone_21: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_4, memory_format = torch.contiguous_format);  squeeze_4 = None
        select_12: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_21, 0, 0)
        select_13: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_21, 0, 1)
        select_14: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_21, 0, 2);  clone_21 = None
        view_64: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_12, [50, 384, 64]);  select_12 = None
        permute_48: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_64, [1, 0, 2]);  view_64 = None
        view_65: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_13, [50, 384, 64]);  select_13 = None
        permute_49: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_65, [1, 0, 2]);  view_65 = None
        view_66: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_14, [50, 384, 64]);  select_14 = None
        permute_50: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_66, [1, 0, 2]);  view_66 = None
        view_67: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_48, [32, 12, 50, 64]);  permute_48 = None
        view_68: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_49, [32, 12, 50, 64]);  permute_49 = None
        view_69: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_50, [32, 12, 50, 64]);  permute_50 = None
        _scaled_dot_product_cudnn_attention_4 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(view_67, view_68, view_69, None, True)
        getitem_56: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_4[0]
        getitem_57: "f32[32, 12, 50, 1][600, 50, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_4[1]
        getitem_62: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_4[6]
        getitem_63: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_4[7];  _scaled_dot_product_cudnn_attention_4 = None
        permute_51: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_56, [2, 0, 1, 3])
        view_70: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_51, [1600, 768]);  permute_51 = None
        convert_element_type_91: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_60, torch.float16);  primals_60 = None
        convert_element_type_92: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_59, torch.float16);  primals_59 = None
        permute_52: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_92, [1, 0]);  convert_element_type_92 = None
        addmm_12: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_91, view_70, permute_52);  convert_element_type_91 = view_70 = None
        view_71: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [50, 32, 768]);  addmm_12 = None
        permute_53: "f16[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_71, [1, 0, 2]);  view_71 = None
        add_34: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_30, permute_53);  add_30 = permute_53 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_34, [2], correction = 0, keepdim = True)
        getitem_65: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_10[0]
        getitem_66: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        add_35: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_65, 1e-05);  getitem_65 = None
        rsqrt_10: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        sub_10: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_34, getitem_66);  getitem_66 = None
        mul_28: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        mul_29: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, primals_61)
        add_36: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_29, primals_62);  mul_29 = primals_62 = None
        convert_element_type_96: "f16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_64, torch.float16);  primals_64 = None
        convert_element_type_97: "f16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_63, torch.float16);  primals_63 = None
        convert_element_type_98: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_36, torch.float16);  add_36 = None
        view_72: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_98, [1600, 768]);  convert_element_type_98 = None
        permute_54: "f16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_97, [1, 0]);  convert_element_type_97 = None
        addmm_13: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_96, view_72, permute_54);  convert_element_type_96 = None
        view_73: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [32, 50, 3072])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_30: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_73, 1.702)
        sigmoid_4: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_30);  mul_30 = None
        mul_31: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_4, view_73);  sigmoid_4 = view_73 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        convert_element_type_102: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_66, torch.float16);  primals_66 = None
        convert_element_type_103: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_65, torch.float16);  primals_65 = None
        view_74: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_31, [1600, 3072]);  mul_31 = None
        permute_55: "f16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_103, [1, 0]);  convert_element_type_103 = None
        addmm_14: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_102, view_74, permute_55);  convert_element_type_102 = None
        view_75: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [32, 50, 768]);  addmm_14 = None
        add_37: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_34, view_75);  add_34 = view_75 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
        getitem_67: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_11[0]
        getitem_68: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        add_38: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_67, 1e-05);  getitem_67 = None
        rsqrt_11: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        sub_11: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_37, getitem_68);  getitem_68 = None
        mul_32: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        mul_33: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_32, primals_67)
        add_39: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_33, primals_68);  mul_33 = primals_68 = None
        permute_56: "f32[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.aten.permute.default(add_39, [1, 0, 2]);  add_39 = None
        convert_element_type_107: "f16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_69, torch.float16);  primals_69 = None
        convert_element_type_108: "f16[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_70, torch.float16);  primals_70 = None
        convert_element_type_109: "f16[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_56, torch.float16);  permute_56 = None
        permute_57: "f16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_108, [1, 0]);  convert_element_type_108 = None
        clone_25: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_109, memory_format = torch.contiguous_format);  convert_element_type_109 = None
        view_76: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_25, [1600, 768]);  clone_25 = None
        mm_5: "f16[1600, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_76, permute_57)
        view_77: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_5, [50, 32, 2304]);  mm_5 = None
        add_40: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(view_77, convert_element_type_107);  view_77 = convert_element_type_107 = None
        view_78: "f16[50, 32, 3, 768][73728, 2304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(add_40, [50, 32, 3, 768]);  add_40 = None
        unsqueeze_6: "f16[1, 50, 32, 3, 768][3686400, 73728, 2304, 768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_78, 0);  view_78 = None
        permute_58: "f16[3, 50, 32, 1, 768][768, 73728, 2304, 3686400, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_6, [3, 1, 2, 0, 4]);  unsqueeze_6 = None
        squeeze_5: "f16[3, 50, 32, 768][768, 73728, 2304, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_58, -2);  permute_58 = None
        clone_26: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_5, memory_format = torch.contiguous_format);  squeeze_5 = None
        select_15: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_26, 0, 0)
        select_16: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_26, 0, 1)
        select_17: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_26, 0, 2);  clone_26 = None
        view_79: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_15, [50, 384, 64]);  select_15 = None
        permute_59: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_79, [1, 0, 2]);  view_79 = None
        view_80: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_16, [50, 384, 64]);  select_16 = None
        permute_60: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_80, [1, 0, 2]);  view_80 = None
        view_81: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_17, [50, 384, 64]);  select_17 = None
        permute_61: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_81, [1, 0, 2]);  view_81 = None
        view_82: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_59, [32, 12, 50, 64]);  permute_59 = None
        view_83: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_60, [32, 12, 50, 64]);  permute_60 = None
        view_84: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_61, [32, 12, 50, 64]);  permute_61 = None
        _scaled_dot_product_cudnn_attention_5 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(view_82, view_83, view_84, None, True)
        getitem_69: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_5[0]
        getitem_70: "f32[32, 12, 50, 1][600, 50, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_5[1]
        getitem_75: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_5[6]
        getitem_76: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_5[7];  _scaled_dot_product_cudnn_attention_5 = None
        permute_62: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_69, [2, 0, 1, 3])
        view_85: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_62, [1600, 768]);  permute_62 = None
        convert_element_type_112: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_72, torch.float16);  primals_72 = None
        convert_element_type_113: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_71, torch.float16);  primals_71 = None
        permute_63: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_113, [1, 0]);  convert_element_type_113 = None
        addmm_15: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_112, view_85, permute_63);  convert_element_type_112 = view_85 = None
        view_86: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [50, 32, 768]);  addmm_15 = None
        permute_64: "f16[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_86, [1, 0, 2]);  view_86 = None
        add_41: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_37, permute_64);  add_37 = permute_64 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
        getitem_78: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_12[0]
        getitem_79: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        add_42: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
        rsqrt_12: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_12: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_41, getitem_79);  getitem_79 = None
        mul_34: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        mul_35: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_34, primals_73)
        add_43: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_35, primals_74);  mul_35 = primals_74 = None
        convert_element_type_117: "f16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_76, torch.float16);  primals_76 = None
        convert_element_type_118: "f16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_75, torch.float16);  primals_75 = None
        convert_element_type_119: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_43, torch.float16);  add_43 = None
        view_87: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_119, [1600, 768]);  convert_element_type_119 = None
        permute_65: "f16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_118, [1, 0]);  convert_element_type_118 = None
        addmm_16: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_117, view_87, permute_65);  convert_element_type_117 = None
        view_88: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [32, 50, 3072])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_36: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_88, 1.702)
        sigmoid_5: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_36);  mul_36 = None
        mul_37: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_5, view_88);  sigmoid_5 = view_88 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        convert_element_type_123: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_78, torch.float16);  primals_78 = None
        convert_element_type_124: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_77, torch.float16);  primals_77 = None
        view_89: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_37, [1600, 3072]);  mul_37 = None
        permute_66: "f16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_124, [1, 0]);  convert_element_type_124 = None
        addmm_17: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_123, view_89, permute_66);  convert_element_type_123 = None
        view_90: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [32, 50, 768]);  addmm_17 = None
        add_44: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_41, view_90);  add_41 = view_90 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_44, [2], correction = 0, keepdim = True)
        getitem_80: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_13[0]
        getitem_81: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        add_45: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
        rsqrt_13: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        sub_13: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_44, getitem_81);  getitem_81 = None
        mul_38: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        mul_39: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, primals_79)
        add_46: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_39, primals_80);  mul_39 = primals_80 = None
        permute_67: "f32[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.aten.permute.default(add_46, [1, 0, 2]);  add_46 = None
        convert_element_type_128: "f16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_81, torch.float16);  primals_81 = None
        convert_element_type_129: "f16[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_82, torch.float16);  primals_82 = None
        convert_element_type_130: "f16[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_67, torch.float16);  permute_67 = None
        permute_68: "f16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_129, [1, 0]);  convert_element_type_129 = None
        clone_30: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_130, memory_format = torch.contiguous_format);  convert_element_type_130 = None
        view_91: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_30, [1600, 768]);  clone_30 = None
        mm_6: "f16[1600, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_91, permute_68)
        view_92: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [50, 32, 2304]);  mm_6 = None
        add_47: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(view_92, convert_element_type_128);  view_92 = convert_element_type_128 = None
        view_93: "f16[50, 32, 3, 768][73728, 2304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(add_47, [50, 32, 3, 768]);  add_47 = None
        unsqueeze_7: "f16[1, 50, 32, 3, 768][3686400, 73728, 2304, 768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_93, 0);  view_93 = None
        permute_69: "f16[3, 50, 32, 1, 768][768, 73728, 2304, 3686400, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_7, [3, 1, 2, 0, 4]);  unsqueeze_7 = None
        squeeze_6: "f16[3, 50, 32, 768][768, 73728, 2304, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_69, -2);  permute_69 = None
        clone_31: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_6, memory_format = torch.contiguous_format);  squeeze_6 = None
        select_18: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_31, 0, 0)
        select_19: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_31, 0, 1)
        select_20: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_31, 0, 2);  clone_31 = None
        view_94: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_18, [50, 384, 64]);  select_18 = None
        permute_70: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_94, [1, 0, 2]);  view_94 = None
        view_95: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_19, [50, 384, 64]);  select_19 = None
        permute_71: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_95, [1, 0, 2]);  view_95 = None
        view_96: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_20, [50, 384, 64]);  select_20 = None
        permute_72: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_96, [1, 0, 2]);  view_96 = None
        view_97: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_70, [32, 12, 50, 64]);  permute_70 = None
        view_98: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_71, [32, 12, 50, 64]);  permute_71 = None
        view_99: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_72, [32, 12, 50, 64]);  permute_72 = None
        _scaled_dot_product_cudnn_attention_6 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(view_97, view_98, view_99, None, True)
        getitem_82: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_6[0]
        getitem_83: "f32[32, 12, 50, 1][600, 50, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_6[1]
        getitem_88: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_6[6]
        getitem_89: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_6[7];  _scaled_dot_product_cudnn_attention_6 = None
        permute_73: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_82, [2, 0, 1, 3])
        view_100: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_73, [1600, 768]);  permute_73 = None
        convert_element_type_133: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_84, torch.float16);  primals_84 = None
        convert_element_type_134: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_83, torch.float16);  primals_83 = None
        permute_74: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_134, [1, 0]);  convert_element_type_134 = None
        addmm_18: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_133, view_100, permute_74);  convert_element_type_133 = view_100 = None
        view_101: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [50, 32, 768]);  addmm_18 = None
        permute_75: "f16[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_101, [1, 0, 2]);  view_101 = None
        add_48: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_44, permute_75);  add_44 = permute_75 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
        getitem_91: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_14[0]
        getitem_92: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        add_49: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_91, 1e-05);  getitem_91 = None
        rsqrt_14: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        sub_14: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_48, getitem_92);  getitem_92 = None
        mul_40: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        mul_41: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_40, primals_85)
        add_50: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_41, primals_86);  mul_41 = primals_86 = None
        convert_element_type_138: "f16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_88, torch.float16);  primals_88 = None
        convert_element_type_139: "f16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_87, torch.float16);  primals_87 = None
        convert_element_type_140: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_50, torch.float16);  add_50 = None
        view_102: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_140, [1600, 768]);  convert_element_type_140 = None
        permute_76: "f16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_139, [1, 0]);  convert_element_type_139 = None
        addmm_19: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_138, view_102, permute_76);  convert_element_type_138 = None
        view_103: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [32, 50, 3072])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_42: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_103, 1.702)
        sigmoid_6: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_42);  mul_42 = None
        mul_43: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_6, view_103);  sigmoid_6 = view_103 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        convert_element_type_144: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_90, torch.float16);  primals_90 = None
        convert_element_type_145: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_89, torch.float16);  primals_89 = None
        view_104: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_43, [1600, 3072]);  mul_43 = None
        permute_77: "f16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_145, [1, 0]);  convert_element_type_145 = None
        addmm_20: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_144, view_104, permute_77);  convert_element_type_144 = None
        view_105: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [32, 50, 768]);  addmm_20 = None
        add_51: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_48, view_105);  add_48 = view_105 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
        getitem_93: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_15[0]
        getitem_94: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        add_52: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_93, 1e-05);  getitem_93 = None
        rsqrt_15: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        sub_15: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_51, getitem_94);  getitem_94 = None
        mul_44: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        mul_45: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_44, primals_91)
        add_53: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_45, primals_92);  mul_45 = primals_92 = None
        permute_78: "f32[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.aten.permute.default(add_53, [1, 0, 2]);  add_53 = None
        convert_element_type_149: "f16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_93, torch.float16);  primals_93 = None
        convert_element_type_150: "f16[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_94, torch.float16);  primals_94 = None
        convert_element_type_151: "f16[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_78, torch.float16);  permute_78 = None
        permute_79: "f16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_150, [1, 0]);  convert_element_type_150 = None
        clone_35: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_151, memory_format = torch.contiguous_format);  convert_element_type_151 = None
        view_106: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_35, [1600, 768]);  clone_35 = None
        mm_7: "f16[1600, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_106, permute_79)
        view_107: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_7, [50, 32, 2304]);  mm_7 = None
        add_54: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(view_107, convert_element_type_149);  view_107 = convert_element_type_149 = None
        view_108: "f16[50, 32, 3, 768][73728, 2304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(add_54, [50, 32, 3, 768]);  add_54 = None
        unsqueeze_8: "f16[1, 50, 32, 3, 768][3686400, 73728, 2304, 768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_108, 0);  view_108 = None
        permute_80: "f16[3, 50, 32, 1, 768][768, 73728, 2304, 3686400, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_8, [3, 1, 2, 0, 4]);  unsqueeze_8 = None
        squeeze_7: "f16[3, 50, 32, 768][768, 73728, 2304, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_80, -2);  permute_80 = None
        clone_36: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_7, memory_format = torch.contiguous_format);  squeeze_7 = None
        select_21: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_36, 0, 0)
        select_22: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_36, 0, 1)
        select_23: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_36, 0, 2);  clone_36 = None
        view_109: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_21, [50, 384, 64]);  select_21 = None
        permute_81: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_109, [1, 0, 2]);  view_109 = None
        view_110: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_22, [50, 384, 64]);  select_22 = None
        permute_82: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_110, [1, 0, 2]);  view_110 = None
        view_111: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_23, [50, 384, 64]);  select_23 = None
        permute_83: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_111, [1, 0, 2]);  view_111 = None
        view_112: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_81, [32, 12, 50, 64]);  permute_81 = None
        view_113: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_82, [32, 12, 50, 64]);  permute_82 = None
        view_114: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_83, [32, 12, 50, 64]);  permute_83 = None
        _scaled_dot_product_cudnn_attention_7 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(view_112, view_113, view_114, None, True)
        getitem_95: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_7[0]
        getitem_96: "f32[32, 12, 50, 1][600, 50, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_7[1]
        getitem_101: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_7[6]
        getitem_102: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_7[7];  _scaled_dot_product_cudnn_attention_7 = None
        permute_84: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_95, [2, 0, 1, 3])
        view_115: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_84, [1600, 768]);  permute_84 = None
        convert_element_type_154: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_96, torch.float16);  primals_96 = None
        convert_element_type_155: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_95, torch.float16);  primals_95 = None
        permute_85: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_155, [1, 0]);  convert_element_type_155 = None
        addmm_21: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_154, view_115, permute_85);  convert_element_type_154 = view_115 = None
        view_116: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [50, 32, 768]);  addmm_21 = None
        permute_86: "f16[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_116, [1, 0, 2]);  view_116 = None
        add_55: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_51, permute_86);  add_51 = permute_86 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
        getitem_104: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_16[0]
        getitem_105: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        add_56: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
        rsqrt_16: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        sub_16: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_55, getitem_105);  getitem_105 = None
        mul_46: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        mul_47: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_46, primals_97)
        add_57: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_47, primals_98);  mul_47 = primals_98 = None
        convert_element_type_159: "f16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_100, torch.float16);  primals_100 = None
        convert_element_type_160: "f16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_99, torch.float16);  primals_99 = None
        convert_element_type_161: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_57, torch.float16);  add_57 = None
        view_117: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_161, [1600, 768]);  convert_element_type_161 = None
        permute_87: "f16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_160, [1, 0]);  convert_element_type_160 = None
        addmm_22: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_159, view_117, permute_87);  convert_element_type_159 = None
        view_118: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [32, 50, 3072])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_48: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_118, 1.702)
        sigmoid_7: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_48);  mul_48 = None
        mul_49: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_7, view_118);  sigmoid_7 = view_118 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        convert_element_type_165: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_102, torch.float16);  primals_102 = None
        convert_element_type_166: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_101, torch.float16);  primals_101 = None
        view_119: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_49, [1600, 3072]);  mul_49 = None
        permute_88: "f16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_166, [1, 0]);  convert_element_type_166 = None
        addmm_23: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_165, view_119, permute_88);  convert_element_type_165 = None
        view_120: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [32, 50, 768]);  addmm_23 = None
        add_58: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_55, view_120);  add_55 = view_120 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_58, [2], correction = 0, keepdim = True)
        getitem_106: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_17[0]
        getitem_107: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        add_59: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_106, 1e-05);  getitem_106 = None
        rsqrt_17: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
        sub_17: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_58, getitem_107);  getitem_107 = None
        mul_50: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        mul_51: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_50, primals_103)
        add_60: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_51, primals_104);  mul_51 = primals_104 = None
        permute_89: "f32[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.aten.permute.default(add_60, [1, 0, 2]);  add_60 = None
        convert_element_type_170: "f16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_105, torch.float16);  primals_105 = None
        convert_element_type_171: "f16[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_106, torch.float16);  primals_106 = None
        convert_element_type_172: "f16[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_89, torch.float16);  permute_89 = None
        permute_90: "f16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_171, [1, 0]);  convert_element_type_171 = None
        clone_40: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_172, memory_format = torch.contiguous_format);  convert_element_type_172 = None
        view_121: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_40, [1600, 768]);  clone_40 = None
        mm_8: "f16[1600, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_121, permute_90)
        view_122: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [50, 32, 2304]);  mm_8 = None
        add_61: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(view_122, convert_element_type_170);  view_122 = convert_element_type_170 = None
        view_123: "f16[50, 32, 3, 768][73728, 2304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(add_61, [50, 32, 3, 768]);  add_61 = None
        unsqueeze_9: "f16[1, 50, 32, 3, 768][3686400, 73728, 2304, 768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_123, 0);  view_123 = None
        permute_91: "f16[3, 50, 32, 1, 768][768, 73728, 2304, 3686400, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_9, [3, 1, 2, 0, 4]);  unsqueeze_9 = None
        squeeze_8: "f16[3, 50, 32, 768][768, 73728, 2304, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_91, -2);  permute_91 = None
        clone_41: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_8, memory_format = torch.contiguous_format);  squeeze_8 = None
        select_24: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_41, 0, 0)
        select_25: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_41, 0, 1)
        select_26: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_41, 0, 2);  clone_41 = None
        view_124: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_24, [50, 384, 64]);  select_24 = None
        permute_92: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_124, [1, 0, 2]);  view_124 = None
        view_125: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_25, [50, 384, 64]);  select_25 = None
        permute_93: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_125, [1, 0, 2]);  view_125 = None
        view_126: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_26, [50, 384, 64]);  select_26 = None
        permute_94: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_126, [1, 0, 2]);  view_126 = None
        view_127: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_92, [32, 12, 50, 64]);  permute_92 = None
        view_128: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_93, [32, 12, 50, 64]);  permute_93 = None
        view_129: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_94, [32, 12, 50, 64]);  permute_94 = None
        _scaled_dot_product_cudnn_attention_8 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(view_127, view_128, view_129, None, True)
        getitem_108: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_8[0]
        getitem_109: "f32[32, 12, 50, 1][600, 50, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_8[1]
        getitem_114: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_8[6]
        getitem_115: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_8[7];  _scaled_dot_product_cudnn_attention_8 = None
        permute_95: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_108, [2, 0, 1, 3])
        view_130: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_95, [1600, 768]);  permute_95 = None
        convert_element_type_175: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_108, torch.float16);  primals_108 = None
        convert_element_type_176: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_107, torch.float16);  primals_107 = None
        permute_96: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_176, [1, 0]);  convert_element_type_176 = None
        addmm_24: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_175, view_130, permute_96);  convert_element_type_175 = view_130 = None
        view_131: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_24, [50, 32, 768]);  addmm_24 = None
        permute_97: "f16[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_131, [1, 0, 2]);  view_131 = None
        add_62: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_58, permute_97);  add_58 = permute_97 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_62, [2], correction = 0, keepdim = True)
        getitem_117: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_18[0]
        getitem_118: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        add_63: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_117, 1e-05);  getitem_117 = None
        rsqrt_18: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
        sub_18: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_62, getitem_118);  getitem_118 = None
        mul_52: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        mul_53: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_52, primals_109)
        add_64: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_53, primals_110);  mul_53 = primals_110 = None
        convert_element_type_180: "f16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_112, torch.float16);  primals_112 = None
        convert_element_type_181: "f16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_111, torch.float16);  primals_111 = None
        convert_element_type_182: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_64, torch.float16);  add_64 = None
        view_132: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_182, [1600, 768]);  convert_element_type_182 = None
        permute_98: "f16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_181, [1, 0]);  convert_element_type_181 = None
        addmm_25: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_180, view_132, permute_98);  convert_element_type_180 = None
        view_133: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [32, 50, 3072])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_54: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_133, 1.702)
        sigmoid_8: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_54);  mul_54 = None
        mul_55: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_8, view_133);  sigmoid_8 = view_133 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        convert_element_type_186: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_114, torch.float16);  primals_114 = None
        convert_element_type_187: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_113, torch.float16);  primals_113 = None
        view_134: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_55, [1600, 3072]);  mul_55 = None
        permute_99: "f16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_187, [1, 0]);  convert_element_type_187 = None
        addmm_26: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_186, view_134, permute_99);  convert_element_type_186 = None
        view_135: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [32, 50, 768]);  addmm_26 = None
        add_65: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_62, view_135);  add_62 = view_135 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
        getitem_119: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_19[0]
        getitem_120: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        add_66: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_119, 1e-05);  getitem_119 = None
        rsqrt_19: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_19: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_65, getitem_120);  getitem_120 = None
        mul_56: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        mul_57: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_56, primals_115)
        add_67: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_57, primals_116);  mul_57 = primals_116 = None
        permute_100: "f32[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.aten.permute.default(add_67, [1, 0, 2]);  add_67 = None
        convert_element_type_191: "f16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_117, torch.float16);  primals_117 = None
        convert_element_type_192: "f16[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_118, torch.float16);  primals_118 = None
        convert_element_type_193: "f16[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_100, torch.float16);  permute_100 = None
        permute_101: "f16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_192, [1, 0]);  convert_element_type_192 = None
        clone_45: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_193, memory_format = torch.contiguous_format);  convert_element_type_193 = None
        view_136: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_45, [1600, 768]);  clone_45 = None
        mm_9: "f16[1600, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_136, permute_101)
        view_137: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_9, [50, 32, 2304]);  mm_9 = None
        add_68: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(view_137, convert_element_type_191);  view_137 = convert_element_type_191 = None
        view_138: "f16[50, 32, 3, 768][73728, 2304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(add_68, [50, 32, 3, 768]);  add_68 = None
        unsqueeze_10: "f16[1, 50, 32, 3, 768][3686400, 73728, 2304, 768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_138, 0);  view_138 = None
        permute_102: "f16[3, 50, 32, 1, 768][768, 73728, 2304, 3686400, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_10, [3, 1, 2, 0, 4]);  unsqueeze_10 = None
        squeeze_9: "f16[3, 50, 32, 768][768, 73728, 2304, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_102, -2);  permute_102 = None
        clone_46: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_9, memory_format = torch.contiguous_format);  squeeze_9 = None
        select_27: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_46, 0, 0)
        select_28: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_46, 0, 1)
        select_29: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_46, 0, 2);  clone_46 = None
        view_139: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_27, [50, 384, 64]);  select_27 = None
        permute_103: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_139, [1, 0, 2]);  view_139 = None
        view_140: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_28, [50, 384, 64]);  select_28 = None
        permute_104: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_140, [1, 0, 2]);  view_140 = None
        view_141: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_29, [50, 384, 64]);  select_29 = None
        permute_105: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_141, [1, 0, 2]);  view_141 = None
        view_142: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_103, [32, 12, 50, 64]);  permute_103 = None
        view_143: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_104, [32, 12, 50, 64]);  permute_104 = None
        view_144: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_105, [32, 12, 50, 64]);  permute_105 = None
        _scaled_dot_product_cudnn_attention_9 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(view_142, view_143, view_144, None, True)
        getitem_121: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_9[0]
        getitem_122: "f32[32, 12, 50, 1][600, 50, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_9[1]
        getitem_127: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_9[6]
        getitem_128: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_9[7];  _scaled_dot_product_cudnn_attention_9 = None
        permute_106: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_121, [2, 0, 1, 3])
        view_145: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_106, [1600, 768]);  permute_106 = None
        convert_element_type_196: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_120, torch.float16);  primals_120 = None
        convert_element_type_197: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_119, torch.float16);  primals_119 = None
        permute_107: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_197, [1, 0]);  convert_element_type_197 = None
        addmm_27: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_196, view_145, permute_107);  convert_element_type_196 = view_145 = None
        view_146: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_27, [50, 32, 768]);  addmm_27 = None
        permute_108: "f16[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_146, [1, 0, 2]);  view_146 = None
        add_69: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_65, permute_108);  add_65 = permute_108 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_69, [2], correction = 0, keepdim = True)
        getitem_130: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_20[0]
        getitem_131: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None
        add_70: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_20: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        sub_20: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_69, getitem_131);  getitem_131 = None
        mul_58: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
        mul_59: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_58, primals_121)
        add_71: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_59, primals_122);  mul_59 = primals_122 = None
        convert_element_type_201: "f16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_124, torch.float16);  primals_124 = None
        convert_element_type_202: "f16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_123, torch.float16);  primals_123 = None
        convert_element_type_203: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_71, torch.float16);  add_71 = None
        view_147: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_203, [1600, 768]);  convert_element_type_203 = None
        permute_109: "f16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_202, [1, 0]);  convert_element_type_202 = None
        addmm_28: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_201, view_147, permute_109);  convert_element_type_201 = None
        view_148: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [32, 50, 3072])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_60: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_148, 1.702)
        sigmoid_9: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_60);  mul_60 = None
        mul_61: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_9, view_148);  sigmoid_9 = view_148 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        convert_element_type_207: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_126, torch.float16);  primals_126 = None
        convert_element_type_208: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_125, torch.float16);  primals_125 = None
        view_149: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_61, [1600, 3072]);  mul_61 = None
        permute_110: "f16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_208, [1, 0]);  convert_element_type_208 = None
        addmm_29: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_207, view_149, permute_110);  convert_element_type_207 = None
        view_150: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_29, [32, 50, 768]);  addmm_29 = None
        add_72: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_69, view_150);  add_69 = view_150 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_72, [2], correction = 0, keepdim = True)
        getitem_132: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_21[0]
        getitem_133: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        add_73: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_132, 1e-05);  getitem_132 = None
        rsqrt_21: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        sub_21: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_72, getitem_133);  getitem_133 = None
        mul_62: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
        mul_63: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_62, primals_127)
        add_74: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_63, primals_128);  mul_63 = primals_128 = None
        permute_111: "f32[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.aten.permute.default(add_74, [1, 0, 2]);  add_74 = None
        convert_element_type_212: "f16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_129, torch.float16);  primals_129 = None
        convert_element_type_213: "f16[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_130, torch.float16);  primals_130 = None
        convert_element_type_214: "f16[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_111, torch.float16);  permute_111 = None
        permute_112: "f16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_213, [1, 0]);  convert_element_type_213 = None
        clone_50: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_214, memory_format = torch.contiguous_format);  convert_element_type_214 = None
        view_151: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_50, [1600, 768]);  clone_50 = None
        mm_10: "f16[1600, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_151, permute_112)
        view_152: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [50, 32, 2304]);  mm_10 = None
        add_75: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(view_152, convert_element_type_212);  view_152 = convert_element_type_212 = None
        view_153: "f16[50, 32, 3, 768][73728, 2304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(add_75, [50, 32, 3, 768]);  add_75 = None
        unsqueeze_11: "f16[1, 50, 32, 3, 768][3686400, 73728, 2304, 768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_153, 0);  view_153 = None
        permute_113: "f16[3, 50, 32, 1, 768][768, 73728, 2304, 3686400, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_11, [3, 1, 2, 0, 4]);  unsqueeze_11 = None
        squeeze_10: "f16[3, 50, 32, 768][768, 73728, 2304, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_113, -2);  permute_113 = None
        clone_51: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_10, memory_format = torch.contiguous_format);  squeeze_10 = None
        select_30: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_51, 0, 0)
        select_31: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_51, 0, 1)
        select_32: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_51, 0, 2);  clone_51 = None
        view_154: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_30, [50, 384, 64]);  select_30 = None
        permute_114: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_154, [1, 0, 2]);  view_154 = None
        view_155: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_31, [50, 384, 64]);  select_31 = None
        permute_115: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_155, [1, 0, 2]);  view_155 = None
        view_156: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_32, [50, 384, 64]);  select_32 = None
        permute_116: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_156, [1, 0, 2]);  view_156 = None
        view_157: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_114, [32, 12, 50, 64]);  permute_114 = None
        view_158: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_115, [32, 12, 50, 64]);  permute_115 = None
        view_159: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_116, [32, 12, 50, 64]);  permute_116 = None
        _scaled_dot_product_cudnn_attention_10 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(view_157, view_158, view_159, None, True)
        getitem_134: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_10[0]
        getitem_135: "f32[32, 12, 50, 1][600, 50, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_10[1]
        getitem_140: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_10[6]
        getitem_141: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_10[7];  _scaled_dot_product_cudnn_attention_10 = None
        permute_117: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_134, [2, 0, 1, 3])
        view_160: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_117, [1600, 768]);  permute_117 = None
        convert_element_type_217: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_132, torch.float16);  primals_132 = None
        convert_element_type_218: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_131, torch.float16);  primals_131 = None
        permute_118: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_218, [1, 0]);  convert_element_type_218 = None
        addmm_30: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_217, view_160, permute_118);  convert_element_type_217 = view_160 = None
        view_161: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [50, 32, 768]);  addmm_30 = None
        permute_119: "f16[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_161, [1, 0, 2]);  view_161 = None
        add_76: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_72, permute_119);  add_72 = permute_119 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_76, [2], correction = 0, keepdim = True)
        getitem_143: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_22[0]
        getitem_144: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None
        add_77: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_143, 1e-05);  getitem_143 = None
        rsqrt_22: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        sub_22: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_76, getitem_144);  getitem_144 = None
        mul_64: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        mul_65: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_64, primals_133)
        add_78: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_65, primals_134);  mul_65 = primals_134 = None
        convert_element_type_222: "f16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_136, torch.float16);  primals_136 = None
        convert_element_type_223: "f16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_135, torch.float16);  primals_135 = None
        convert_element_type_224: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_78, torch.float16);  add_78 = None
        view_162: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_224, [1600, 768]);  convert_element_type_224 = None
        permute_120: "f16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_223, [1, 0]);  convert_element_type_223 = None
        addmm_31: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_222, view_162, permute_120);  convert_element_type_222 = None
        view_163: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [32, 50, 3072])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_66: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_163, 1.702)
        sigmoid_10: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_66);  mul_66 = None
        mul_67: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_10, view_163);  sigmoid_10 = view_163 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        convert_element_type_228: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_138, torch.float16);  primals_138 = None
        convert_element_type_229: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_137, torch.float16);  primals_137 = None
        view_164: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_67, [1600, 3072]);  mul_67 = None
        permute_121: "f16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_229, [1, 0]);  convert_element_type_229 = None
        addmm_32: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_228, view_164, permute_121);  convert_element_type_228 = None
        view_165: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_32, [32, 50, 768]);  addmm_32 = None
        add_79: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_76, view_165);  add_76 = view_165 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_79, [2], correction = 0, keepdim = True)
        getitem_145: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_23[0]
        getitem_146: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        add_80: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_145, 1e-05);  getitem_145 = None
        rsqrt_23: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        sub_23: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_79, getitem_146);  getitem_146 = None
        mul_68: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
        mul_69: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, primals_139)
        add_81: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_69, primals_140);  mul_69 = primals_140 = None
        permute_122: "f32[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.aten.permute.default(add_81, [1, 0, 2]);  add_81 = None
        convert_element_type_233: "f16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_141, torch.float16);  primals_141 = None
        convert_element_type_234: "f16[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_142, torch.float16);  primals_142 = None
        convert_element_type_235: "f16[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_122, torch.float16);  permute_122 = None
        permute_123: "f16[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_234, [1, 0]);  convert_element_type_234 = None
        clone_55: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_235, memory_format = torch.contiguous_format);  convert_element_type_235 = None
        view_166: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_55, [1600, 768]);  clone_55 = None
        mm_11: "f16[1600, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(view_166, permute_123)
        view_167: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(mm_11, [50, 32, 2304]);  mm_11 = None
        add_82: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.add.Tensor(view_167, convert_element_type_233);  view_167 = convert_element_type_233 = None
        view_168: "f16[50, 32, 3, 768][73728, 2304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(add_82, [50, 32, 3, 768]);  add_82 = None
        unsqueeze_12: "f16[1, 50, 32, 3, 768][3686400, 73728, 2304, 768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_168, 0);  view_168 = None
        permute_124: "f16[3, 50, 32, 1, 768][768, 73728, 2304, 3686400, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_12, [3, 1, 2, 0, 4]);  unsqueeze_12 = None
        squeeze_11: "f16[3, 50, 32, 768][768, 73728, 2304, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_124, -2);  permute_124 = None
        clone_56: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_11, memory_format = torch.contiguous_format);  squeeze_11 = None
        select_33: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_56, 0, 0)
        select_34: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_56, 0, 1)
        select_35: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.select.int(clone_56, 0, 2);  clone_56 = None
        view_169: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_33, [50, 384, 64]);  select_33 = None
        permute_125: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_169, [1, 0, 2]);  view_169 = None
        view_170: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_34, [50, 384, 64]);  select_34 = None
        permute_126: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_170, [1, 0, 2]);  view_170 = None
        view_171: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_35, [50, 384, 64]);  select_35 = None
        permute_127: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_171, [1, 0, 2]);  view_171 = None
        view_172: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_125, [32, 12, 50, 64]);  permute_125 = None
        view_173: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_126, [32, 12, 50, 64]);  permute_126 = None
        view_174: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(permute_127, [32, 12, 50, 64]);  permute_127 = None
        _scaled_dot_product_cudnn_attention_11 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(view_172, view_173, view_174, None, True)
        getitem_147: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_11[0]
        getitem_148: "f32[32, 12, 50, 1][600, 50, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_11[1]
        getitem_153: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_11[6]
        getitem_154: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_11[7];  _scaled_dot_product_cudnn_attention_11 = None
        permute_128: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_147, [2, 0, 1, 3])
        view_175: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_128, [1600, 768]);  permute_128 = None
        convert_element_type_238: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_144, torch.float16);  primals_144 = None
        convert_element_type_239: "f16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_143, torch.float16);  primals_143 = None
        permute_129: "f16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_239, [1, 0]);  convert_element_type_239 = None
        addmm_33: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_238, view_175, permute_129);  convert_element_type_238 = view_175 = None
        view_176: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [50, 32, 768]);  addmm_33 = None
        permute_130: "f16[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_176, [1, 0, 2]);  view_176 = None
        add_83: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_79, permute_130);  add_79 = permute_130 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
        getitem_156: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_24[0]
        getitem_157: "f32[32, 50, 1][50, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None
        add_84: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_156, 1e-05);  getitem_156 = None
        rsqrt_24: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        sub_24: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_83, getitem_157);  getitem_157 = None
        mul_70: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        mul_71: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, primals_145)
        add_85: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_71, primals_146);  mul_71 = primals_146 = None
        convert_element_type_243: "f16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_148, torch.float16);  primals_148 = None
        convert_element_type_244: "f16[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_147, torch.float16);  primals_147 = None
        convert_element_type_245: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_85, torch.float16);  add_85 = None
        view_177: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_245, [1600, 768]);  convert_element_type_245 = None
        permute_131: "f16[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_244, [1, 0]);  convert_element_type_244 = None
        addmm_34: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_243, view_177, permute_131);  convert_element_type_243 = None
        view_178: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [32, 50, 3072])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_72: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_178, 1.702)
        sigmoid_11: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_72);  mul_72 = None
        mul_73: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_11, view_178);  sigmoid_11 = view_178 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        convert_element_type_249: "f16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_150, torch.float16);  primals_150 = None
        convert_element_type_250: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_149, torch.float16);  primals_149 = None
        view_179: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_73, [1600, 3072]);  mul_73 = None
        permute_132: "f16[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(convert_element_type_250, [1, 0]);  convert_element_type_250 = None
        addmm_35: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_249, view_179, permute_132);  convert_element_type_249 = None
        view_180: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_35, [32, 50, 768]);  addmm_35 = None
        add_86: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_83, view_180);  add_83 = view_180 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:111 in forward, code: x = self.ln_post(x[:, 0, :])
        select_36: "f32[32, 768][38400, 1]cuda:0" = torch.ops.aten.select.int(add_86, 1, 0);  add_86 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/normalizations.py:18 in forward, code: output = nn.functional.layer_norm(
        clone_60: "f32[32, 768][768, 1]cuda:0" = torch.ops.aten.clone.default(select_36, memory_format = torch.contiguous_format)
        var_mean_25 = torch.ops.aten.var_mean.correction(clone_60, [1], correction = 0, keepdim = True)
        getitem_158: "f32[32, 1][1, 1]cuda:0" = var_mean_25[0]
        getitem_159: "f32[32, 1][1, 1]cuda:0" = var_mean_25[1];  var_mean_25 = None
        add_87: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_158, 1e-05);  getitem_158 = None
        rsqrt_25: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
        sub_25: "f32[32, 768][768, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_60, getitem_159);  clone_60 = None
        mul_74: "f32[32, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        mul_75: "f32[32, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_74, primals_151);  mul_74 = None
        add_88: "f32[32, 768][768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_75, primals_152);  mul_75 = primals_152 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:112 in forward, code: x = x @ self.projection
        convert_element_type_254: "f16[768, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_153, torch.float16);  primals_153 = None
        convert_element_type_255: "f16[32, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_88, torch.float16);  add_88 = None
        mm_12: "f16[32, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(convert_element_type_255, convert_element_type_254)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:118 in forward, code: embeddings = self.token_embedding(text)
        embedding: "f32[32, 77, 512][39424, 512, 1]cuda:0" = torch.ops.aten.embedding.default(primals_155, primals_154);  primals_155 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:119 in forward, code: embeddings = embeddings + self.positional_embedding
        add_89: "f32[32, 77, 512][39424, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(embedding, primals_156)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:120 in forward, code: embeddings = embeddings.permute(1, 0, 2)
        permute_133: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.permute.default(add_89, [1, 0, 2]);  add_89 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        clone_61: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_133, memory_format = torch.contiguous_format)
        var_mean_26 = torch.ops.aten.var_mean.correction(clone_61, [2], correction = 0, keepdim = True)
        getitem_160: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_26[0]
        getitem_161: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_26[1];  var_mean_26 = None
        add_90: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_160, 1e-05);  getitem_160 = None
        rsqrt_26: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        sub_26: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_61, getitem_161);  clone_61 = None
        mul_76: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
        mul_77: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_76, primals_158);  mul_76 = None
        add_91: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_77, primals_159);  mul_77 = primals_159 = None
        convert_element_type_258: "f16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_160, torch.float16);  primals_160 = None
        convert_element_type_259: "f16[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_161, torch.float16);  primals_161 = None
        convert_element_type_260: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_91, torch.float16);  add_91 = None
        view_181: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_260, [2464, 512]);  convert_element_type_260 = None
        permute_134: "f16[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_259, [1, 0]);  convert_element_type_259 = None
        addmm_36: "f16[2464, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_258, view_181, permute_134);  convert_element_type_258 = None
        view_182: "f16[77, 32, 1536][49152, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_36, [77, 32, 1536]);  addmm_36 = None
        view_183: "f16[77, 32, 3, 512][49152, 1536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_182, [77, 32, 3, 512]);  view_182 = None
        unsqueeze_13: "f16[1, 77, 32, 3, 512][3784704, 49152, 1536, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_183, 0);  view_183 = None
        permute_135: "f16[3, 77, 32, 1, 512][512, 49152, 1536, 3784704, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_13, [3, 1, 2, 0, 4]);  unsqueeze_13 = None
        squeeze_12: "f16[3, 77, 32, 512][512, 49152, 1536, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_135, -2);  permute_135 = None
        clone_62: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_12, memory_format = torch.contiguous_format);  squeeze_12 = None
        select_37: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_62, 0, 0)
        select_38: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_62, 0, 1)
        select_39: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_62, 0, 2);  clone_62 = None
        view_184: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_37, [77, 256, 64]);  select_37 = None
        permute_136: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_184, [1, 0, 2]);  view_184 = None
        view_185: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_38, [77, 256, 64]);  select_38 = None
        permute_137: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_185, [1, 0, 2]);  view_185 = None
        view_186: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_39, [77, 256, 64]);  select_39 = None
        permute_138: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_186, [1, 0, 2]);  view_186 = None
        view_187: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_136, [32, 8, 77, 64]);  permute_136 = None
        view_188: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_137, [32, 8, 77, 64]);  permute_137 = None
        view_189: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_138, [32, 8, 77, 64]);  permute_138 = None
        _scaled_dot_product_cudnn_attention_12 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(view_187, view_188, view_189, None, True, 0.0, True)
        getitem_162: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_12[0]
        getitem_163: "f32[32, 8, 77, 1][616, 77, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_12[1]
        getitem_168: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_12[6]
        getitem_169: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_12[7];  _scaled_dot_product_cudnn_attention_12 = None
        permute_139: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_162, [2, 0, 1, 3])
        view_190: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_139, [2464, 512]);  permute_139 = None
        convert_element_type_264: "f16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_163, torch.float16);  primals_163 = None
        convert_element_type_265: "f16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_162, torch.float16);  primals_162 = None
        permute_140: "f16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_265, [1, 0]);  convert_element_type_265 = None
        addmm_37: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_264, view_190, permute_140);  convert_element_type_264 = view_190 = None
        view_191: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_37, [77, 32, 512]);  addmm_37 = None
        add_92: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(permute_133, view_191);  permute_133 = view_191 = None
        clone_64: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(add_92, memory_format = torch.contiguous_format)
        var_mean_27 = torch.ops.aten.var_mean.correction(clone_64, [2], correction = 0, keepdim = True)
        getitem_171: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_27[0]
        getitem_172: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_27[1];  var_mean_27 = None
        add_93: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_171, 1e-05);  getitem_171 = None
        rsqrt_27: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        sub_27: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_64, getitem_172);  clone_64 = None
        mul_78: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
        mul_79: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_78, primals_164);  mul_78 = None
        add_94: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_79, primals_165);  mul_79 = primals_165 = None
        convert_element_type_269: "f16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_167, torch.float16);  primals_167 = None
        convert_element_type_270: "f16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_166, torch.float16);  primals_166 = None
        convert_element_type_271: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_94, torch.float16);  add_94 = None
        view_192: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_271, [2464, 512]);  convert_element_type_271 = None
        permute_141: "f16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_270, [1, 0]);  convert_element_type_270 = None
        addmm_38: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_269, view_192, permute_141);  convert_element_type_269 = None
        view_193: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [77, 32, 2048])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_80: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_193, 1.702)
        sigmoid_12: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_80);  mul_80 = None
        mul_81: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_12, view_193);  sigmoid_12 = view_193 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        convert_element_type_275: "f16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_169, torch.float16);  primals_169 = None
        convert_element_type_276: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_168, torch.float16);  primals_168 = None
        view_194: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(mul_81, [2464, 2048]);  mul_81 = None
        permute_142: "f16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_276, [1, 0]);  convert_element_type_276 = None
        addmm_39: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_275, view_194, permute_142);  convert_element_type_275 = None
        view_195: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_39, [77, 32, 512]);  addmm_39 = None
        add_95: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_92, view_195);  view_195 = None
        clone_67: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(add_95, memory_format = torch.contiguous_format)
        var_mean_28 = torch.ops.aten.var_mean.correction(clone_67, [2], correction = 0, keepdim = True)
        getitem_173: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_28[0]
        getitem_174: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_28[1];  var_mean_28 = None
        add_96: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_173, 1e-05);  getitem_173 = None
        rsqrt_28: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        sub_28: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_67, getitem_174);  clone_67 = None
        mul_82: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
        mul_83: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_82, primals_170);  mul_82 = None
        add_97: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_83, primals_171);  mul_83 = primals_171 = None
        convert_element_type_280: "f16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_172, torch.float16);  primals_172 = None
        convert_element_type_281: "f16[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_173, torch.float16);  primals_173 = None
        convert_element_type_282: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_97, torch.float16);  add_97 = None
        view_196: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_282, [2464, 512]);  convert_element_type_282 = None
        permute_143: "f16[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_281, [1, 0]);  convert_element_type_281 = None
        addmm_40: "f16[2464, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_280, view_196, permute_143);  convert_element_type_280 = None
        view_197: "f16[77, 32, 1536][49152, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [77, 32, 1536]);  addmm_40 = None
        view_198: "f16[77, 32, 3, 512][49152, 1536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_197, [77, 32, 3, 512]);  view_197 = None
        unsqueeze_14: "f16[1, 77, 32, 3, 512][3784704, 49152, 1536, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_198, 0);  view_198 = None
        permute_144: "f16[3, 77, 32, 1, 512][512, 49152, 1536, 3784704, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_14, [3, 1, 2, 0, 4]);  unsqueeze_14 = None
        squeeze_13: "f16[3, 77, 32, 512][512, 49152, 1536, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_144, -2);  permute_144 = None
        clone_68: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_13, memory_format = torch.contiguous_format);  squeeze_13 = None
        select_40: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_68, 0, 0)
        select_41: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_68, 0, 1)
        select_42: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_68, 0, 2);  clone_68 = None
        view_199: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_40, [77, 256, 64]);  select_40 = None
        permute_145: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_199, [1, 0, 2]);  view_199 = None
        view_200: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_41, [77, 256, 64]);  select_41 = None
        permute_146: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_200, [1, 0, 2]);  view_200 = None
        view_201: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_42, [77, 256, 64]);  select_42 = None
        permute_147: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_201, [1, 0, 2]);  view_201 = None
        view_202: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_145, [32, 8, 77, 64]);  permute_145 = None
        view_203: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_146, [32, 8, 77, 64]);  permute_146 = None
        view_204: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_147, [32, 8, 77, 64]);  permute_147 = None
        _scaled_dot_product_cudnn_attention_13 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(view_202, view_203, view_204, None, True, 0.0, True)
        getitem_175: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_13[0]
        getitem_176: "f32[32, 8, 77, 1][616, 77, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_13[1]
        getitem_181: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_13[6]
        getitem_182: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_13[7];  _scaled_dot_product_cudnn_attention_13 = None
        permute_148: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_175, [2, 0, 1, 3])
        view_205: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_148, [2464, 512]);  permute_148 = None
        convert_element_type_286: "f16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_175, torch.float16);  primals_175 = None
        convert_element_type_287: "f16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_174, torch.float16);  primals_174 = None
        permute_149: "f16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_287, [1, 0]);  convert_element_type_287 = None
        addmm_41: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_286, view_205, permute_149);  convert_element_type_286 = view_205 = None
        view_206: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_41, [77, 32, 512]);  addmm_41 = None
        add_98: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_95, view_206);  view_206 = None
        clone_70: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(add_98, memory_format = torch.contiguous_format)
        var_mean_29 = torch.ops.aten.var_mean.correction(clone_70, [2], correction = 0, keepdim = True)
        getitem_184: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_29[0]
        getitem_185: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_29[1];  var_mean_29 = None
        add_99: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_184, 1e-05);  getitem_184 = None
        rsqrt_29: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
        sub_29: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_70, getitem_185);  clone_70 = None
        mul_84: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
        mul_85: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_84, primals_176);  mul_84 = None
        add_100: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_85, primals_177);  mul_85 = primals_177 = None
        convert_element_type_291: "f16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_179, torch.float16);  primals_179 = None
        convert_element_type_292: "f16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_178, torch.float16);  primals_178 = None
        convert_element_type_293: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_100, torch.float16);  add_100 = None
        view_207: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_293, [2464, 512]);  convert_element_type_293 = None
        permute_150: "f16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_292, [1, 0]);  convert_element_type_292 = None
        addmm_42: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_291, view_207, permute_150);  convert_element_type_291 = None
        view_208: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [77, 32, 2048])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_86: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_208, 1.702)
        sigmoid_13: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_86);  mul_86 = None
        mul_87: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_13, view_208);  sigmoid_13 = view_208 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        convert_element_type_297: "f16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_181, torch.float16);  primals_181 = None
        convert_element_type_298: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_180, torch.float16);  primals_180 = None
        view_209: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(mul_87, [2464, 2048]);  mul_87 = None
        permute_151: "f16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_298, [1, 0]);  convert_element_type_298 = None
        addmm_43: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_297, view_209, permute_151);  convert_element_type_297 = None
        view_210: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_43, [77, 32, 512]);  addmm_43 = None
        add_101: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_98, view_210);  view_210 = None
        clone_73: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(add_101, memory_format = torch.contiguous_format)
        var_mean_30 = torch.ops.aten.var_mean.correction(clone_73, [2], correction = 0, keepdim = True)
        getitem_186: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_30[0]
        getitem_187: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_30[1];  var_mean_30 = None
        add_102: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_186, 1e-05);  getitem_186 = None
        rsqrt_30: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        sub_30: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_73, getitem_187);  clone_73 = None
        mul_88: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
        mul_89: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_88, primals_182);  mul_88 = None
        add_103: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_89, primals_183);  mul_89 = primals_183 = None
        convert_element_type_302: "f16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_184, torch.float16);  primals_184 = None
        convert_element_type_303: "f16[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_185, torch.float16);  primals_185 = None
        convert_element_type_304: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_103, torch.float16);  add_103 = None
        view_211: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_304, [2464, 512]);  convert_element_type_304 = None
        permute_152: "f16[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_303, [1, 0]);  convert_element_type_303 = None
        addmm_44: "f16[2464, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_302, view_211, permute_152);  convert_element_type_302 = None
        view_212: "f16[77, 32, 1536][49152, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_44, [77, 32, 1536]);  addmm_44 = None
        view_213: "f16[77, 32, 3, 512][49152, 1536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_212, [77, 32, 3, 512]);  view_212 = None
        unsqueeze_15: "f16[1, 77, 32, 3, 512][3784704, 49152, 1536, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_213, 0);  view_213 = None
        permute_153: "f16[3, 77, 32, 1, 512][512, 49152, 1536, 3784704, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_15, [3, 1, 2, 0, 4]);  unsqueeze_15 = None
        squeeze_14: "f16[3, 77, 32, 512][512, 49152, 1536, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_153, -2);  permute_153 = None
        clone_74: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_14, memory_format = torch.contiguous_format);  squeeze_14 = None
        select_43: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_74, 0, 0)
        select_44: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_74, 0, 1)
        select_45: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_74, 0, 2);  clone_74 = None
        view_214: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_43, [77, 256, 64]);  select_43 = None
        permute_154: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_214, [1, 0, 2]);  view_214 = None
        view_215: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_44, [77, 256, 64]);  select_44 = None
        permute_155: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_215, [1, 0, 2]);  view_215 = None
        view_216: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_45, [77, 256, 64]);  select_45 = None
        permute_156: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_216, [1, 0, 2]);  view_216 = None
        view_217: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_154, [32, 8, 77, 64]);  permute_154 = None
        view_218: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_155, [32, 8, 77, 64]);  permute_155 = None
        view_219: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_156, [32, 8, 77, 64]);  permute_156 = None
        _scaled_dot_product_cudnn_attention_14 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(view_217, view_218, view_219, None, True, 0.0, True)
        getitem_188: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_14[0]
        getitem_189: "f32[32, 8, 77, 1][616, 77, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_14[1]
        getitem_194: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_14[6]
        getitem_195: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_14[7];  _scaled_dot_product_cudnn_attention_14 = None
        permute_157: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_188, [2, 0, 1, 3])
        view_220: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_157, [2464, 512]);  permute_157 = None
        convert_element_type_308: "f16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_187, torch.float16);  primals_187 = None
        convert_element_type_309: "f16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_186, torch.float16);  primals_186 = None
        permute_158: "f16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_309, [1, 0]);  convert_element_type_309 = None
        addmm_45: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_308, view_220, permute_158);  convert_element_type_308 = view_220 = None
        view_221: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_45, [77, 32, 512]);  addmm_45 = None
        add_104: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_101, view_221);  view_221 = None
        clone_76: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(add_104, memory_format = torch.contiguous_format)
        var_mean_31 = torch.ops.aten.var_mean.correction(clone_76, [2], correction = 0, keepdim = True)
        getitem_197: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_31[0]
        getitem_198: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_31[1];  var_mean_31 = None
        add_105: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_197, 1e-05);  getitem_197 = None
        rsqrt_31: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
        sub_31: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_76, getitem_198);  clone_76 = None
        mul_90: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
        mul_91: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_90, primals_188);  mul_90 = None
        add_106: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_91, primals_189);  mul_91 = primals_189 = None
        convert_element_type_313: "f16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_191, torch.float16);  primals_191 = None
        convert_element_type_314: "f16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_190, torch.float16);  primals_190 = None
        convert_element_type_315: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_106, torch.float16);  add_106 = None
        view_222: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_315, [2464, 512]);  convert_element_type_315 = None
        permute_159: "f16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_314, [1, 0]);  convert_element_type_314 = None
        addmm_46: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_313, view_222, permute_159);  convert_element_type_313 = None
        view_223: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [77, 32, 2048])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_92: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_223, 1.702)
        sigmoid_14: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_92);  mul_92 = None
        mul_93: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_14, view_223);  sigmoid_14 = view_223 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        convert_element_type_319: "f16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_193, torch.float16);  primals_193 = None
        convert_element_type_320: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_192, torch.float16);  primals_192 = None
        view_224: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(mul_93, [2464, 2048]);  mul_93 = None
        permute_160: "f16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_320, [1, 0]);  convert_element_type_320 = None
        addmm_47: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_319, view_224, permute_160);  convert_element_type_319 = None
        view_225: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_47, [77, 32, 512]);  addmm_47 = None
        add_107: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_104, view_225);  view_225 = None
        clone_79: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(add_107, memory_format = torch.contiguous_format)
        var_mean_32 = torch.ops.aten.var_mean.correction(clone_79, [2], correction = 0, keepdim = True)
        getitem_199: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_32[0]
        getitem_200: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_32[1];  var_mean_32 = None
        add_108: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_199, 1e-05);  getitem_199 = None
        rsqrt_32: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
        sub_32: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_79, getitem_200);  clone_79 = None
        mul_94: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
        mul_95: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_94, primals_194);  mul_94 = None
        add_109: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_95, primals_195);  mul_95 = primals_195 = None
        convert_element_type_324: "f16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_196, torch.float16);  primals_196 = None
        convert_element_type_325: "f16[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_197, torch.float16);  primals_197 = None
        convert_element_type_326: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_109, torch.float16);  add_109 = None
        view_226: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_326, [2464, 512]);  convert_element_type_326 = None
        permute_161: "f16[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_325, [1, 0]);  convert_element_type_325 = None
        addmm_48: "f16[2464, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_324, view_226, permute_161);  convert_element_type_324 = None
        view_227: "f16[77, 32, 1536][49152, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_48, [77, 32, 1536]);  addmm_48 = None
        view_228: "f16[77, 32, 3, 512][49152, 1536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_227, [77, 32, 3, 512]);  view_227 = None
        unsqueeze_16: "f16[1, 77, 32, 3, 512][3784704, 49152, 1536, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_228, 0);  view_228 = None
        permute_162: "f16[3, 77, 32, 1, 512][512, 49152, 1536, 3784704, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_16, [3, 1, 2, 0, 4]);  unsqueeze_16 = None
        squeeze_15: "f16[3, 77, 32, 512][512, 49152, 1536, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_162, -2);  permute_162 = None
        clone_80: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_15, memory_format = torch.contiguous_format);  squeeze_15 = None
        select_46: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_80, 0, 0)
        select_47: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_80, 0, 1)
        select_48: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_80, 0, 2);  clone_80 = None
        view_229: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_46, [77, 256, 64]);  select_46 = None
        permute_163: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_229, [1, 0, 2]);  view_229 = None
        view_230: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_47, [77, 256, 64]);  select_47 = None
        permute_164: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_230, [1, 0, 2]);  view_230 = None
        view_231: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_48, [77, 256, 64]);  select_48 = None
        permute_165: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_231, [1, 0, 2]);  view_231 = None
        view_232: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_163, [32, 8, 77, 64]);  permute_163 = None
        view_233: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_164, [32, 8, 77, 64]);  permute_164 = None
        view_234: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_165, [32, 8, 77, 64]);  permute_165 = None
        _scaled_dot_product_cudnn_attention_15 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(view_232, view_233, view_234, None, True, 0.0, True)
        getitem_201: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_15[0]
        getitem_202: "f32[32, 8, 77, 1][616, 77, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_15[1]
        getitem_207: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_15[6]
        getitem_208: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_15[7];  _scaled_dot_product_cudnn_attention_15 = None
        permute_166: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_201, [2, 0, 1, 3])
        view_235: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_166, [2464, 512]);  permute_166 = None
        convert_element_type_330: "f16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_199, torch.float16);  primals_199 = None
        convert_element_type_331: "f16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_198, torch.float16);  primals_198 = None
        permute_167: "f16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_331, [1, 0]);  convert_element_type_331 = None
        addmm_49: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_330, view_235, permute_167);  convert_element_type_330 = view_235 = None
        view_236: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_49, [77, 32, 512]);  addmm_49 = None
        add_110: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_107, view_236);  view_236 = None
        clone_82: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(add_110, memory_format = torch.contiguous_format)
        var_mean_33 = torch.ops.aten.var_mean.correction(clone_82, [2], correction = 0, keepdim = True)
        getitem_210: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_33[0]
        getitem_211: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_33[1];  var_mean_33 = None
        add_111: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_210, 1e-05);  getitem_210 = None
        rsqrt_33: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
        sub_33: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_82, getitem_211);  clone_82 = None
        mul_96: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
        mul_97: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_96, primals_200);  mul_96 = None
        add_112: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_97, primals_201);  mul_97 = primals_201 = None
        convert_element_type_335: "f16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_203, torch.float16);  primals_203 = None
        convert_element_type_336: "f16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_202, torch.float16);  primals_202 = None
        convert_element_type_337: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_112, torch.float16);  add_112 = None
        view_237: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_337, [2464, 512]);  convert_element_type_337 = None
        permute_168: "f16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_336, [1, 0]);  convert_element_type_336 = None
        addmm_50: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_335, view_237, permute_168);  convert_element_type_335 = None
        view_238: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_50, [77, 32, 2048])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_98: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_238, 1.702)
        sigmoid_15: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_98);  mul_98 = None
        mul_99: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_15, view_238);  sigmoid_15 = view_238 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        convert_element_type_341: "f16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_205, torch.float16);  primals_205 = None
        convert_element_type_342: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_204, torch.float16);  primals_204 = None
        view_239: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(mul_99, [2464, 2048]);  mul_99 = None
        permute_169: "f16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_342, [1, 0]);  convert_element_type_342 = None
        addmm_51: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_341, view_239, permute_169);  convert_element_type_341 = None
        view_240: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_51, [77, 32, 512]);  addmm_51 = None
        add_113: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_110, view_240);  view_240 = None
        clone_85: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(add_113, memory_format = torch.contiguous_format)
        var_mean_34 = torch.ops.aten.var_mean.correction(clone_85, [2], correction = 0, keepdim = True)
        getitem_212: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_34[0]
        getitem_213: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_34[1];  var_mean_34 = None
        add_114: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_212, 1e-05);  getitem_212 = None
        rsqrt_34: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
        sub_34: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_85, getitem_213);  clone_85 = None
        mul_100: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
        mul_101: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_100, primals_206);  mul_100 = None
        add_115: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_101, primals_207);  mul_101 = primals_207 = None
        convert_element_type_346: "f16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_208, torch.float16);  primals_208 = None
        convert_element_type_347: "f16[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_209, torch.float16);  primals_209 = None
        convert_element_type_348: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_115, torch.float16);  add_115 = None
        view_241: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_348, [2464, 512]);  convert_element_type_348 = None
        permute_170: "f16[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_347, [1, 0]);  convert_element_type_347 = None
        addmm_52: "f16[2464, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_346, view_241, permute_170);  convert_element_type_346 = None
        view_242: "f16[77, 32, 1536][49152, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_52, [77, 32, 1536]);  addmm_52 = None
        view_243: "f16[77, 32, 3, 512][49152, 1536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_242, [77, 32, 3, 512]);  view_242 = None
        unsqueeze_17: "f16[1, 77, 32, 3, 512][3784704, 49152, 1536, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_243, 0);  view_243 = None
        permute_171: "f16[3, 77, 32, 1, 512][512, 49152, 1536, 3784704, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_17, [3, 1, 2, 0, 4]);  unsqueeze_17 = None
        squeeze_16: "f16[3, 77, 32, 512][512, 49152, 1536, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_171, -2);  permute_171 = None
        clone_86: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_16, memory_format = torch.contiguous_format);  squeeze_16 = None
        select_49: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_86, 0, 0)
        select_50: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_86, 0, 1)
        select_51: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_86, 0, 2);  clone_86 = None
        view_244: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_49, [77, 256, 64]);  select_49 = None
        permute_172: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_244, [1, 0, 2]);  view_244 = None
        view_245: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_50, [77, 256, 64]);  select_50 = None
        permute_173: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_245, [1, 0, 2]);  view_245 = None
        view_246: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_51, [77, 256, 64]);  select_51 = None
        permute_174: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_246, [1, 0, 2]);  view_246 = None
        view_247: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_172, [32, 8, 77, 64]);  permute_172 = None
        view_248: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_173, [32, 8, 77, 64]);  permute_173 = None
        view_249: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_174, [32, 8, 77, 64]);  permute_174 = None
        _scaled_dot_product_cudnn_attention_16 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(view_247, view_248, view_249, None, True, 0.0, True)
        getitem_214: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_16[0]
        getitem_215: "f32[32, 8, 77, 1][616, 77, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_16[1]
        getitem_220: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_16[6]
        getitem_221: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_16[7];  _scaled_dot_product_cudnn_attention_16 = None
        permute_175: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_214, [2, 0, 1, 3])
        view_250: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_175, [2464, 512]);  permute_175 = None
        convert_element_type_352: "f16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_211, torch.float16);  primals_211 = None
        convert_element_type_353: "f16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_210, torch.float16);  primals_210 = None
        permute_176: "f16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_353, [1, 0]);  convert_element_type_353 = None
        addmm_53: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_352, view_250, permute_176);  convert_element_type_352 = view_250 = None
        view_251: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_53, [77, 32, 512]);  addmm_53 = None
        add_116: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_113, view_251);  view_251 = None
        clone_88: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(add_116, memory_format = torch.contiguous_format)
        var_mean_35 = torch.ops.aten.var_mean.correction(clone_88, [2], correction = 0, keepdim = True)
        getitem_223: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_35[0]
        getitem_224: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_35[1];  var_mean_35 = None
        add_117: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_223, 1e-05);  getitem_223 = None
        rsqrt_35: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
        sub_35: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_88, getitem_224);  clone_88 = None
        mul_102: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
        mul_103: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_102, primals_212);  mul_102 = None
        add_118: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_103, primals_213);  mul_103 = primals_213 = None
        convert_element_type_357: "f16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_215, torch.float16);  primals_215 = None
        convert_element_type_358: "f16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_214, torch.float16);  primals_214 = None
        convert_element_type_359: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_118, torch.float16);  add_118 = None
        view_252: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_359, [2464, 512]);  convert_element_type_359 = None
        permute_177: "f16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_358, [1, 0]);  convert_element_type_358 = None
        addmm_54: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_357, view_252, permute_177);  convert_element_type_357 = None
        view_253: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_54, [77, 32, 2048])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_104: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_253, 1.702)
        sigmoid_16: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_104);  mul_104 = None
        mul_105: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_16, view_253);  sigmoid_16 = view_253 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        convert_element_type_363: "f16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_217, torch.float16);  primals_217 = None
        convert_element_type_364: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_216, torch.float16);  primals_216 = None
        view_254: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(mul_105, [2464, 2048]);  mul_105 = None
        permute_178: "f16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_364, [1, 0]);  convert_element_type_364 = None
        addmm_55: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_363, view_254, permute_178);  convert_element_type_363 = None
        view_255: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_55, [77, 32, 512]);  addmm_55 = None
        add_119: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_116, view_255);  view_255 = None
        clone_91: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(add_119, memory_format = torch.contiguous_format)
        var_mean_36 = torch.ops.aten.var_mean.correction(clone_91, [2], correction = 0, keepdim = True)
        getitem_225: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_36[0]
        getitem_226: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_36[1];  var_mean_36 = None
        add_120: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_225, 1e-05);  getitem_225 = None
        rsqrt_36: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
        sub_36: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_91, getitem_226);  clone_91 = None
        mul_106: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
        mul_107: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_106, primals_218);  mul_106 = None
        add_121: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_107, primals_219);  mul_107 = primals_219 = None
        convert_element_type_368: "f16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_220, torch.float16);  primals_220 = None
        convert_element_type_369: "f16[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_221, torch.float16);  primals_221 = None
        convert_element_type_370: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_121, torch.float16);  add_121 = None
        view_256: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_370, [2464, 512]);  convert_element_type_370 = None
        permute_179: "f16[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_369, [1, 0]);  convert_element_type_369 = None
        addmm_56: "f16[2464, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_368, view_256, permute_179);  convert_element_type_368 = None
        view_257: "f16[77, 32, 1536][49152, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_56, [77, 32, 1536]);  addmm_56 = None
        view_258: "f16[77, 32, 3, 512][49152, 1536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_257, [77, 32, 3, 512]);  view_257 = None
        unsqueeze_18: "f16[1, 77, 32, 3, 512][3784704, 49152, 1536, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_258, 0);  view_258 = None
        permute_180: "f16[3, 77, 32, 1, 512][512, 49152, 1536, 3784704, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_18, [3, 1, 2, 0, 4]);  unsqueeze_18 = None
        squeeze_17: "f16[3, 77, 32, 512][512, 49152, 1536, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_180, -2);  permute_180 = None
        clone_92: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_17, memory_format = torch.contiguous_format);  squeeze_17 = None
        select_52: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_92, 0, 0)
        select_53: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_92, 0, 1)
        select_54: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_92, 0, 2);  clone_92 = None
        view_259: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_52, [77, 256, 64]);  select_52 = None
        permute_181: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_259, [1, 0, 2]);  view_259 = None
        view_260: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_53, [77, 256, 64]);  select_53 = None
        permute_182: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_260, [1, 0, 2]);  view_260 = None
        view_261: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_54, [77, 256, 64]);  select_54 = None
        permute_183: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_261, [1, 0, 2]);  view_261 = None
        view_262: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_181, [32, 8, 77, 64]);  permute_181 = None
        view_263: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_182, [32, 8, 77, 64]);  permute_182 = None
        view_264: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_183, [32, 8, 77, 64]);  permute_183 = None
        _scaled_dot_product_cudnn_attention_17 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(view_262, view_263, view_264, None, True, 0.0, True)
        getitem_227: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_17[0]
        getitem_228: "f32[32, 8, 77, 1][616, 77, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_17[1]
        getitem_233: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_17[6]
        getitem_234: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_17[7];  _scaled_dot_product_cudnn_attention_17 = None
        permute_184: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_227, [2, 0, 1, 3])
        view_265: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_184, [2464, 512]);  permute_184 = None
        convert_element_type_374: "f16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_223, torch.float16);  primals_223 = None
        convert_element_type_375: "f16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_222, torch.float16);  primals_222 = None
        permute_185: "f16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_375, [1, 0]);  convert_element_type_375 = None
        addmm_57: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_374, view_265, permute_185);  convert_element_type_374 = view_265 = None
        view_266: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_57, [77, 32, 512]);  addmm_57 = None
        add_122: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_119, view_266);  view_266 = None
        clone_94: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(add_122, memory_format = torch.contiguous_format)
        var_mean_37 = torch.ops.aten.var_mean.correction(clone_94, [2], correction = 0, keepdim = True)
        getitem_236: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_37[0]
        getitem_237: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_37[1];  var_mean_37 = None
        add_123: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_236, 1e-05);  getitem_236 = None
        rsqrt_37: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
        sub_37: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_94, getitem_237);  clone_94 = None
        mul_108: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
        mul_109: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_108, primals_224);  mul_108 = None
        add_124: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_109, primals_225);  mul_109 = primals_225 = None
        convert_element_type_379: "f16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_227, torch.float16);  primals_227 = None
        convert_element_type_380: "f16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_226, torch.float16);  primals_226 = None
        convert_element_type_381: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_124, torch.float16);  add_124 = None
        view_267: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_381, [2464, 512]);  convert_element_type_381 = None
        permute_186: "f16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_380, [1, 0]);  convert_element_type_380 = None
        addmm_58: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_379, view_267, permute_186);  convert_element_type_379 = None
        view_268: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_58, [77, 32, 2048])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_110: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_268, 1.702)
        sigmoid_17: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_110);  mul_110 = None
        mul_111: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_17, view_268);  sigmoid_17 = view_268 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        convert_element_type_385: "f16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_229, torch.float16);  primals_229 = None
        convert_element_type_386: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_228, torch.float16);  primals_228 = None
        view_269: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(mul_111, [2464, 2048]);  mul_111 = None
        permute_187: "f16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_386, [1, 0]);  convert_element_type_386 = None
        addmm_59: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_385, view_269, permute_187);  convert_element_type_385 = None
        view_270: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_59, [77, 32, 512]);  addmm_59 = None
        add_125: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_122, view_270);  view_270 = None
        clone_97: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(add_125, memory_format = torch.contiguous_format)
        var_mean_38 = torch.ops.aten.var_mean.correction(clone_97, [2], correction = 0, keepdim = True)
        getitem_238: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_38[0]
        getitem_239: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_38[1];  var_mean_38 = None
        add_126: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_238, 1e-05);  getitem_238 = None
        rsqrt_38: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        sub_38: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_97, getitem_239);  clone_97 = None
        mul_112: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
        mul_113: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_112, primals_230);  mul_112 = None
        add_127: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_113, primals_231);  mul_113 = primals_231 = None
        convert_element_type_390: "f16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_232, torch.float16);  primals_232 = None
        convert_element_type_391: "f16[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_233, torch.float16);  primals_233 = None
        convert_element_type_392: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_127, torch.float16);  add_127 = None
        view_271: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_392, [2464, 512]);  convert_element_type_392 = None
        permute_188: "f16[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_391, [1, 0]);  convert_element_type_391 = None
        addmm_60: "f16[2464, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_390, view_271, permute_188);  convert_element_type_390 = None
        view_272: "f16[77, 32, 1536][49152, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_60, [77, 32, 1536]);  addmm_60 = None
        view_273: "f16[77, 32, 3, 512][49152, 1536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_272, [77, 32, 3, 512]);  view_272 = None
        unsqueeze_19: "f16[1, 77, 32, 3, 512][3784704, 49152, 1536, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_273, 0);  view_273 = None
        permute_189: "f16[3, 77, 32, 1, 512][512, 49152, 1536, 3784704, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_19, [3, 1, 2, 0, 4]);  unsqueeze_19 = None
        squeeze_18: "f16[3, 77, 32, 512][512, 49152, 1536, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_189, -2);  permute_189 = None
        clone_98: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_18, memory_format = torch.contiguous_format);  squeeze_18 = None
        select_55: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_98, 0, 0)
        select_56: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_98, 0, 1)
        select_57: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_98, 0, 2);  clone_98 = None
        view_274: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_55, [77, 256, 64]);  select_55 = None
        permute_190: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_274, [1, 0, 2]);  view_274 = None
        view_275: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_56, [77, 256, 64]);  select_56 = None
        permute_191: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_275, [1, 0, 2]);  view_275 = None
        view_276: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_57, [77, 256, 64]);  select_57 = None
        permute_192: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_276, [1, 0, 2]);  view_276 = None
        view_277: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_190, [32, 8, 77, 64]);  permute_190 = None
        view_278: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_191, [32, 8, 77, 64]);  permute_191 = None
        view_279: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_192, [32, 8, 77, 64]);  permute_192 = None
        _scaled_dot_product_cudnn_attention_18 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(view_277, view_278, view_279, None, True, 0.0, True)
        getitem_240: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_18[0]
        getitem_241: "f32[32, 8, 77, 1][616, 77, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_18[1]
        getitem_246: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_18[6]
        getitem_247: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_18[7];  _scaled_dot_product_cudnn_attention_18 = None
        permute_193: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_240, [2, 0, 1, 3])
        view_280: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_193, [2464, 512]);  permute_193 = None
        convert_element_type_396: "f16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_235, torch.float16);  primals_235 = None
        convert_element_type_397: "f16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_234, torch.float16);  primals_234 = None
        permute_194: "f16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_397, [1, 0]);  convert_element_type_397 = None
        addmm_61: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_396, view_280, permute_194);  convert_element_type_396 = view_280 = None
        view_281: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_61, [77, 32, 512]);  addmm_61 = None
        add_128: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_125, view_281);  view_281 = None
        clone_100: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(add_128, memory_format = torch.contiguous_format)
        var_mean_39 = torch.ops.aten.var_mean.correction(clone_100, [2], correction = 0, keepdim = True)
        getitem_249: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_39[0]
        getitem_250: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_39[1];  var_mean_39 = None
        add_129: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_249, 1e-05);  getitem_249 = None
        rsqrt_39: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
        sub_39: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_100, getitem_250);  clone_100 = None
        mul_114: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
        mul_115: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_114, primals_236);  mul_114 = None
        add_130: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_115, primals_237);  mul_115 = primals_237 = None
        convert_element_type_401: "f16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_239, torch.float16);  primals_239 = None
        convert_element_type_402: "f16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_238, torch.float16);  primals_238 = None
        convert_element_type_403: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_130, torch.float16);  add_130 = None
        view_282: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_403, [2464, 512]);  convert_element_type_403 = None
        permute_195: "f16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_402, [1, 0]);  convert_element_type_402 = None
        addmm_62: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_401, view_282, permute_195);  convert_element_type_401 = None
        view_283: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_62, [77, 32, 2048])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_116: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_283, 1.702)
        sigmoid_18: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_116);  mul_116 = None
        mul_117: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_18, view_283);  sigmoid_18 = view_283 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        convert_element_type_407: "f16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_241, torch.float16);  primals_241 = None
        convert_element_type_408: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_240, torch.float16);  primals_240 = None
        view_284: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(mul_117, [2464, 2048]);  mul_117 = None
        permute_196: "f16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_408, [1, 0]);  convert_element_type_408 = None
        addmm_63: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_407, view_284, permute_196);  convert_element_type_407 = None
        view_285: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_63, [77, 32, 512]);  addmm_63 = None
        add_131: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_128, view_285);  view_285 = None
        clone_103: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(add_131, memory_format = torch.contiguous_format)
        var_mean_40 = torch.ops.aten.var_mean.correction(clone_103, [2], correction = 0, keepdim = True)
        getitem_251: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_40[0]
        getitem_252: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_40[1];  var_mean_40 = None
        add_132: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_251, 1e-05);  getitem_251 = None
        rsqrt_40: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
        sub_40: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_103, getitem_252);  clone_103 = None
        mul_118: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
        mul_119: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_118, primals_242);  mul_118 = None
        add_133: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_119, primals_243);  mul_119 = primals_243 = None
        convert_element_type_412: "f16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_244, torch.float16);  primals_244 = None
        convert_element_type_413: "f16[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_245, torch.float16);  primals_245 = None
        convert_element_type_414: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_133, torch.float16);  add_133 = None
        view_286: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_414, [2464, 512]);  convert_element_type_414 = None
        permute_197: "f16[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_413, [1, 0]);  convert_element_type_413 = None
        addmm_64: "f16[2464, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_412, view_286, permute_197);  convert_element_type_412 = None
        view_287: "f16[77, 32, 1536][49152, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_64, [77, 32, 1536]);  addmm_64 = None
        view_288: "f16[77, 32, 3, 512][49152, 1536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_287, [77, 32, 3, 512]);  view_287 = None
        unsqueeze_20: "f16[1, 77, 32, 3, 512][3784704, 49152, 1536, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_288, 0);  view_288 = None
        permute_198: "f16[3, 77, 32, 1, 512][512, 49152, 1536, 3784704, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_20, [3, 1, 2, 0, 4]);  unsqueeze_20 = None
        squeeze_19: "f16[3, 77, 32, 512][512, 49152, 1536, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_198, -2);  permute_198 = None
        clone_104: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_19, memory_format = torch.contiguous_format);  squeeze_19 = None
        select_58: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_104, 0, 0)
        select_59: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_104, 0, 1)
        select_60: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_104, 0, 2);  clone_104 = None
        view_289: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_58, [77, 256, 64]);  select_58 = None
        permute_199: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_289, [1, 0, 2]);  view_289 = None
        view_290: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_59, [77, 256, 64]);  select_59 = None
        permute_200: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_290, [1, 0, 2]);  view_290 = None
        view_291: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_60, [77, 256, 64]);  select_60 = None
        permute_201: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_291, [1, 0, 2]);  view_291 = None
        view_292: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_199, [32, 8, 77, 64]);  permute_199 = None
        view_293: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_200, [32, 8, 77, 64]);  permute_200 = None
        view_294: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_201, [32, 8, 77, 64]);  permute_201 = None
        _scaled_dot_product_cudnn_attention_19 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(view_292, view_293, view_294, None, True, 0.0, True)
        getitem_253: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_19[0]
        getitem_254: "f32[32, 8, 77, 1][616, 77, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_19[1]
        getitem_259: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_19[6]
        getitem_260: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_19[7];  _scaled_dot_product_cudnn_attention_19 = None
        permute_202: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_253, [2, 0, 1, 3])
        view_295: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_202, [2464, 512]);  permute_202 = None
        convert_element_type_418: "f16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_247, torch.float16);  primals_247 = None
        convert_element_type_419: "f16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_246, torch.float16);  primals_246 = None
        permute_203: "f16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_419, [1, 0]);  convert_element_type_419 = None
        addmm_65: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_418, view_295, permute_203);  convert_element_type_418 = view_295 = None
        view_296: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_65, [77, 32, 512]);  addmm_65 = None
        add_134: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_131, view_296);  view_296 = None
        clone_106: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(add_134, memory_format = torch.contiguous_format)
        var_mean_41 = torch.ops.aten.var_mean.correction(clone_106, [2], correction = 0, keepdim = True)
        getitem_262: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_41[0]
        getitem_263: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_41[1];  var_mean_41 = None
        add_135: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_262, 1e-05);  getitem_262 = None
        rsqrt_41: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
        sub_41: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_106, getitem_263);  clone_106 = None
        mul_120: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
        mul_121: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_120, primals_248);  mul_120 = None
        add_136: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_121, primals_249);  mul_121 = primals_249 = None
        convert_element_type_423: "f16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_251, torch.float16);  primals_251 = None
        convert_element_type_424: "f16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_250, torch.float16);  primals_250 = None
        convert_element_type_425: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_136, torch.float16);  add_136 = None
        view_297: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_425, [2464, 512]);  convert_element_type_425 = None
        permute_204: "f16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_424, [1, 0]);  convert_element_type_424 = None
        addmm_66: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_423, view_297, permute_204);  convert_element_type_423 = None
        view_298: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_66, [77, 32, 2048])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_122: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_298, 1.702)
        sigmoid_19: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_122);  mul_122 = None
        mul_123: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_19, view_298);  sigmoid_19 = view_298 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        convert_element_type_429: "f16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_253, torch.float16);  primals_253 = None
        convert_element_type_430: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_252, torch.float16);  primals_252 = None
        view_299: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(mul_123, [2464, 2048]);  mul_123 = None
        permute_205: "f16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_430, [1, 0]);  convert_element_type_430 = None
        addmm_67: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_429, view_299, permute_205);  convert_element_type_429 = None
        view_300: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_67, [77, 32, 512]);  addmm_67 = None
        add_137: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_134, view_300);  view_300 = None
        clone_109: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(add_137, memory_format = torch.contiguous_format)
        var_mean_42 = torch.ops.aten.var_mean.correction(clone_109, [2], correction = 0, keepdim = True)
        getitem_264: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_42[0]
        getitem_265: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_42[1];  var_mean_42 = None
        add_138: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_264, 1e-05);  getitem_264 = None
        rsqrt_42: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
        sub_42: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_109, getitem_265);  clone_109 = None
        mul_124: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
        mul_125: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_124, primals_254);  mul_124 = None
        add_139: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_125, primals_255);  mul_125 = primals_255 = None
        convert_element_type_434: "f16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_256, torch.float16);  primals_256 = None
        convert_element_type_435: "f16[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_257, torch.float16);  primals_257 = None
        convert_element_type_436: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_139, torch.float16);  add_139 = None
        view_301: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_436, [2464, 512]);  convert_element_type_436 = None
        permute_206: "f16[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_435, [1, 0]);  convert_element_type_435 = None
        addmm_68: "f16[2464, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_434, view_301, permute_206);  convert_element_type_434 = None
        view_302: "f16[77, 32, 1536][49152, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_68, [77, 32, 1536]);  addmm_68 = None
        view_303: "f16[77, 32, 3, 512][49152, 1536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_302, [77, 32, 3, 512]);  view_302 = None
        unsqueeze_21: "f16[1, 77, 32, 3, 512][3784704, 49152, 1536, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_303, 0);  view_303 = None
        permute_207: "f16[3, 77, 32, 1, 512][512, 49152, 1536, 3784704, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_21, [3, 1, 2, 0, 4]);  unsqueeze_21 = None
        squeeze_20: "f16[3, 77, 32, 512][512, 49152, 1536, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_207, -2);  permute_207 = None
        clone_110: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_20, memory_format = torch.contiguous_format);  squeeze_20 = None
        select_61: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_110, 0, 0)
        select_62: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_110, 0, 1)
        select_63: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_110, 0, 2);  clone_110 = None
        view_304: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_61, [77, 256, 64]);  select_61 = None
        permute_208: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_304, [1, 0, 2]);  view_304 = None
        view_305: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_62, [77, 256, 64]);  select_62 = None
        permute_209: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_305, [1, 0, 2]);  view_305 = None
        view_306: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_63, [77, 256, 64]);  select_63 = None
        permute_210: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_306, [1, 0, 2]);  view_306 = None
        view_307: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_208, [32, 8, 77, 64]);  permute_208 = None
        view_308: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_209, [32, 8, 77, 64]);  permute_209 = None
        view_309: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_210, [32, 8, 77, 64]);  permute_210 = None
        _scaled_dot_product_cudnn_attention_20 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(view_307, view_308, view_309, None, True, 0.0, True)
        getitem_266: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_20[0]
        getitem_267: "f32[32, 8, 77, 1][616, 77, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_20[1]
        getitem_272: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_20[6]
        getitem_273: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_20[7];  _scaled_dot_product_cudnn_attention_20 = None
        permute_211: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_266, [2, 0, 1, 3])
        view_310: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_211, [2464, 512]);  permute_211 = None
        convert_element_type_440: "f16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_259, torch.float16);  primals_259 = None
        convert_element_type_441: "f16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_258, torch.float16);  primals_258 = None
        permute_212: "f16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_441, [1, 0]);  convert_element_type_441 = None
        addmm_69: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_440, view_310, permute_212);  convert_element_type_440 = view_310 = None
        view_311: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_69, [77, 32, 512]);  addmm_69 = None
        add_140: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_137, view_311);  view_311 = None
        clone_112: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(add_140, memory_format = torch.contiguous_format)
        var_mean_43 = torch.ops.aten.var_mean.correction(clone_112, [2], correction = 0, keepdim = True)
        getitem_275: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_43[0]
        getitem_276: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_43[1];  var_mean_43 = None
        add_141: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_275, 1e-05);  getitem_275 = None
        rsqrt_43: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
        sub_43: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_112, getitem_276);  clone_112 = None
        mul_126: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
        mul_127: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_126, primals_260);  mul_126 = None
        add_142: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_127, primals_261);  mul_127 = primals_261 = None
        convert_element_type_445: "f16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_263, torch.float16);  primals_263 = None
        convert_element_type_446: "f16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_262, torch.float16);  primals_262 = None
        convert_element_type_447: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_142, torch.float16);  add_142 = None
        view_312: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_447, [2464, 512]);  convert_element_type_447 = None
        permute_213: "f16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_446, [1, 0]);  convert_element_type_446 = None
        addmm_70: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_445, view_312, permute_213);  convert_element_type_445 = None
        view_313: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_70, [77, 32, 2048])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_128: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_313, 1.702)
        sigmoid_20: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_128);  mul_128 = None
        mul_129: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_20, view_313);  sigmoid_20 = view_313 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        convert_element_type_451: "f16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_265, torch.float16);  primals_265 = None
        convert_element_type_452: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_264, torch.float16);  primals_264 = None
        view_314: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(mul_129, [2464, 2048]);  mul_129 = None
        permute_214: "f16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_452, [1, 0]);  convert_element_type_452 = None
        addmm_71: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_451, view_314, permute_214);  convert_element_type_451 = None
        view_315: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_71, [77, 32, 512]);  addmm_71 = None
        add_143: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_140, view_315);  view_315 = None
        clone_115: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(add_143, memory_format = torch.contiguous_format)
        var_mean_44 = torch.ops.aten.var_mean.correction(clone_115, [2], correction = 0, keepdim = True)
        getitem_277: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_44[0]
        getitem_278: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_44[1];  var_mean_44 = None
        add_144: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_277, 1e-05);  getitem_277 = None
        rsqrt_44: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
        sub_44: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_115, getitem_278);  clone_115 = None
        mul_130: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
        mul_131: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_130, primals_266);  mul_130 = None
        add_145: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_131, primals_267);  mul_131 = primals_267 = None
        convert_element_type_456: "f16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_268, torch.float16);  primals_268 = None
        convert_element_type_457: "f16[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_269, torch.float16);  primals_269 = None
        convert_element_type_458: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_145, torch.float16);  add_145 = None
        view_316: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_458, [2464, 512]);  convert_element_type_458 = None
        permute_215: "f16[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_457, [1, 0]);  convert_element_type_457 = None
        addmm_72: "f16[2464, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_456, view_316, permute_215);  convert_element_type_456 = None
        view_317: "f16[77, 32, 1536][49152, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_72, [77, 32, 1536]);  addmm_72 = None
        view_318: "f16[77, 32, 3, 512][49152, 1536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_317, [77, 32, 3, 512]);  view_317 = None
        unsqueeze_22: "f16[1, 77, 32, 3, 512][3784704, 49152, 1536, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_318, 0);  view_318 = None
        permute_216: "f16[3, 77, 32, 1, 512][512, 49152, 1536, 3784704, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_22, [3, 1, 2, 0, 4]);  unsqueeze_22 = None
        squeeze_21: "f16[3, 77, 32, 512][512, 49152, 1536, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_216, -2);  permute_216 = None
        clone_116: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_21, memory_format = torch.contiguous_format);  squeeze_21 = None
        select_64: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_116, 0, 0)
        select_65: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_116, 0, 1)
        select_66: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_116, 0, 2);  clone_116 = None
        view_319: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_64, [77, 256, 64]);  select_64 = None
        permute_217: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_319, [1, 0, 2]);  view_319 = None
        view_320: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_65, [77, 256, 64]);  select_65 = None
        permute_218: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_320, [1, 0, 2]);  view_320 = None
        view_321: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_66, [77, 256, 64]);  select_66 = None
        permute_219: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_321, [1, 0, 2]);  view_321 = None
        view_322: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_217, [32, 8, 77, 64]);  permute_217 = None
        view_323: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_218, [32, 8, 77, 64]);  permute_218 = None
        view_324: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_219, [32, 8, 77, 64]);  permute_219 = None
        _scaled_dot_product_cudnn_attention_21 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(view_322, view_323, view_324, None, True, 0.0, True)
        getitem_279: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_21[0]
        getitem_280: "f32[32, 8, 77, 1][616, 77, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_21[1]
        getitem_285: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_21[6]
        getitem_286: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_21[7];  _scaled_dot_product_cudnn_attention_21 = None
        permute_220: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_279, [2, 0, 1, 3])
        view_325: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_220, [2464, 512]);  permute_220 = None
        convert_element_type_462: "f16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_271, torch.float16);  primals_271 = None
        convert_element_type_463: "f16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_270, torch.float16);  primals_270 = None
        permute_221: "f16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_463, [1, 0]);  convert_element_type_463 = None
        addmm_73: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_462, view_325, permute_221);  convert_element_type_462 = view_325 = None
        view_326: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_73, [77, 32, 512]);  addmm_73 = None
        add_146: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_143, view_326);  view_326 = None
        clone_118: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(add_146, memory_format = torch.contiguous_format)
        var_mean_45 = torch.ops.aten.var_mean.correction(clone_118, [2], correction = 0, keepdim = True)
        getitem_288: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_45[0]
        getitem_289: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_45[1];  var_mean_45 = None
        add_147: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_288, 1e-05);  getitem_288 = None
        rsqrt_45: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
        sub_45: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_118, getitem_289);  clone_118 = None
        mul_132: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
        mul_133: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_132, primals_272);  mul_132 = None
        add_148: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_133, primals_273);  mul_133 = primals_273 = None
        convert_element_type_467: "f16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_275, torch.float16);  primals_275 = None
        convert_element_type_468: "f16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_274, torch.float16);  primals_274 = None
        convert_element_type_469: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_148, torch.float16);  add_148 = None
        view_327: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_469, [2464, 512]);  convert_element_type_469 = None
        permute_222: "f16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_468, [1, 0]);  convert_element_type_468 = None
        addmm_74: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_467, view_327, permute_222);  convert_element_type_467 = None
        view_328: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_74, [77, 32, 2048])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_134: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_328, 1.702)
        sigmoid_21: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_134);  mul_134 = None
        mul_135: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_21, view_328);  sigmoid_21 = view_328 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        convert_element_type_473: "f16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_277, torch.float16);  primals_277 = None
        convert_element_type_474: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_276, torch.float16);  primals_276 = None
        view_329: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(mul_135, [2464, 2048]);  mul_135 = None
        permute_223: "f16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_474, [1, 0]);  convert_element_type_474 = None
        addmm_75: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_473, view_329, permute_223);  convert_element_type_473 = None
        view_330: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_75, [77, 32, 512]);  addmm_75 = None
        add_149: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_146, view_330);  view_330 = None
        clone_121: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(add_149, memory_format = torch.contiguous_format)
        var_mean_46 = torch.ops.aten.var_mean.correction(clone_121, [2], correction = 0, keepdim = True)
        getitem_290: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_46[0]
        getitem_291: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_46[1];  var_mean_46 = None
        add_150: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_290, 1e-05);  getitem_290 = None
        rsqrt_46: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
        sub_46: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_121, getitem_291);  clone_121 = None
        mul_136: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
        mul_137: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_136, primals_278);  mul_136 = None
        add_151: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_137, primals_279);  mul_137 = primals_279 = None
        convert_element_type_478: "f16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_280, torch.float16);  primals_280 = None
        convert_element_type_479: "f16[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_281, torch.float16);  primals_281 = None
        convert_element_type_480: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_151, torch.float16);  add_151 = None
        view_331: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_480, [2464, 512]);  convert_element_type_480 = None
        permute_224: "f16[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_479, [1, 0]);  convert_element_type_479 = None
        addmm_76: "f16[2464, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_478, view_331, permute_224);  convert_element_type_478 = None
        view_332: "f16[77, 32, 1536][49152, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_76, [77, 32, 1536]);  addmm_76 = None
        view_333: "f16[77, 32, 3, 512][49152, 1536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_332, [77, 32, 3, 512]);  view_332 = None
        unsqueeze_23: "f16[1, 77, 32, 3, 512][3784704, 49152, 1536, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_333, 0);  view_333 = None
        permute_225: "f16[3, 77, 32, 1, 512][512, 49152, 1536, 3784704, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_23, [3, 1, 2, 0, 4]);  unsqueeze_23 = None
        squeeze_22: "f16[3, 77, 32, 512][512, 49152, 1536, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_225, -2);  permute_225 = None
        clone_122: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_22, memory_format = torch.contiguous_format);  squeeze_22 = None
        select_67: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_122, 0, 0)
        select_68: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_122, 0, 1)
        select_69: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_122, 0, 2);  clone_122 = None
        view_334: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_67, [77, 256, 64]);  select_67 = None
        permute_226: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_334, [1, 0, 2]);  view_334 = None
        view_335: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_68, [77, 256, 64]);  select_68 = None
        permute_227: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_335, [1, 0, 2]);  view_335 = None
        view_336: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_69, [77, 256, 64]);  select_69 = None
        permute_228: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_336, [1, 0, 2]);  view_336 = None
        view_337: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_226, [32, 8, 77, 64]);  permute_226 = None
        view_338: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_227, [32, 8, 77, 64]);  permute_227 = None
        view_339: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_228, [32, 8, 77, 64]);  permute_228 = None
        _scaled_dot_product_cudnn_attention_22 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(view_337, view_338, view_339, None, True, 0.0, True)
        getitem_292: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_22[0]
        getitem_293: "f32[32, 8, 77, 1][616, 77, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_22[1]
        getitem_298: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_22[6]
        getitem_299: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_22[7];  _scaled_dot_product_cudnn_attention_22 = None
        permute_229: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_292, [2, 0, 1, 3])
        view_340: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_229, [2464, 512]);  permute_229 = None
        convert_element_type_484: "f16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_283, torch.float16);  primals_283 = None
        convert_element_type_485: "f16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_282, torch.float16);  primals_282 = None
        permute_230: "f16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_485, [1, 0]);  convert_element_type_485 = None
        addmm_77: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_484, view_340, permute_230);  convert_element_type_484 = view_340 = None
        view_341: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_77, [77, 32, 512]);  addmm_77 = None
        add_152: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_149, view_341);  view_341 = None
        clone_124: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(add_152, memory_format = torch.contiguous_format)
        var_mean_47 = torch.ops.aten.var_mean.correction(clone_124, [2], correction = 0, keepdim = True)
        getitem_301: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_47[0]
        getitem_302: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_47[1];  var_mean_47 = None
        add_153: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_301, 1e-05);  getitem_301 = None
        rsqrt_47: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
        sub_47: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_124, getitem_302);  clone_124 = None
        mul_138: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
        mul_139: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_138, primals_284);  mul_138 = None
        add_154: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_139, primals_285);  mul_139 = primals_285 = None
        convert_element_type_489: "f16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_287, torch.float16);  primals_287 = None
        convert_element_type_490: "f16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_286, torch.float16);  primals_286 = None
        convert_element_type_491: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_154, torch.float16);  add_154 = None
        view_342: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_491, [2464, 512]);  convert_element_type_491 = None
        permute_231: "f16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_490, [1, 0]);  convert_element_type_490 = None
        addmm_78: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_489, view_342, permute_231);  convert_element_type_489 = None
        view_343: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_78, [77, 32, 2048])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_140: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_343, 1.702)
        sigmoid_22: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_140);  mul_140 = None
        mul_141: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_22, view_343);  sigmoid_22 = view_343 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        convert_element_type_495: "f16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_289, torch.float16);  primals_289 = None
        convert_element_type_496: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_288, torch.float16);  primals_288 = None
        view_344: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(mul_141, [2464, 2048]);  mul_141 = None
        permute_232: "f16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_496, [1, 0]);  convert_element_type_496 = None
        addmm_79: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_495, view_344, permute_232);  convert_element_type_495 = None
        view_345: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_79, [77, 32, 512]);  addmm_79 = None
        add_155: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_152, view_345);  view_345 = None
        clone_127: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(add_155, memory_format = torch.contiguous_format)
        var_mean_48 = torch.ops.aten.var_mean.correction(clone_127, [2], correction = 0, keepdim = True)
        getitem_303: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_48[0]
        getitem_304: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_48[1];  var_mean_48 = None
        add_156: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_303, 1e-05);  getitem_303 = None
        rsqrt_48: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
        sub_48: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_127, getitem_304);  clone_127 = None
        mul_142: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
        mul_143: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_142, primals_290);  mul_142 = None
        add_157: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_143, primals_291);  mul_143 = primals_291 = None
        convert_element_type_500: "f16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_292, torch.float16);  primals_292 = None
        convert_element_type_501: "f16[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_293, torch.float16);  primals_293 = None
        convert_element_type_502: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_157, torch.float16);  add_157 = None
        view_346: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_502, [2464, 512]);  convert_element_type_502 = None
        permute_233: "f16[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_501, [1, 0]);  convert_element_type_501 = None
        addmm_80: "f16[2464, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_500, view_346, permute_233);  convert_element_type_500 = None
        view_347: "f16[77, 32, 1536][49152, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_80, [77, 32, 1536]);  addmm_80 = None
        view_348: "f16[77, 32, 3, 512][49152, 1536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_347, [77, 32, 3, 512]);  view_347 = None
        unsqueeze_24: "f16[1, 77, 32, 3, 512][3784704, 49152, 1536, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_348, 0);  view_348 = None
        permute_234: "f16[3, 77, 32, 1, 512][512, 49152, 1536, 3784704, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_24, [3, 1, 2, 0, 4]);  unsqueeze_24 = None
        squeeze_23: "f16[3, 77, 32, 512][512, 49152, 1536, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_234, -2);  permute_234 = None
        clone_128: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_23, memory_format = torch.contiguous_format);  squeeze_23 = None
        select_70: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_128, 0, 0)
        select_71: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_128, 0, 1)
        select_72: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.select.int(clone_128, 0, 2);  clone_128 = None
        view_349: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_70, [77, 256, 64]);  select_70 = None
        permute_235: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_349, [1, 0, 2]);  view_349 = None
        view_350: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_71, [77, 256, 64]);  select_71 = None
        permute_236: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_350, [1, 0, 2]);  view_350 = None
        view_351: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(select_72, [77, 256, 64]);  select_72 = None
        permute_237: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_351, [1, 0, 2]);  view_351 = None
        view_352: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_235, [32, 8, 77, 64]);  permute_235 = None
        view_353: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_236, [32, 8, 77, 64]);  permute_236 = None
        view_354: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_237, [32, 8, 77, 64]);  permute_237 = None
        _scaled_dot_product_cudnn_attention_23 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(view_352, view_353, view_354, None, True, 0.0, True)
        getitem_305: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_23[0]
        getitem_306: "f32[32, 8, 77, 1][616, 77, 1, 1]cuda:0" = _scaled_dot_product_cudnn_attention_23[1]
        getitem_311: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_23[6]
        getitem_312: "i64[][]cuda:0" = _scaled_dot_product_cudnn_attention_23[7];  _scaled_dot_product_cudnn_attention_23 = None
        permute_238: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_305, [2, 0, 1, 3])
        view_355: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_238, [2464, 512]);  permute_238 = None
        convert_element_type_506: "f16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_295, torch.float16);  primals_295 = None
        convert_element_type_507: "f16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_294, torch.float16);  primals_294 = None
        permute_239: "f16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_507, [1, 0]);  convert_element_type_507 = None
        addmm_81: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_506, view_355, permute_239);  convert_element_type_506 = view_355 = None
        view_356: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_81, [77, 32, 512]);  addmm_81 = None
        add_158: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_155, view_356);  view_356 = None
        clone_130: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(add_158, memory_format = torch.contiguous_format)
        var_mean_49 = torch.ops.aten.var_mean.correction(clone_130, [2], correction = 0, keepdim = True)
        getitem_314: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_49[0]
        getitem_315: "f32[77, 32, 1][32, 1, 1]cuda:0" = var_mean_49[1];  var_mean_49 = None
        add_159: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_314, 1e-05);  getitem_314 = None
        rsqrt_49: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
        sub_49: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_130, getitem_315);  clone_130 = None
        mul_144: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
        mul_145: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_144, primals_296);  mul_144 = None
        add_160: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_145, primals_297);  mul_145 = primals_297 = None
        convert_element_type_511: "f16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_299, torch.float16);  primals_299 = None
        convert_element_type_512: "f16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_298, torch.float16);  primals_298 = None
        convert_element_type_513: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_160, torch.float16);  add_160 = None
        view_357: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_513, [2464, 512]);  convert_element_type_513 = None
        permute_240: "f16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_512, [1, 0]);  convert_element_type_512 = None
        addmm_82: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_511, view_357, permute_240);  convert_element_type_511 = None
        view_358: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_82, [77, 32, 2048])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_146: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_358, 1.702)
        sigmoid_23: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_146);  mul_146 = None
        mul_147: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_23, view_358);  sigmoid_23 = view_358 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        convert_element_type_517: "f16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_301, torch.float16);  primals_301 = None
        convert_element_type_518: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_300, torch.float16);  primals_300 = None
        view_359: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(mul_147, [2464, 2048]);  mul_147 = None
        permute_241: "f16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_518, [1, 0]);  convert_element_type_518 = None
        addmm_83: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_517, view_359, permute_241);  convert_element_type_517 = None
        view_360: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_83, [77, 32, 512]);  addmm_83 = None
        add_161: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_158, view_360);  view_360 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:124 in forward, code: embeddings = torch.permute(embeddings, (1, 0, 2))
        permute_242: "f32[32, 77, 512][39424, 512, 1]cuda:0" = torch.ops.aten.permute.default(add_161, [1, 0, 2]);  add_161 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/normalizations.py:18 in forward, code: output = nn.functional.layer_norm(
        var_mean_50 = torch.ops.aten.var_mean.correction(permute_242, [2], correction = 0, keepdim = True)
        getitem_316: "f32[32, 77, 1][77, 1, 1]cuda:0" = var_mean_50[0]
        getitem_317: "f32[32, 77, 1][77, 1, 1]cuda:0" = var_mean_50[1];  var_mean_50 = None
        add_162: "f32[32, 77, 1][77, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_316, 1e-05);  getitem_316 = None
        rsqrt_50: "f32[32, 77, 1][77, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
        sub_50: "f32[32, 77, 512][39424, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute_242, getitem_317);  permute_242 = getitem_317 = None
        mul_148: "f32[32, 77, 512][39424, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
        mul_149: "f32[32, 77, 512][39424, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_148, primals_302)
        add_163: "f32[32, 77, 512][39424, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_149, primals_303);  mul_149 = primals_303 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:131 in forward, code: hidden_state[torch.arange(hidden_state.shape[0]), text.argmax(dim=-1)]
        iota_default: "i64[32][1]cuda:0" = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        argmax: "i64[32][1]cuda:0" = torch.ops.aten.argmax.default(primals_154, -1)
        index: "f32[32, 512][512, 1]cuda:0" = torch.ops.aten.index.Tensor(add_163, [iota_default, argmax]);  add_163 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:130 in forward, code: projected_embeddings = self.projection(
        convert_element_type_522: "f16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_304, torch.float16);  primals_304 = None
        convert_element_type_523: "f16[32, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(index, torch.float16);  index = None
        permute_243: "f16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_522, [1, 0]);  convert_element_type_522 = None
        mm_13: "f16[32, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(convert_element_type_523, permute_243)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/model.py:72 in forward, code: embeddings_a = F.normalize(embeddings_a)
        convert_element_type_526: "f32[32, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_12, torch.float32)
        pow_1: "f32[32, 512][512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_526, 2.0);  convert_element_type_526 = None
        sum_1: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_1, [1], True);  pow_1 = None
        pow_2: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_1, 0.5);  sum_1 = None
        clamp_min: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.clamp_min.default(pow_2, 1e-12)
        expand_1: "f32[32, 512][1, 0]cuda:0" = torch.ops.aten.expand.default(clamp_min, [32, 512]);  clamp_min = None
        div: "f32[32, 512][512, 1]cuda:0" = torch.ops.aten.div.Tensor(mm_12, expand_1);  expand_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/model.py:73 in forward, code: embeddings_b = F.normalize(embeddings_b)
        convert_element_type_527: "f32[32, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_13, torch.float32)
        pow_3: "f32[32, 512][512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_527, 2.0);  convert_element_type_527 = None
        sum_2: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(pow_3, [1], True);  pow_3 = None
        pow_4: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(sum_2, 0.5);  sum_2 = None
        clamp_min_1: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.clamp_min.default(pow_4, 1e-12)
        expand_2: "f32[32, 512][1, 0]cuda:0" = torch.ops.aten.expand.default(clamp_min_1, [32, 512]);  clamp_min_1 = None
        div_1: "f32[32, 512][512, 1]cuda:0" = torch.ops.aten.div.Tensor(mm_13, expand_2);  expand_2 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:130 in forward, code: projected_embeddings = self.projection(
        permute_246: "f16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/normalizations.py:18 in forward, code: output = nn.functional.layer_norm(
        div_10: "f32[32, 77, 1][77, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_50, 512);  rsqrt_50 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        permute_249: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_241, [1, 0]);  permute_241 = None
        permute_253: "f16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None
        sub_55: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_158, getitem_315);  add_158 = getitem_315 = None
        mul_166: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_49);  sub_55 = None
        div_11: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_49, 512);  rsqrt_49 = None
        permute_257: "f16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
        permute_266: "f16[1536, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
        sub_58: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_155, getitem_304);  add_155 = getitem_304 = None
        mul_173: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_48);  sub_58 = None
        div_12: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_48, 512);  rsqrt_48 = None
        permute_270: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
        permute_274: "f16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None
        sub_62: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_152, getitem_302);  add_152 = getitem_302 = None
        mul_185: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_47);  sub_62 = None
        div_13: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_47, 512);  rsqrt_47 = None
        permute_278: "f16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_230, [1, 0]);  permute_230 = None
        permute_287: "f16[1536, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_224, [1, 0]);  permute_224 = None
        sub_65: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_149, getitem_291);  add_149 = getitem_291 = None
        mul_192: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_46);  sub_65 = None
        div_14: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_46, 512);  rsqrt_46 = None
        permute_291: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_223, [1, 0]);  permute_223 = None
        permute_295: "f16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
        sub_69: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_146, getitem_289);  add_146 = getitem_289 = None
        mul_204: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_45);  sub_69 = None
        div_15: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_45, 512);  rsqrt_45 = None
        permute_299: "f16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_221, [1, 0]);  permute_221 = None
        permute_308: "f16[1536, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_215, [1, 0]);  permute_215 = None
        sub_72: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_143, getitem_278);  add_143 = getitem_278 = None
        mul_211: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_44);  sub_72 = None
        div_16: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_44, 512);  rsqrt_44 = None
        permute_312: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_214, [1, 0]);  permute_214 = None
        permute_316: "f16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_213, [1, 0]);  permute_213 = None
        sub_76: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_140, getitem_276);  add_140 = getitem_276 = None
        mul_223: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_43);  sub_76 = None
        div_17: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_43, 512);  rsqrt_43 = None
        permute_320: "f16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_212, [1, 0]);  permute_212 = None
        permute_329: "f16[1536, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
        sub_79: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_137, getitem_265);  add_137 = getitem_265 = None
        mul_230: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_42);  sub_79 = None
        div_18: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_42, 512);  rsqrt_42 = None
        permute_333: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_205, [1, 0]);  permute_205 = None
        permute_337: "f16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_204, [1, 0]);  permute_204 = None
        sub_83: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_134, getitem_263);  add_134 = getitem_263 = None
        mul_242: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_41);  sub_83 = None
        div_19: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_41, 512);  rsqrt_41 = None
        permute_341: "f16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_203, [1, 0]);  permute_203 = None
        permute_350: "f16[1536, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
        sub_86: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_131, getitem_252);  add_131 = getitem_252 = None
        mul_249: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_40);  sub_86 = None
        div_20: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_40, 512);  rsqrt_40 = None
        permute_354: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
        permute_358: "f16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_195, [1, 0]);  permute_195 = None
        sub_90: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_128, getitem_250);  add_128 = getitem_250 = None
        mul_261: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_39);  sub_90 = None
        div_21: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_39, 512);  rsqrt_39 = None
        permute_362: "f16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_194, [1, 0]);  permute_194 = None
        permute_371: "f16[1536, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_188, [1, 0]);  permute_188 = None
        sub_93: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_125, getitem_239);  add_125 = getitem_239 = None
        mul_268: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_93, rsqrt_38);  sub_93 = None
        div_22: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_38, 512);  rsqrt_38 = None
        permute_375: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None
        permute_379: "f16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
        sub_97: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_122, getitem_237);  add_122 = getitem_237 = None
        mul_280: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_97, rsqrt_37);  sub_97 = None
        div_23: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_37, 512);  rsqrt_37 = None
        permute_383: "f16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
        permute_392: "f16[1536, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_179, [1, 0]);  permute_179 = None
        sub_100: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_119, getitem_226);  add_119 = getitem_226 = None
        mul_287: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_100, rsqrt_36);  sub_100 = None
        div_24: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_36, 512);  rsqrt_36 = None
        permute_396: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
        permute_400: "f16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
        sub_104: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_116, getitem_224);  add_116 = getitem_224 = None
        mul_299: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_104, rsqrt_35);  sub_104 = None
        div_25: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_35, 512);  rsqrt_35 = None
        permute_404: "f16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
        permute_413: "f16[1536, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None
        sub_107: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_113, getitem_213);  add_113 = getitem_213 = None
        mul_306: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_107, rsqrt_34);  sub_107 = None
        div_26: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_34, 512);  rsqrt_34 = None
        permute_417: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_169, [1, 0]);  permute_169 = None
        permute_421: "f16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_168, [1, 0]);  permute_168 = None
        sub_111: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_110, getitem_211);  add_110 = getitem_211 = None
        mul_318: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_111, rsqrt_33);  sub_111 = None
        div_27: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_33, 512);  rsqrt_33 = None
        permute_425: "f16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
        permute_434: "f16[1536, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None
        sub_114: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_107, getitem_200);  add_107 = getitem_200 = None
        mul_325: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_114, rsqrt_32);  sub_114 = None
        div_28: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_32, 512);  rsqrt_32 = None
        permute_438: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_160, [1, 0]);  permute_160 = None
        permute_442: "f16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_159, [1, 0]);  permute_159 = None
        sub_118: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_104, getitem_198);  add_104 = getitem_198 = None
        mul_337: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_118, rsqrt_31);  sub_118 = None
        div_29: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_31, 512);  rsqrt_31 = None
        permute_446: "f16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_158, [1, 0]);  permute_158 = None
        permute_455: "f16[1536, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
        sub_121: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_101, getitem_187);  add_101 = getitem_187 = None
        mul_344: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_121, rsqrt_30);  sub_121 = None
        div_30: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_30, 512);  rsqrt_30 = None
        permute_459: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
        permute_463: "f16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_150, [1, 0]);  permute_150 = None
        sub_125: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_98, getitem_185);  add_98 = getitem_185 = None
        mul_356: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_125, rsqrt_29);  sub_125 = None
        div_31: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_29, 512);  rsqrt_29 = None
        permute_467: "f16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_149, [1, 0]);  permute_149 = None
        permute_476: "f16[1536, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
        sub_128: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_95, getitem_174);  add_95 = getitem_174 = None
        mul_363: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_128, rsqrt_28);  sub_128 = None
        div_32: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_28, 512);  rsqrt_28 = None
        permute_480: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
        permute_484: "f16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
        sub_132: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_92, getitem_172);  add_92 = getitem_172 = None
        mul_375: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_132, rsqrt_27);  sub_132 = None
        div_33: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_27, 512);  rsqrt_27 = None
        permute_488: "f16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
        permute_497: "f16[1536, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:112 in forward, code: x = x @ self.projection
        permute_502: "f16[768, 32][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_255, [1, 0]);  convert_element_type_255 = None
        permute_503: "f16[512, 768][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_254, [1, 0]);  convert_element_type_254 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/normalizations.py:18 in forward, code: output = nn.functional.layer_norm(
        sub_138: "f32[32, 768][768, 1]cuda:0" = torch.ops.aten.sub.Tensor(select_36, getitem_159);  select_36 = getitem_159 = None
        mul_389: "f32[32, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_138, rsqrt_25);  sub_138 = None
        div_35: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_25, 768);  rsqrt_25 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        permute_504: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
        permute_508: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
        div_36: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
        permute_513: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
        permute_524: "f16[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
        div_37: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
        permute_527: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        permute_531: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
        div_38: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
        permute_536: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
        permute_547: "f16[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
        div_39: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
        permute_550: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
        permute_554: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
        div_40: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
        permute_559: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
        permute_570: "f16[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
        div_41: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
        permute_573: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
        permute_577: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
        div_42: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
        permute_582: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        permute_593: "f16[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
        div_43: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
        permute_596: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
        permute_600: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        div_44: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
        permute_605: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        permute_616: "f16[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
        div_45: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
        permute_619: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        permute_623: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        div_46: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
        permute_628: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
        permute_639: "f16[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
        div_47: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
        permute_642: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        permute_646: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        div_48: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
        permute_651: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        permute_662: "f16[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        div_49: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
        permute_665: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        permute_669: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        div_50: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
        permute_674: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        permute_685: "f16[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        div_51: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
        permute_688: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        permute_692: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        div_52: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
        permute_697: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        permute_708: "f16[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        div_53: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
        permute_711: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        permute_715: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        div_54: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
        permute_720: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        permute_731: "f16[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        div_55: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
        permute_734: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        permute_738: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        div_56: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
        permute_743: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        permute_754: "f16[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        div_57: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
        permute_757: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        permute_761: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        div_58: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
        permute_766: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        permute_777: "f16[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        return (div, div_1, primals_4, primals_5, primals_6, primals_7, primals_13, primals_19, primals_25, primals_31, primals_37, primals_43, primals_49, primals_55, primals_61, primals_67, primals_73, primals_79, primals_85, primals_91, primals_97, primals_103, primals_109, primals_115, primals_121, primals_127, primals_133, primals_139, primals_145, primals_151, primals_154, primals_156, primals_158, primals_164, primals_170, primals_176, primals_182, primals_188, primals_194, primals_200, primals_206, primals_212, primals_218, primals_224, primals_230, primals_236, primals_242, primals_248, primals_254, primals_260, primals_266, primals_272, primals_278, primals_284, primals_290, primals_296, primals_302, convert_element_type, convert_element_type_1, cat, getitem_1, rsqrt, getitem_3, rsqrt_1, view_1, view_7, view_8, view_9, getitem_4, getitem_5, getitem_10, getitem_11, mul_4, view_12, addmm_1, view_14, mul_8, view_16, view_22, view_23, view_24, getitem_17, getitem_18, getitem_23, getitem_24, mul_10, view_27, addmm_4, view_29, mul_14, view_31, view_37, view_38, view_39, getitem_30, getitem_31, getitem_36, getitem_37, mul_16, view_42, addmm_7, view_44, mul_20, view_46, view_52, view_53, view_54, getitem_43, getitem_44, getitem_49, getitem_50, mul_22, view_57, addmm_10, view_59, mul_26, view_61, view_67, view_68, view_69, getitem_56, getitem_57, getitem_62, getitem_63, mul_28, view_72, addmm_13, view_74, mul_32, view_76, view_82, view_83, view_84, getitem_69, getitem_70, getitem_75, getitem_76, mul_34, view_87, addmm_16, view_89, mul_38, view_91, view_97, view_98, view_99, getitem_82, getitem_83, getitem_88, getitem_89, mul_40, view_102, addmm_19, view_104, mul_44, view_106, view_112, view_113, view_114, getitem_95, getitem_96, getitem_101, getitem_102, mul_46, view_117, addmm_22, view_119, mul_50, view_121, view_127, view_128, view_129, getitem_108, getitem_109, getitem_114, getitem_115, mul_52, view_132, addmm_25, view_134, mul_56, view_136, view_142, view_143, view_144, getitem_121, getitem_122, getitem_127, getitem_128, mul_58, view_147, addmm_28, view_149, mul_62, view_151, view_157, view_158, view_159, getitem_134, getitem_135, getitem_140, getitem_141, mul_64, view_162, addmm_31, view_164, mul_68, view_166, view_172, view_173, view_174, getitem_147, getitem_148, getitem_153, getitem_154, mul_70, view_177, addmm_34, view_179, mm_12, embedding, getitem_161, rsqrt_26, view_181, view_187, view_188, view_189, getitem_162, getitem_163, getitem_168, getitem_169, view_192, addmm_38, view_194, view_196, view_202, view_203, view_204, getitem_175, getitem_176, getitem_181, getitem_182, view_207, addmm_42, view_209, view_211, view_217, view_218, view_219, getitem_188, getitem_189, getitem_194, getitem_195, view_222, addmm_46, view_224, view_226, view_232, view_233, view_234, getitem_201, getitem_202, getitem_207, getitem_208, view_237, addmm_50, view_239, view_241, view_247, view_248, view_249, getitem_214, getitem_215, getitem_220, getitem_221, view_252, addmm_54, view_254, view_256, view_262, view_263, view_264, getitem_227, getitem_228, getitem_233, getitem_234, view_267, addmm_58, view_269, view_271, view_277, view_278, view_279, getitem_240, getitem_241, getitem_246, getitem_247, view_282, addmm_62, view_284, view_286, view_292, view_293, view_294, getitem_253, getitem_254, getitem_259, getitem_260, view_297, addmm_66, view_299, view_301, view_307, view_308, view_309, getitem_266, getitem_267, getitem_272, getitem_273, view_312, addmm_70, view_314, view_316, view_322, view_323, view_324, getitem_279, getitem_280, getitem_285, getitem_286, view_327, addmm_74, view_329, view_331, view_337, view_338, view_339, getitem_292, getitem_293, getitem_298, getitem_299, view_342, addmm_78, view_344, view_346, view_352, view_353, view_354, getitem_305, getitem_306, getitem_311, getitem_312, view_357, addmm_82, view_359, mul_148, iota_default, argmax, convert_element_type_523, mm_13, pow_2, pow_4, permute_246, div_10, permute_249, permute_253, mul_166, div_11, permute_257, permute_266, mul_173, div_12, permute_270, permute_274, mul_185, div_13, permute_278, permute_287, mul_192, div_14, permute_291, permute_295, mul_204, div_15, permute_299, permute_308, mul_211, div_16, permute_312, permute_316, mul_223, div_17, permute_320, permute_329, mul_230, div_18, permute_333, permute_337, mul_242, div_19, permute_341, permute_350, mul_249, div_20, permute_354, permute_358, mul_261, div_21, permute_362, permute_371, mul_268, div_22, permute_375, permute_379, mul_280, div_23, permute_383, permute_392, mul_287, div_24, permute_396, permute_400, mul_299, div_25, permute_404, permute_413, mul_306, div_26, permute_417, permute_421, mul_318, div_27, permute_425, permute_434, mul_325, div_28, permute_438, permute_442, mul_337, div_29, permute_446, permute_455, mul_344, div_30, permute_459, permute_463, mul_356, div_31, permute_467, permute_476, mul_363, div_32, permute_480, permute_484, mul_375, div_33, permute_488, permute_497, permute_502, permute_503, mul_389, div_35, permute_504, permute_508, div_36, permute_513, permute_524, div_37, permute_527, permute_531, div_38, permute_536, permute_547, div_39, permute_550, permute_554, div_40, permute_559, permute_570, div_41, permute_573, permute_577, div_42, permute_582, permute_593, div_43, permute_596, permute_600, div_44, permute_605, permute_616, div_45, permute_619, permute_623, div_46, permute_628, permute_639, div_47, permute_642, permute_646, div_48, permute_651, permute_662, div_49, permute_665, permute_669, div_50, permute_674, permute_685, div_51, permute_688, permute_692, div_52, permute_697, permute_708, div_53, permute_711, permute_715, div_54, permute_720, permute_731, div_55, permute_734, permute_738, div_56, permute_743, permute_754, div_57, permute_757, permute_761, div_58, permute_766, permute_777)
