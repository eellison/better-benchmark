class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[24, 3, 3, 3][27, 9, 3, 1]cuda:0", primals_2: "f32[128, 3, 224, 224][150528, 50176, 224, 1]cuda:0", primals_3: "i64[][]cuda:0", primals_4: "f32[24][1]cuda:0", primals_5: "f32[24][1]cuda:0", primals_6: "f32[24][1]cuda:0", primals_7: "f32[24][1]cuda:0", primals_8: "f32[24, 1, 3, 3][9, 9, 3, 1]cuda:0", primals_9: "i64[][]cuda:0", primals_10: "f32[24][1]cuda:0", primals_11: "f32[24][1]cuda:0", primals_12: "f32[24][1]cuda:0", primals_13: "f32[24][1]cuda:0", primals_14: "f32[58, 24, 1, 1][24, 1, 1, 1]cuda:0", primals_15: "i64[][]cuda:0", primals_16: "f32[58][1]cuda:0", primals_17: "f32[58][1]cuda:0", primals_18: "f32[58][1]cuda:0", primals_19: "f32[58][1]cuda:0", primals_20: "f32[58, 24, 1, 1][24, 1, 1, 1]cuda:0", primals_21: "i64[][]cuda:0", primals_22: "f32[58][1]cuda:0", primals_23: "f32[58][1]cuda:0", primals_24: "f32[58][1]cuda:0", primals_25: "f32[58][1]cuda:0", primals_26: "f32[58, 1, 3, 3][9, 9, 3, 1]cuda:0", primals_27: "i64[][]cuda:0", primals_28: "f32[58][1]cuda:0", primals_29: "f32[58][1]cuda:0", primals_30: "f32[58][1]cuda:0", primals_31: "f32[58][1]cuda:0", primals_32: "f32[58, 58, 1, 1][58, 1, 1, 1]cuda:0", primals_33: "i64[][]cuda:0", primals_34: "f32[58][1]cuda:0", primals_35: "f32[58][1]cuda:0", primals_36: "f32[58][1]cuda:0", primals_37: "f32[58][1]cuda:0", primals_38: "f32[58, 58, 1, 1][58, 1, 1, 1]cuda:0", primals_39: "i64[][]cuda:0", primals_40: "f32[58][1]cuda:0", primals_41: "f32[58][1]cuda:0", primals_42: "f32[58][1]cuda:0", primals_43: "f32[58][1]cuda:0", primals_44: "f32[58, 1, 3, 3][9, 9, 3, 1]cuda:0", primals_45: "i64[][]cuda:0", primals_46: "f32[58][1]cuda:0", primals_47: "f32[58][1]cuda:0", primals_48: "f32[58][1]cuda:0", primals_49: "f32[58][1]cuda:0", primals_50: "f32[58, 58, 1, 1][58, 1, 1, 1]cuda:0", primals_51: "i64[][]cuda:0", primals_52: "f32[58][1]cuda:0", primals_53: "f32[58][1]cuda:0", primals_54: "f32[58][1]cuda:0", primals_55: "f32[58][1]cuda:0", primals_56: "f32[58, 58, 1, 1][58, 1, 1, 1]cuda:0", primals_57: "i64[][]cuda:0", primals_58: "f32[58][1]cuda:0", primals_59: "f32[58][1]cuda:0", primals_60: "f32[58][1]cuda:0", primals_61: "f32[58][1]cuda:0", primals_62: "f32[58, 1, 3, 3][9, 9, 3, 1]cuda:0", primals_63: "i64[][]cuda:0", primals_64: "f32[58][1]cuda:0", primals_65: "f32[58][1]cuda:0", primals_66: "f32[58][1]cuda:0", primals_67: "f32[58][1]cuda:0", primals_68: "f32[58, 58, 1, 1][58, 1, 1, 1]cuda:0", primals_69: "i64[][]cuda:0", primals_70: "f32[58][1]cuda:0", primals_71: "f32[58][1]cuda:0", primals_72: "f32[58][1]cuda:0", primals_73: "f32[58][1]cuda:0", primals_74: "f32[58, 58, 1, 1][58, 1, 1, 1]cuda:0", primals_75: "i64[][]cuda:0", primals_76: "f32[58][1]cuda:0", primals_77: "f32[58][1]cuda:0", primals_78: "f32[58][1]cuda:0", primals_79: "f32[58][1]cuda:0", primals_80: "f32[58, 1, 3, 3][9, 9, 3, 1]cuda:0", primals_81: "i64[][]cuda:0", primals_82: "f32[58][1]cuda:0", primals_83: "f32[58][1]cuda:0", primals_84: "f32[58][1]cuda:0", primals_85: "f32[58][1]cuda:0", primals_86: "f32[58, 58, 1, 1][58, 1, 1, 1]cuda:0", primals_87: "i64[][]cuda:0", primals_88: "f32[58][1]cuda:0", primals_89: "f32[58][1]cuda:0", primals_90: "f32[58][1]cuda:0", primals_91: "f32[58][1]cuda:0", primals_92: "f32[116, 1, 3, 3][9, 9, 3, 1]cuda:0", primals_93: "i64[][]cuda:0", primals_94: "f32[116][1]cuda:0", primals_95: "f32[116][1]cuda:0", primals_96: "f32[116][1]cuda:0", primals_97: "f32[116][1]cuda:0", primals_98: "f32[116, 116, 1, 1][116, 1, 1, 1]cuda:0", primals_99: "i64[][]cuda:0", primals_100: "f32[116][1]cuda:0", primals_101: "f32[116][1]cuda:0", primals_102: "f32[116][1]cuda:0", primals_103: "f32[116][1]cuda:0", primals_104: "f32[116, 116, 1, 1][116, 1, 1, 1]cuda:0", primals_105: "i64[][]cuda:0", primals_106: "f32[116][1]cuda:0", primals_107: "f32[116][1]cuda:0", primals_108: "f32[116][1]cuda:0", primals_109: "f32[116][1]cuda:0", primals_110: "f32[116, 1, 3, 3][9, 9, 3, 1]cuda:0", primals_111: "i64[][]cuda:0", primals_112: "f32[116][1]cuda:0", primals_113: "f32[116][1]cuda:0", primals_114: "f32[116][1]cuda:0", primals_115: "f32[116][1]cuda:0", primals_116: "f32[116, 116, 1, 1][116, 1, 1, 1]cuda:0", primals_117: "i64[][]cuda:0", primals_118: "f32[116][1]cuda:0", primals_119: "f32[116][1]cuda:0", primals_120: "f32[116][1]cuda:0", primals_121: "f32[116][1]cuda:0", primals_122: "f32[116, 116, 1, 1][116, 1, 1, 1]cuda:0", primals_123: "i64[][]cuda:0", primals_124: "f32[116][1]cuda:0", primals_125: "f32[116][1]cuda:0", primals_126: "f32[116][1]cuda:0", primals_127: "f32[116][1]cuda:0", primals_128: "f32[116, 1, 3, 3][9, 9, 3, 1]cuda:0", primals_129: "i64[][]cuda:0", primals_130: "f32[116][1]cuda:0", primals_131: "f32[116][1]cuda:0", primals_132: "f32[116][1]cuda:0", primals_133: "f32[116][1]cuda:0", primals_134: "f32[116, 116, 1, 1][116, 1, 1, 1]cuda:0", primals_135: "i64[][]cuda:0", primals_136: "f32[116][1]cuda:0", primals_137: "f32[116][1]cuda:0", primals_138: "f32[116][1]cuda:0", primals_139: "f32[116][1]cuda:0", primals_140: "f32[116, 116, 1, 1][116, 1, 1, 1]cuda:0", primals_141: "i64[][]cuda:0", primals_142: "f32[116][1]cuda:0", primals_143: "f32[116][1]cuda:0", primals_144: "f32[116][1]cuda:0", primals_145: "f32[116][1]cuda:0", primals_146: "f32[116, 1, 3, 3][9, 9, 3, 1]cuda:0", primals_147: "i64[][]cuda:0", primals_148: "f32[116][1]cuda:0", primals_149: "f32[116][1]cuda:0", primals_150: "f32[116][1]cuda:0", primals_151: "f32[116][1]cuda:0", primals_152: "f32[116, 116, 1, 1][116, 1, 1, 1]cuda:0", primals_153: "i64[][]cuda:0", primals_154: "f32[116][1]cuda:0", primals_155: "f32[116][1]cuda:0", primals_156: "f32[116][1]cuda:0", primals_157: "f32[116][1]cuda:0", primals_158: "f32[116, 116, 1, 1][116, 1, 1, 1]cuda:0", primals_159: "i64[][]cuda:0", primals_160: "f32[116][1]cuda:0", primals_161: "f32[116][1]cuda:0", primals_162: "f32[116][1]cuda:0", primals_163: "f32[116][1]cuda:0", primals_164: "f32[116, 1, 3, 3][9, 9, 3, 1]cuda:0", primals_165: "i64[][]cuda:0", primals_166: "f32[116][1]cuda:0", primals_167: "f32[116][1]cuda:0", primals_168: "f32[116][1]cuda:0", primals_169: "f32[116][1]cuda:0", primals_170: "f32[116, 116, 1, 1][116, 1, 1, 1]cuda:0", primals_171: "i64[][]cuda:0", primals_172: "f32[116][1]cuda:0", primals_173: "f32[116][1]cuda:0", primals_174: "f32[116][1]cuda:0", primals_175: "f32[116][1]cuda:0", primals_176: "f32[116, 116, 1, 1][116, 1, 1, 1]cuda:0", primals_177: "i64[][]cuda:0", primals_178: "f32[116][1]cuda:0", primals_179: "f32[116][1]cuda:0", primals_180: "f32[116][1]cuda:0", primals_181: "f32[116][1]cuda:0", primals_182: "f32[116, 1, 3, 3][9, 9, 3, 1]cuda:0", primals_183: "i64[][]cuda:0", primals_184: "f32[116][1]cuda:0", primals_185: "f32[116][1]cuda:0", primals_186: "f32[116][1]cuda:0", primals_187: "f32[116][1]cuda:0", primals_188: "f32[116, 116, 1, 1][116, 1, 1, 1]cuda:0", primals_189: "i64[][]cuda:0", primals_190: "f32[116][1]cuda:0", primals_191: "f32[116][1]cuda:0", primals_192: "f32[116][1]cuda:0", primals_193: "f32[116][1]cuda:0", primals_194: "f32[116, 116, 1, 1][116, 1, 1, 1]cuda:0", primals_195: "i64[][]cuda:0", primals_196: "f32[116][1]cuda:0", primals_197: "f32[116][1]cuda:0", primals_198: "f32[116][1]cuda:0", primals_199: "f32[116][1]cuda:0", primals_200: "f32[116, 1, 3, 3][9, 9, 3, 1]cuda:0", primals_201: "i64[][]cuda:0", primals_202: "f32[116][1]cuda:0", primals_203: "f32[116][1]cuda:0", primals_204: "f32[116][1]cuda:0", primals_205: "f32[116][1]cuda:0", primals_206: "f32[116, 116, 1, 1][116, 1, 1, 1]cuda:0", primals_207: "i64[][]cuda:0", primals_208: "f32[116][1]cuda:0", primals_209: "f32[116][1]cuda:0", primals_210: "f32[116][1]cuda:0", primals_211: "f32[116][1]cuda:0", primals_212: "f32[116, 116, 1, 1][116, 1, 1, 1]cuda:0", primals_213: "i64[][]cuda:0", primals_214: "f32[116][1]cuda:0", primals_215: "f32[116][1]cuda:0", primals_216: "f32[116][1]cuda:0", primals_217: "f32[116][1]cuda:0", primals_218: "f32[116, 1, 3, 3][9, 9, 3, 1]cuda:0", primals_219: "i64[][]cuda:0", primals_220: "f32[116][1]cuda:0", primals_221: "f32[116][1]cuda:0", primals_222: "f32[116][1]cuda:0", primals_223: "f32[116][1]cuda:0", primals_224: "f32[116, 116, 1, 1][116, 1, 1, 1]cuda:0", primals_225: "i64[][]cuda:0", primals_226: "f32[116][1]cuda:0", primals_227: "f32[116][1]cuda:0", primals_228: "f32[116][1]cuda:0", primals_229: "f32[116][1]cuda:0", primals_230: "f32[116, 116, 1, 1][116, 1, 1, 1]cuda:0", primals_231: "i64[][]cuda:0", primals_232: "f32[116][1]cuda:0", primals_233: "f32[116][1]cuda:0", primals_234: "f32[116][1]cuda:0", primals_235: "f32[116][1]cuda:0", primals_236: "f32[116, 1, 3, 3][9, 9, 3, 1]cuda:0", primals_237: "i64[][]cuda:0", primals_238: "f32[116][1]cuda:0", primals_239: "f32[116][1]cuda:0", primals_240: "f32[116][1]cuda:0", primals_241: "f32[116][1]cuda:0", primals_242: "f32[116, 116, 1, 1][116, 1, 1, 1]cuda:0", primals_243: "i64[][]cuda:0", primals_244: "f32[116][1]cuda:0", primals_245: "f32[116][1]cuda:0", primals_246: "f32[116][1]cuda:0", primals_247: "f32[116][1]cuda:0", primals_248: "f32[232, 1, 3, 3][9, 9, 3, 1]cuda:0", primals_249: "i64[][]cuda:0", primals_250: "f32[232][1]cuda:0", primals_251: "f32[232][1]cuda:0", primals_252: "f32[232][1]cuda:0", primals_253: "f32[232][1]cuda:0", primals_254: "f32[232, 232, 1, 1][232, 1, 1, 1]cuda:0", primals_255: "i64[][]cuda:0", primals_256: "f32[232][1]cuda:0", primals_257: "f32[232][1]cuda:0", primals_258: "f32[232][1]cuda:0", primals_259: "f32[232][1]cuda:0", primals_260: "f32[232, 232, 1, 1][232, 1, 1, 1]cuda:0", primals_261: "i64[][]cuda:0", primals_262: "f32[232][1]cuda:0", primals_263: "f32[232][1]cuda:0", primals_264: "f32[232][1]cuda:0", primals_265: "f32[232][1]cuda:0", primals_266: "f32[232, 1, 3, 3][9, 9, 3, 1]cuda:0", primals_267: "i64[][]cuda:0", primals_268: "f32[232][1]cuda:0", primals_269: "f32[232][1]cuda:0", primals_270: "f32[232][1]cuda:0", primals_271: "f32[232][1]cuda:0", primals_272: "f32[232, 232, 1, 1][232, 1, 1, 1]cuda:0", primals_273: "i64[][]cuda:0", primals_274: "f32[232][1]cuda:0", primals_275: "f32[232][1]cuda:0", primals_276: "f32[232][1]cuda:0", primals_277: "f32[232][1]cuda:0", primals_278: "f32[232, 232, 1, 1][232, 1, 1, 1]cuda:0", primals_279: "i64[][]cuda:0", primals_280: "f32[232][1]cuda:0", primals_281: "f32[232][1]cuda:0", primals_282: "f32[232][1]cuda:0", primals_283: "f32[232][1]cuda:0", primals_284: "f32[232, 1, 3, 3][9, 9, 3, 1]cuda:0", primals_285: "i64[][]cuda:0", primals_286: "f32[232][1]cuda:0", primals_287: "f32[232][1]cuda:0", primals_288: "f32[232][1]cuda:0", primals_289: "f32[232][1]cuda:0", primals_290: "f32[232, 232, 1, 1][232, 1, 1, 1]cuda:0", primals_291: "i64[][]cuda:0", primals_292: "f32[232][1]cuda:0", primals_293: "f32[232][1]cuda:0", primals_294: "f32[232][1]cuda:0", primals_295: "f32[232][1]cuda:0", primals_296: "f32[232, 232, 1, 1][232, 1, 1, 1]cuda:0", primals_297: "i64[][]cuda:0", primals_298: "f32[232][1]cuda:0", primals_299: "f32[232][1]cuda:0", primals_300: "f32[232][1]cuda:0", primals_301: "f32[232][1]cuda:0", primals_302: "f32[232, 1, 3, 3][9, 9, 3, 1]cuda:0", primals_303: "i64[][]cuda:0", primals_304: "f32[232][1]cuda:0", primals_305: "f32[232][1]cuda:0", primals_306: "f32[232][1]cuda:0", primals_307: "f32[232][1]cuda:0", primals_308: "f32[232, 232, 1, 1][232, 1, 1, 1]cuda:0", primals_309: "i64[][]cuda:0", primals_310: "f32[232][1]cuda:0", primals_311: "f32[232][1]cuda:0", primals_312: "f32[232][1]cuda:0", primals_313: "f32[232][1]cuda:0", primals_314: "f32[232, 232, 1, 1][232, 1, 1, 1]cuda:0", primals_315: "i64[][]cuda:0", primals_316: "f32[232][1]cuda:0", primals_317: "f32[232][1]cuda:0", primals_318: "f32[232][1]cuda:0", primals_319: "f32[232][1]cuda:0", primals_320: "f32[232, 1, 3, 3][9, 9, 3, 1]cuda:0", primals_321: "i64[][]cuda:0", primals_322: "f32[232][1]cuda:0", primals_323: "f32[232][1]cuda:0", primals_324: "f32[232][1]cuda:0", primals_325: "f32[232][1]cuda:0", primals_326: "f32[232, 232, 1, 1][232, 1, 1, 1]cuda:0", primals_327: "i64[][]cuda:0", primals_328: "f32[232][1]cuda:0", primals_329: "f32[232][1]cuda:0", primals_330: "f32[232][1]cuda:0", primals_331: "f32[232][1]cuda:0", primals_332: "f32[1024, 464, 1, 1][464, 1, 1, 1]cuda:0", primals_333: "i64[][]cuda:0", primals_334: "f32[1024][1]cuda:0", primals_335: "f32[1024][1]cuda:0", primals_336: "f32[1024][1]cuda:0", primals_337: "f32[1024][1]cuda:0", primals_338: "f32[1000, 1024][1024, 1]cuda:0", primals_339: "f32[1000][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:155 in _forward_impl, code: x = self.conv1(x)
        convert_element_type: "bf16[24, 3, 3, 3][27, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_1, torch.bfloat16);  primals_1 = None
        convert_element_type_1: "bf16[128, 3, 224, 224][150528, 50176, 224, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16);  primals_2 = None
        convolution: "bf16[128, 24, 112, 112][301056, 12544, 112, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_1, convert_element_type, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
        add: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_3, 1)
        convert_element_type_2: "f32[128, 24, 112, 112][301056, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_2, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_2 = None
        getitem: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        add_1: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[128, 24, 112, 112][301056, 12544, 112, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul: "f32[128, 24, 112, 112][301056, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze: "f32[24][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3])
        mul_1: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze, 0.1);  squeeze = None
        mul_2: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_4, 0.9)
        add_2: "f32[24][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_2: "f32[24][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_3: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_2, 1.0000006228081046);  squeeze_2 = None
        mul_4: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
        mul_5: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_5, 0.9)
        add_3: "f32[24][1]cuda:0" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze: "f32[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_6, -1)
        unsqueeze_1: "f32[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[128, 24, 112, 112][301056, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_7, -1)
        unsqueeze_3: "f32[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_4: "f32[128, 24, 112, 112][301056, 12544, 112, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        convert_element_type_3: "bf16[128, 24, 112, 112][301056, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None
        relu: "bf16[128, 24, 112, 112][301056, 12544, 112, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_3);  convert_element_type_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:156 in _forward_impl, code: x = self.maxpool(x)
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu, [3, 3], [2, 2], [1, 1], [1, 1], False);  relu = None
        getitem_2: "bf16[128, 24, 56, 56][75264, 3136, 56, 1]cuda:0" = _low_memory_max_pool_with_offsets[0]
        getitem_3: "i8[128, 24, 56, 56][75264, 3136, 56, 1]cuda:0" = _low_memory_max_pool_with_offsets[1];  _low_memory_max_pool_with_offsets = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:97 in forward, code: out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        convert_element_type_4: "bf16[24, 1, 3, 3][9, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_8, torch.bfloat16);  primals_8 = None
        convolution_1: "bf16[128, 24, 28, 28][18816, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(getitem_2, convert_element_type_4, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 24)
        add_5: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_9, 1)
        convert_element_type_5: "f32[128, 24, 28, 28][18816, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_5, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_5 = None
        getitem_4: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = var_mean_1[0]
        getitem_5: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        add_6: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
        rsqrt_1: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_1: "f32[128, 24, 28, 28][18816, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_1, getitem_5)
        mul_7: "f32[128, 24, 28, 28][18816, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        squeeze_3: "f32[24][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        squeeze_4: "f32[24][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_8: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
        mul_9: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_10, 0.9)
        add_7: "f32[24][1]cuda:0" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        squeeze_5: "f32[24][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_10: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_5, 1.00000996502277);  squeeze_5 = None
        mul_11: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
        mul_12: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_11, 0.9)
        add_8: "f32[24][1]cuda:0" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        unsqueeze_4: "f32[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_12, -1)
        unsqueeze_5: "f32[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_13: "f32[128, 24, 28, 28][18816, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
        unsqueeze_6: "f32[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_13, -1);  primals_13 = None
        unsqueeze_7: "f32[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_9: "f32[128, 24, 28, 28][18816, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
        convert_element_type_6: "bf16[128, 24, 28, 28][18816, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_9, torch.bfloat16);  add_9 = None
        convert_element_type_7: "bf16[58, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
        convolution_2: "bf16[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_6, convert_element_type_7, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_10: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_15, 1)
        convert_element_type_8: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_2, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_8, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_8 = None
        getitem_6: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = var_mean_2[0]
        getitem_7: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        add_11: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
        rsqrt_2: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_2: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_2, getitem_7)
        mul_14: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        squeeze_6: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3])
        mul_15: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1);  squeeze_6 = None
        mul_16: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_16, 0.9)
        add_12: "f32[58][1]cuda:0" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
        squeeze_8: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_17: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_8, 1.00000996502277);  squeeze_8 = None
        mul_18: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
        mul_19: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_17, 0.9)
        add_13: "f32[58][1]cuda:0" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
        unsqueeze_8: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_18, -1)
        unsqueeze_9: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        mul_20: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
        unsqueeze_10: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_19, -1)
        unsqueeze_11: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        add_14: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
        convert_element_type_9: "bf16[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_14, torch.bfloat16);  add_14 = None
        relu_1: "bf16[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_9);  convert_element_type_9 = None
        convert_element_type_10: "bf16[58, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_20, torch.bfloat16);  primals_20 = None
        convolution_3: "bf16[128, 58, 56, 56][181888, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(getitem_2, convert_element_type_10, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_15: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_21, 1)
        convert_element_type_11: "f32[128, 58, 56, 56][181888, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_3, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_11, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_11 = None
        getitem_8: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = var_mean_3[0]
        getitem_9: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        add_16: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
        rsqrt_3: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_3: "f32[128, 58, 56, 56][181888, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_3, getitem_9)
        mul_21: "f32[128, 58, 56, 56][181888, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        squeeze_9: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        squeeze_10: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_22: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
        mul_23: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_22, 0.9)
        add_17: "f32[58][1]cuda:0" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
        squeeze_11: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_24: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0000024912370735);  squeeze_11 = None
        mul_25: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
        mul_26: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_23, 0.9)
        add_18: "f32[58][1]cuda:0" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
        unsqueeze_12: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_24, -1)
        unsqueeze_13: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_27: "f32[128, 58, 56, 56][181888, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
        unsqueeze_14: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_25, -1);  primals_25 = None
        unsqueeze_15: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_19: "f32[128, 58, 56, 56][181888, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
        convert_element_type_12: "bf16[128, 58, 56, 56][181888, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_19, torch.bfloat16);  add_19 = None
        relu_2: "bf16[128, 58, 56, 56][181888, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_12);  convert_element_type_12 = None
        convert_element_type_13: "bf16[58, 1, 3, 3][9, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_26, torch.bfloat16);  primals_26 = None
        convolution_4: "bf16[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_2, convert_element_type_13, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 58)
        add_20: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_27, 1)
        convert_element_type_14: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_4, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_14, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_14 = None
        getitem_10: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = var_mean_4[0]
        getitem_11: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        add_21: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
        rsqrt_4: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_4: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_4, getitem_11)
        mul_28: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        squeeze_12: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        squeeze_13: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_29: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
        mul_30: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_28, 0.9)
        add_22: "f32[58][1]cuda:0" = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
        squeeze_14: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_31: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_14, 1.00000996502277);  squeeze_14 = None
        mul_32: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
        mul_33: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_29, 0.9)
        add_23: "f32[58][1]cuda:0" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
        unsqueeze_16: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_30, -1)
        unsqueeze_17: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        mul_34: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
        unsqueeze_18: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_31, -1);  primals_31 = None
        unsqueeze_19: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        add_24: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None
        convert_element_type_15: "bf16[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_24, torch.bfloat16);  add_24 = None
        convert_element_type_16: "bf16[58, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_32, torch.bfloat16);  primals_32 = None
        convolution_5: "bf16[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_15, convert_element_type_16, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_25: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_33, 1)
        convert_element_type_17: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_5, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_17, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_17 = None
        getitem_12: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = var_mean_5[0]
        getitem_13: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        add_26: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
        rsqrt_5: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_5: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_5, getitem_13)
        mul_35: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        squeeze_15: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3])
        mul_36: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1);  squeeze_15 = None
        mul_37: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_34, 0.9)
        add_27: "f32[58][1]cuda:0" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
        squeeze_17: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
        mul_38: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_17, 1.00000996502277);  squeeze_17 = None
        mul_39: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
        mul_40: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_35, 0.9)
        add_28: "f32[58][1]cuda:0" = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
        unsqueeze_20: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_36, -1)
        unsqueeze_21: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_41: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
        unsqueeze_22: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_37, -1)
        unsqueeze_23: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_29: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
        convert_element_type_18: "bf16[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_29, torch.bfloat16);  add_29 = None
        relu_3: "bf16[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_18);  convert_element_type_18 = None
        cat: "bf16[128, 116, 28, 28][90944, 784, 28, 1]cuda:0" = torch.ops.aten.cat.default([relu_1, relu_3], 1);  relu_1 = relu_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        view: "bf16[128, 2, 58, 28, 28][90944, 45472, 784, 28, 1]cuda:0" = torch.ops.aten.reshape.default(cat, [128, 2, 58, 28, 28]);  cat = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute: "bf16[128, 58, 2, 28, 28][90944, 784, 45472, 28, 1]cuda:0" = torch.ops.aten.permute.default(view, [0, 2, 1, 3, 4]);  view = None
        clone: "bf16[128, 58, 2, 28, 28][90944, 1568, 784, 28, 1]cuda:0" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_1: "bf16[128, 116, 28, 28][90944, 784, 28, 1]cuda:0" = torch.ops.aten.reshape.default(clone, [128, 116, 28, 28]);  clone = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        split = torch.ops.aten.split.Tensor(view_1, 58, 1);  view_1 = None
        getitem_14: "bf16[128, 58, 28, 28][90944, 784, 28, 1]cuda:0" = split[0]
        getitem_15: "bf16[128, 58, 28, 28][90944, 784, 28, 1]cuda:0" = split[1];  split = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        convert_element_type_19: "bf16[58, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_38, torch.bfloat16);  primals_38 = None
        convolution_6: "bf16[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(getitem_15, convert_element_type_19, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_30: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_39, 1)
        convert_element_type_20: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_6, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_20, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_20 = None
        getitem_16: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = var_mean_6[0]
        getitem_17: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        add_31: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
        rsqrt_6: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        sub_6: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_6, getitem_17)
        mul_42: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        squeeze_18: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
        squeeze_19: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_43: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
        mul_44: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_40, 0.9)
        add_32: "f32[58][1]cuda:0" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
        squeeze_20: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
        mul_45: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_20, 1.00000996502277);  squeeze_20 = None
        mul_46: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_45, 0.1);  mul_45 = None
        mul_47: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_41, 0.9)
        add_33: "f32[58][1]cuda:0" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
        unsqueeze_24: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_42, -1)
        unsqueeze_25: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        mul_48: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_25);  mul_42 = unsqueeze_25 = None
        unsqueeze_26: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_43, -1);  primals_43 = None
        unsqueeze_27: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        add_34: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_48, unsqueeze_27);  mul_48 = unsqueeze_27 = None
        convert_element_type_21: "bf16[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_34, torch.bfloat16);  add_34 = None
        relu_4: "bf16[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_21);  convert_element_type_21 = None
        convert_element_type_22: "bf16[58, 1, 3, 3][9, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16);  primals_44 = None
        convolution_7: "bf16[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_4, convert_element_type_22, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 58)
        add_35: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_45, 1)
        convert_element_type_23: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_7, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_23, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_23 = None
        getitem_18: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = var_mean_7[0]
        getitem_19: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        add_36: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
        rsqrt_7: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_7: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_7, getitem_19)
        mul_49: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        squeeze_21: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        squeeze_22: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
        mul_50: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
        mul_51: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_46, 0.9)
        add_37: "f32[58][1]cuda:0" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
        squeeze_23: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
        mul_52: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_23, 1.00000996502277);  squeeze_23 = None
        mul_53: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_52, 0.1);  mul_52 = None
        mul_54: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_47, 0.9)
        add_38: "f32[58][1]cuda:0" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
        unsqueeze_28: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_48, -1)
        unsqueeze_29: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_55: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
        unsqueeze_30: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_49, -1);  primals_49 = None
        unsqueeze_31: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_39: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None
        convert_element_type_24: "bf16[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_39, torch.bfloat16);  add_39 = None
        convert_element_type_25: "bf16[58, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_50, torch.bfloat16);  primals_50 = None
        convolution_8: "bf16[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_24, convert_element_type_25, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_40: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_51, 1)
        convert_element_type_26: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_8, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_26, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_26 = None
        getitem_20: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = var_mean_8[0]
        getitem_21: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        add_41: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
        rsqrt_8: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
        sub_8: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_8, getitem_21)
        mul_56: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        squeeze_24: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3])
        mul_57: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1);  squeeze_24 = None
        mul_58: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_52, 0.9)
        add_42: "f32[58][1]cuda:0" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
        squeeze_26: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        mul_59: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_26, 1.00000996502277);  squeeze_26 = None
        mul_60: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
        mul_61: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_53, 0.9)
        add_43: "f32[58][1]cuda:0" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
        unsqueeze_32: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_54, -1)
        unsqueeze_33: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        mul_62: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
        unsqueeze_34: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_55, -1)
        unsqueeze_35: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        add_44: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None
        convert_element_type_27: "bf16[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_44, torch.bfloat16);  add_44 = None
        relu_5: "bf16[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_27);  convert_element_type_27 = None
        cat_1: "bf16[128, 116, 28, 28][90944, 784, 28, 1]cuda:0" = torch.ops.aten.cat.default([getitem_14, relu_5], 1);  getitem_14 = relu_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        view_2: "bf16[128, 2, 58, 28, 28][90944, 45472, 784, 28, 1]cuda:0" = torch.ops.aten.reshape.default(cat_1, [128, 2, 58, 28, 28]);  cat_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_1: "bf16[128, 58, 2, 28, 28][90944, 784, 45472, 28, 1]cuda:0" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3, 4]);  view_2 = None
        clone_1: "bf16[128, 58, 2, 28, 28][90944, 1568, 784, 28, 1]cuda:0" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_3: "bf16[128, 116, 28, 28][90944, 784, 28, 1]cuda:0" = torch.ops.aten.reshape.default(clone_1, [128, 116, 28, 28]);  clone_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        split_1 = torch.ops.aten.split.Tensor(view_3, 58, 1);  view_3 = None
        getitem_22: "bf16[128, 58, 28, 28][90944, 784, 28, 1]cuda:0" = split_1[0]
        getitem_23: "bf16[128, 58, 28, 28][90944, 784, 28, 1]cuda:0" = split_1[1];  split_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        convert_element_type_28: "bf16[58, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_56, torch.bfloat16);  primals_56 = None
        convolution_9: "bf16[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(getitem_23, convert_element_type_28, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_45: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_57, 1)
        convert_element_type_29: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_9, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_29, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_29 = None
        getitem_24: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = var_mean_9[0]
        getitem_25: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        add_46: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
        rsqrt_9: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_9: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_9, getitem_25)
        mul_63: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        squeeze_27: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        squeeze_28: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_64: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
        mul_65: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_58, 0.9)
        add_47: "f32[58][1]cuda:0" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
        squeeze_29: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
        mul_66: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_29, 1.00000996502277);  squeeze_29 = None
        mul_67: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, 0.1);  mul_66 = None
        mul_68: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_59, 0.9)
        add_48: "f32[58][1]cuda:0" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
        unsqueeze_36: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_60, -1)
        unsqueeze_37: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_69: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_37);  mul_63 = unsqueeze_37 = None
        unsqueeze_38: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_61, -1);  primals_61 = None
        unsqueeze_39: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_49: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_39);  mul_69 = unsqueeze_39 = None
        convert_element_type_30: "bf16[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_49, torch.bfloat16);  add_49 = None
        relu_6: "bf16[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_30);  convert_element_type_30 = None
        convert_element_type_31: "bf16[58, 1, 3, 3][9, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_62, torch.bfloat16);  primals_62 = None
        convolution_10: "bf16[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_6, convert_element_type_31, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 58)
        add_50: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_63, 1)
        convert_element_type_32: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_10, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_32, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_32 = None
        getitem_26: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = var_mean_10[0]
        getitem_27: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        add_51: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
        rsqrt_10: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        sub_10: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_10, getitem_27)
        mul_70: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        squeeze_30: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        squeeze_31: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_71: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
        mul_72: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_64, 0.9)
        add_52: "f32[58][1]cuda:0" = torch.ops.aten.add.Tensor(mul_71, mul_72);  mul_71 = mul_72 = None
        squeeze_32: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        mul_73: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_32, 1.00000996502277);  squeeze_32 = None
        mul_74: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_73, 0.1);  mul_73 = None
        mul_75: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_65, 0.9)
        add_53: "f32[58][1]cuda:0" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
        unsqueeze_40: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_66, -1)
        unsqueeze_41: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        mul_76: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_41);  mul_70 = unsqueeze_41 = None
        unsqueeze_42: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_67, -1);  primals_67 = None
        unsqueeze_43: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        add_54: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_76, unsqueeze_43);  mul_76 = unsqueeze_43 = None
        convert_element_type_33: "bf16[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_54, torch.bfloat16);  add_54 = None
        convert_element_type_34: "bf16[58, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_68, torch.bfloat16);  primals_68 = None
        convolution_11: "bf16[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_33, convert_element_type_34, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_55: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_69, 1)
        convert_element_type_35: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_11, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_35, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_35 = None
        getitem_28: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = var_mean_11[0]
        getitem_29: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        add_56: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
        rsqrt_11: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        sub_11: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_11, getitem_29)
        mul_77: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        squeeze_33: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3])
        mul_78: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1);  squeeze_33 = None
        mul_79: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_70, 0.9)
        add_57: "f32[58][1]cuda:0" = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
        squeeze_35: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        mul_80: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_35, 1.00000996502277);  squeeze_35 = None
        mul_81: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_80, 0.1);  mul_80 = None
        mul_82: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_71, 0.9)
        add_58: "f32[58][1]cuda:0" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
        unsqueeze_44: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_72, -1)
        unsqueeze_45: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_83: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_45);  mul_77 = unsqueeze_45 = None
        unsqueeze_46: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_73, -1)
        unsqueeze_47: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_59: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_47);  mul_83 = unsqueeze_47 = None
        convert_element_type_36: "bf16[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_59, torch.bfloat16);  add_59 = None
        relu_7: "bf16[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_36);  convert_element_type_36 = None
        cat_2: "bf16[128, 116, 28, 28][90944, 784, 28, 1]cuda:0" = torch.ops.aten.cat.default([getitem_22, relu_7], 1);  getitem_22 = relu_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        view_4: "bf16[128, 2, 58, 28, 28][90944, 45472, 784, 28, 1]cuda:0" = torch.ops.aten.reshape.default(cat_2, [128, 2, 58, 28, 28]);  cat_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_2: "bf16[128, 58, 2, 28, 28][90944, 784, 45472, 28, 1]cuda:0" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3, 4]);  view_4 = None
        clone_2: "bf16[128, 58, 2, 28, 28][90944, 1568, 784, 28, 1]cuda:0" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_5: "bf16[128, 116, 28, 28][90944, 784, 28, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [128, 116, 28, 28]);  clone_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        split_2 = torch.ops.aten.split.Tensor(view_5, 58, 1);  view_5 = None
        getitem_30: "bf16[128, 58, 28, 28][90944, 784, 28, 1]cuda:0" = split_2[0]
        getitem_31: "bf16[128, 58, 28, 28][90944, 784, 28, 1]cuda:0" = split_2[1];  split_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        convert_element_type_37: "bf16[58, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_74, torch.bfloat16);  primals_74 = None
        convolution_12: "bf16[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(getitem_31, convert_element_type_37, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_60: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_75, 1)
        convert_element_type_38: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_12, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_38, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_38 = None
        getitem_32: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = var_mean_12[0]
        getitem_33: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        add_61: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
        rsqrt_12: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        sub_12: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_12, getitem_33)
        mul_84: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        squeeze_36: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        squeeze_37: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_85: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
        mul_86: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_76, 0.9)
        add_62: "f32[58][1]cuda:0" = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
        squeeze_38: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
        mul_87: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_38, 1.00000996502277);  squeeze_38 = None
        mul_88: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_87, 0.1);  mul_87 = None
        mul_89: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_77, 0.9)
        add_63: "f32[58][1]cuda:0" = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
        unsqueeze_48: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_78, -1)
        unsqueeze_49: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        mul_90: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_49);  mul_84 = unsqueeze_49 = None
        unsqueeze_50: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_79, -1);  primals_79 = None
        unsqueeze_51: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        add_64: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_51);  mul_90 = unsqueeze_51 = None
        convert_element_type_39: "bf16[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_64, torch.bfloat16);  add_64 = None
        relu_8: "bf16[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_39);  convert_element_type_39 = None
        convert_element_type_40: "bf16[58, 1, 3, 3][9, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_80, torch.bfloat16);  primals_80 = None
        convolution_13: "bf16[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_8, convert_element_type_40, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 58)
        add_65: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_81, 1)
        convert_element_type_41: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_13, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_41, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_41 = None
        getitem_34: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = var_mean_13[0]
        getitem_35: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        add_66: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
        rsqrt_13: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_13: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_13, getitem_35)
        mul_91: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        squeeze_39: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
        squeeze_40: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
        mul_92: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
        mul_93: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_82, 0.9)
        add_67: "f32[58][1]cuda:0" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
        squeeze_41: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
        mul_94: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_41, 1.00000996502277);  squeeze_41 = None
        mul_95: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_94, 0.1);  mul_94 = None
        mul_96: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_83, 0.9)
        add_68: "f32[58][1]cuda:0" = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None
        unsqueeze_52: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_84, -1)
        unsqueeze_53: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_97: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_53);  mul_91 = unsqueeze_53 = None
        unsqueeze_54: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_85, -1);  primals_85 = None
        unsqueeze_55: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_69: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_55);  mul_97 = unsqueeze_55 = None
        convert_element_type_42: "bf16[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_69, torch.bfloat16);  add_69 = None
        convert_element_type_43: "bf16[58, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_86, torch.bfloat16);  primals_86 = None
        convolution_14: "bf16[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_42, convert_element_type_43, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_70: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_87, 1)
        convert_element_type_44: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_14, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_44, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_44 = None
        getitem_36: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = var_mean_14[0]
        getitem_37: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        add_71: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
        rsqrt_14: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        sub_14: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_14, getitem_37)
        mul_98: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        squeeze_42: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3])
        mul_99: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1);  squeeze_42 = None
        mul_100: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_88, 0.9)
        add_72: "f32[58][1]cuda:0" = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
        squeeze_44: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
        mul_101: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_44, 1.00000996502277);  squeeze_44 = None
        mul_102: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_101, 0.1);  mul_101 = None
        mul_103: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_89, 0.9)
        add_73: "f32[58][1]cuda:0" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
        unsqueeze_56: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_90, -1)
        unsqueeze_57: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        mul_104: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_57);  mul_98 = unsqueeze_57 = None
        unsqueeze_58: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_91, -1)
        unsqueeze_59: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        add_74: "f32[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_59);  mul_104 = unsqueeze_59 = None
        convert_element_type_45: "bf16[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_74, torch.bfloat16);  add_74 = None
        relu_9: "bf16[128, 58, 28, 28][45472, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_45);  convert_element_type_45 = None
        cat_3: "bf16[128, 116, 28, 28][90944, 784, 28, 1]cuda:0" = torch.ops.aten.cat.default([getitem_30, relu_9], 1);  getitem_30 = relu_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        view_6: "bf16[128, 2, 58, 28, 28][90944, 45472, 784, 28, 1]cuda:0" = torch.ops.aten.reshape.default(cat_3, [128, 2, 58, 28, 28]);  cat_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_3: "bf16[128, 58, 2, 28, 28][90944, 784, 45472, 28, 1]cuda:0" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3, 4]);  view_6 = None
        clone_3: "bf16[128, 58, 2, 28, 28][90944, 1568, 784, 28, 1]cuda:0" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_7: "bf16[128, 116, 28, 28][90944, 784, 28, 1]cuda:0" = torch.ops.aten.reshape.default(clone_3, [128, 116, 28, 28]);  clone_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:97 in forward, code: out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        convert_element_type_46: "bf16[116, 1, 3, 3][9, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_92, torch.bfloat16);  primals_92 = None
        convolution_15: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(view_7, convert_element_type_46, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 116)
        add_75: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_93, 1)
        convert_element_type_47: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_15, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_47, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_47 = None
        getitem_38: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_15[0]
        getitem_39: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        add_76: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_38, 1e-05)
        rsqrt_15: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        sub_15: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_15, getitem_39)
        mul_105: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        squeeze_45: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
        squeeze_46: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_106: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
        mul_107: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_94, 0.9)
        add_77: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
        squeeze_47: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
        mul_108: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0000398612827361);  squeeze_47 = None
        mul_109: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_108, 0.1);  mul_108 = None
        mul_110: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_95, 0.9)
        add_78: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
        unsqueeze_60: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_96, -1)
        unsqueeze_61: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_111: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_61);  mul_105 = unsqueeze_61 = None
        unsqueeze_62: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_97, -1);  primals_97 = None
        unsqueeze_63: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_79: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_63);  mul_111 = unsqueeze_63 = None
        convert_element_type_48: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_79, torch.bfloat16);  add_79 = None
        convert_element_type_49: "bf16[116, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_98, torch.bfloat16);  primals_98 = None
        convolution_16: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_48, convert_element_type_49, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_80: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_99, 1)
        convert_element_type_50: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_16, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_50, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_50 = None
        getitem_40: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_16[0]
        getitem_41: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        add_81: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_40, 1e-05)
        rsqrt_16: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
        sub_16: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_16, getitem_41)
        mul_112: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        squeeze_48: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3])
        mul_113: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1);  squeeze_48 = None
        mul_114: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_100, 0.9)
        add_82: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
        squeeze_50: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
        mul_115: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0000398612827361);  squeeze_50 = None
        mul_116: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_115, 0.1);  mul_115 = None
        mul_117: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_101, 0.9)
        add_83: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
        unsqueeze_64: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_102, -1)
        unsqueeze_65: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        mul_118: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_65);  mul_112 = unsqueeze_65 = None
        unsqueeze_66: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_103, -1)
        unsqueeze_67: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        add_84: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_67);  mul_118 = unsqueeze_67 = None
        convert_element_type_51: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_84, torch.bfloat16);  add_84 = None
        relu_10: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_51);  convert_element_type_51 = None
        convert_element_type_52: "bf16[116, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_104, torch.bfloat16);  primals_104 = None
        convolution_17: "bf16[128, 116, 28, 28][90944, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(view_7, convert_element_type_52, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_85: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_105, 1)
        convert_element_type_53: "f32[128, 116, 28, 28][90944, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_17, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_53, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_53 = None
        getitem_42: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_17[0]
        getitem_43: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        add_86: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_42, 1e-05)
        rsqrt_17: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        sub_17: "f32[128, 116, 28, 28][90944, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_17, getitem_43)
        mul_119: "f32[128, 116, 28, 28][90944, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        squeeze_51: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
        squeeze_52: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
        mul_120: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
        mul_121: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_106, 0.9)
        add_87: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_120, mul_121);  mul_120 = mul_121 = None
        squeeze_53: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
        mul_122: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_53, 1.00000996502277);  squeeze_53 = None
        mul_123: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_122, 0.1);  mul_122 = None
        mul_124: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_107, 0.9)
        add_88: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
        unsqueeze_68: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_108, -1)
        unsqueeze_69: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_125: "f32[128, 116, 28, 28][90944, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_69);  mul_119 = unsqueeze_69 = None
        unsqueeze_70: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_109, -1);  primals_109 = None
        unsqueeze_71: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_89: "f32[128, 116, 28, 28][90944, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_71);  mul_125 = unsqueeze_71 = None
        convert_element_type_54: "bf16[128, 116, 28, 28][90944, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_89, torch.bfloat16);  add_89 = None
        relu_11: "bf16[128, 116, 28, 28][90944, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_54);  convert_element_type_54 = None
        convert_element_type_55: "bf16[116, 1, 3, 3][9, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_110, torch.bfloat16);  primals_110 = None
        convolution_18: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_11, convert_element_type_55, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 116)
        add_90: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_111, 1)
        convert_element_type_56: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_18, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_56, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_56 = None
        getitem_44: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_18[0]
        getitem_45: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        add_91: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_44, 1e-05)
        rsqrt_18: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
        sub_18: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_18, getitem_45)
        mul_126: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        squeeze_54: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
        squeeze_55: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
        mul_127: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
        mul_128: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_112, 0.9)
        add_92: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
        squeeze_56: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
        mul_129: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0000398612827361);  squeeze_56 = None
        mul_130: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_129, 0.1);  mul_129 = None
        mul_131: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_113, 0.9)
        add_93: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
        unsqueeze_72: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_114, -1)
        unsqueeze_73: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        mul_132: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_126, unsqueeze_73);  mul_126 = unsqueeze_73 = None
        unsqueeze_74: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_115, -1);  primals_115 = None
        unsqueeze_75: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        add_94: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_132, unsqueeze_75);  mul_132 = unsqueeze_75 = None
        convert_element_type_57: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_94, torch.bfloat16);  add_94 = None
        convert_element_type_58: "bf16[116, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_116, torch.bfloat16);  primals_116 = None
        convolution_19: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_57, convert_element_type_58, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_95: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_117, 1)
        convert_element_type_59: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_19, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_59, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_59 = None
        getitem_46: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_19[0]
        getitem_47: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        add_96: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
        rsqrt_19: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        sub_19: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_19, getitem_47)
        mul_133: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        squeeze_57: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3])
        mul_134: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1);  squeeze_57 = None
        mul_135: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_118, 0.9)
        add_97: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
        squeeze_59: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
        mul_136: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0000398612827361);  squeeze_59 = None
        mul_137: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_136, 0.1);  mul_136 = None
        mul_138: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_119, 0.9)
        add_98: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
        unsqueeze_76: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_120, -1)
        unsqueeze_77: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_139: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_77);  mul_133 = unsqueeze_77 = None
        unsqueeze_78: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_121, -1)
        unsqueeze_79: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_99: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_139, unsqueeze_79);  mul_139 = unsqueeze_79 = None
        convert_element_type_60: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_99, torch.bfloat16);  add_99 = None
        relu_12: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_60);  convert_element_type_60 = None
        cat_4: "bf16[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([relu_10, relu_12], 1);  relu_10 = relu_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        view_8: "bf16[128, 2, 116, 14, 14][45472, 22736, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(cat_4, [128, 2, 116, 14, 14]);  cat_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_4: "bf16[128, 116, 2, 14, 14][45472, 196, 22736, 14, 1]cuda:0" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3, 4]);  view_8 = None
        clone_4: "bf16[128, 116, 2, 14, 14][45472, 392, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_9: "bf16[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(clone_4, [128, 232, 14, 14]);  clone_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        split_3 = torch.ops.aten.split.Tensor(view_9, 116, 1);  view_9 = None
        getitem_48: "bf16[128, 116, 14, 14][45472, 196, 14, 1]cuda:0" = split_3[0]
        getitem_49: "bf16[128, 116, 14, 14][45472, 196, 14, 1]cuda:0" = split_3[1];  split_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        convert_element_type_61: "bf16[116, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_122, torch.bfloat16);  primals_122 = None
        convolution_20: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(getitem_49, convert_element_type_61, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_100: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_123, 1)
        convert_element_type_62: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_20, torch.float32)
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_62, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_62 = None
        getitem_50: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_20[0]
        getitem_51: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None
        add_101: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_50, 1e-05)
        rsqrt_20: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
        sub_20: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_20, getitem_51)
        mul_140: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
        squeeze_60: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
        squeeze_61: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
        mul_141: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
        mul_142: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_124, 0.9)
        add_102: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None
        squeeze_62: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
        mul_143: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0000398612827361);  squeeze_62 = None
        mul_144: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_143, 0.1);  mul_143 = None
        mul_145: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_125, 0.9)
        add_103: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
        unsqueeze_80: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_126, -1)
        unsqueeze_81: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        mul_146: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_140, unsqueeze_81);  mul_140 = unsqueeze_81 = None
        unsqueeze_82: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_127, -1);  primals_127 = None
        unsqueeze_83: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        add_104: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_83);  mul_146 = unsqueeze_83 = None
        convert_element_type_63: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_104, torch.bfloat16);  add_104 = None
        relu_13: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_63);  convert_element_type_63 = None
        convert_element_type_64: "bf16[116, 1, 3, 3][9, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_128, torch.bfloat16);  primals_128 = None
        convolution_21: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_13, convert_element_type_64, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 116)
        add_105: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_129, 1)
        convert_element_type_65: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_21, torch.float32)
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_65, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_65 = None
        getitem_52: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_21[0]
        getitem_53: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        add_106: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_52, 1e-05)
        rsqrt_21: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        sub_21: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_21, getitem_53)
        mul_147: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
        squeeze_63: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
        squeeze_64: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
        mul_148: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
        mul_149: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_130, 0.9)
        add_107: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None
        squeeze_65: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
        mul_150: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0000398612827361);  squeeze_65 = None
        mul_151: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_150, 0.1);  mul_150 = None
        mul_152: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_131, 0.9)
        add_108: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
        unsqueeze_84: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_132, -1)
        unsqueeze_85: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_153: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_147, unsqueeze_85);  mul_147 = unsqueeze_85 = None
        unsqueeze_86: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_133, -1);  primals_133 = None
        unsqueeze_87: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_109: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_153, unsqueeze_87);  mul_153 = unsqueeze_87 = None
        convert_element_type_66: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_109, torch.bfloat16);  add_109 = None
        convert_element_type_67: "bf16[116, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_134, torch.bfloat16);  primals_134 = None
        convolution_22: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_66, convert_element_type_67, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_110: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_135, 1)
        convert_element_type_68: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_22, torch.float32)
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_68, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_68 = None
        getitem_54: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_22[0]
        getitem_55: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None
        add_111: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_54, 1e-05)
        rsqrt_22: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
        sub_22: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_22, getitem_55)
        mul_154: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        squeeze_66: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3])
        mul_155: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1);  squeeze_66 = None
        mul_156: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_136, 0.9)
        add_112: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
        squeeze_68: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
        mul_157: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0000398612827361);  squeeze_68 = None
        mul_158: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_157, 0.1);  mul_157 = None
        mul_159: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_137, 0.9)
        add_113: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None
        unsqueeze_88: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_138, -1)
        unsqueeze_89: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        mul_160: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_89);  mul_154 = unsqueeze_89 = None
        unsqueeze_90: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_139, -1)
        unsqueeze_91: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        add_114: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_160, unsqueeze_91);  mul_160 = unsqueeze_91 = None
        convert_element_type_69: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_114, torch.bfloat16);  add_114 = None
        relu_14: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_69);  convert_element_type_69 = None
        cat_5: "bf16[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([getitem_48, relu_14], 1);  getitem_48 = relu_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        view_10: "bf16[128, 2, 116, 14, 14][45472, 22736, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(cat_5, [128, 2, 116, 14, 14]);  cat_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_5: "bf16[128, 116, 2, 14, 14][45472, 196, 22736, 14, 1]cuda:0" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3, 4]);  view_10 = None
        clone_5: "bf16[128, 116, 2, 14, 14][45472, 392, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_11: "bf16[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(clone_5, [128, 232, 14, 14]);  clone_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        split_4 = torch.ops.aten.split.Tensor(view_11, 116, 1);  view_11 = None
        getitem_56: "bf16[128, 116, 14, 14][45472, 196, 14, 1]cuda:0" = split_4[0]
        getitem_57: "bf16[128, 116, 14, 14][45472, 196, 14, 1]cuda:0" = split_4[1];  split_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        convert_element_type_70: "bf16[116, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_140, torch.bfloat16);  primals_140 = None
        convolution_23: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(getitem_57, convert_element_type_70, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_115: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_141, 1)
        convert_element_type_71: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_23, torch.float32)
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_71, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_71 = None
        getitem_58: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_23[0]
        getitem_59: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        add_116: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_58, 1e-05)
        rsqrt_23: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        sub_23: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_23, getitem_59)
        mul_161: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
        squeeze_69: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
        squeeze_70: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
        mul_162: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
        mul_163: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_142, 0.9)
        add_117: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
        squeeze_71: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
        mul_164: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0000398612827361);  squeeze_71 = None
        mul_165: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_164, 0.1);  mul_164 = None
        mul_166: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_143, 0.9)
        add_118: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
        unsqueeze_92: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_144, -1)
        unsqueeze_93: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        mul_167: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_93);  mul_161 = unsqueeze_93 = None
        unsqueeze_94: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_145, -1);  primals_145 = None
        unsqueeze_95: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        add_119: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_95);  mul_167 = unsqueeze_95 = None
        convert_element_type_72: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_119, torch.bfloat16);  add_119 = None
        relu_15: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_72);  convert_element_type_72 = None
        convert_element_type_73: "bf16[116, 1, 3, 3][9, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_146, torch.bfloat16);  primals_146 = None
        convolution_24: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_15, convert_element_type_73, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 116)
        add_120: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_147, 1)
        convert_element_type_74: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_24, torch.float32)
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_74, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_74 = None
        getitem_60: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_24[0]
        getitem_61: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None
        add_121: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_60, 1e-05)
        rsqrt_24: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
        sub_24: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_24, getitem_61)
        mul_168: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        squeeze_72: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
        squeeze_73: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
        mul_169: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
        mul_170: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_148, 0.9)
        add_122: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
        squeeze_74: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
        mul_171: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0000398612827361);  squeeze_74 = None
        mul_172: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_171, 0.1);  mul_171 = None
        mul_173: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_149, 0.9)
        add_123: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
        unsqueeze_96: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_150, -1)
        unsqueeze_97: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        mul_174: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_168, unsqueeze_97);  mul_168 = unsqueeze_97 = None
        unsqueeze_98: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_151, -1);  primals_151 = None
        unsqueeze_99: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        add_124: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_174, unsqueeze_99);  mul_174 = unsqueeze_99 = None
        convert_element_type_75: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_124, torch.bfloat16);  add_124 = None
        convert_element_type_76: "bf16[116, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_152, torch.bfloat16);  primals_152 = None
        convolution_25: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_75, convert_element_type_76, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_125: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_153, 1)
        convert_element_type_77: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_25, torch.float32)
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_77, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_77 = None
        getitem_62: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_25[0]
        getitem_63: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_25[1];  var_mean_25 = None
        add_126: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
        rsqrt_25: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        sub_25: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_25, getitem_63)
        mul_175: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        squeeze_75: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3])
        mul_176: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1);  squeeze_75 = None
        mul_177: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_154, 0.9)
        add_127: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
        squeeze_77: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
        mul_178: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0000398612827361);  squeeze_77 = None
        mul_179: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_178, 0.1);  mul_178 = None
        mul_180: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_155, 0.9)
        add_128: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
        unsqueeze_100: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_156, -1)
        unsqueeze_101: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_181: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_101);  mul_175 = unsqueeze_101 = None
        unsqueeze_102: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_157, -1)
        unsqueeze_103: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_129: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_103);  mul_181 = unsqueeze_103 = None
        convert_element_type_78: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_129, torch.bfloat16);  add_129 = None
        relu_16: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_78);  convert_element_type_78 = None
        cat_6: "bf16[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([getitem_56, relu_16], 1);  getitem_56 = relu_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        view_12: "bf16[128, 2, 116, 14, 14][45472, 22736, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(cat_6, [128, 2, 116, 14, 14]);  cat_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_6: "bf16[128, 116, 2, 14, 14][45472, 196, 22736, 14, 1]cuda:0" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3, 4]);  view_12 = None
        clone_6: "bf16[128, 116, 2, 14, 14][45472, 392, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_13: "bf16[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(clone_6, [128, 232, 14, 14]);  clone_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        split_5 = torch.ops.aten.split.Tensor(view_13, 116, 1);  view_13 = None
        getitem_64: "bf16[128, 116, 14, 14][45472, 196, 14, 1]cuda:0" = split_5[0]
        getitem_65: "bf16[128, 116, 14, 14][45472, 196, 14, 1]cuda:0" = split_5[1];  split_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        convert_element_type_79: "bf16[116, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_158, torch.bfloat16);  primals_158 = None
        convolution_26: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(getitem_65, convert_element_type_79, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_130: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_159, 1)
        convert_element_type_80: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_26, torch.float32)
        var_mean_26 = torch.ops.aten.var_mean.correction(convert_element_type_80, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_80 = None
        getitem_66: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_26[0]
        getitem_67: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_26[1];  var_mean_26 = None
        add_131: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_66, 1e-05)
        rsqrt_26: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
        sub_26: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_26, getitem_67)
        mul_182: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
        squeeze_78: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
        squeeze_79: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
        mul_183: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
        mul_184: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_160, 0.9)
        add_132: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_183, mul_184);  mul_183 = mul_184 = None
        squeeze_80: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
        mul_185: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0000398612827361);  squeeze_80 = None
        mul_186: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_185, 0.1);  mul_185 = None
        mul_187: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_161, 0.9)
        add_133: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_186, mul_187);  mul_186 = mul_187 = None
        unsqueeze_104: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_162, -1)
        unsqueeze_105: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        mul_188: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_182, unsqueeze_105);  mul_182 = unsqueeze_105 = None
        unsqueeze_106: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_163, -1);  primals_163 = None
        unsqueeze_107: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
        add_134: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_107);  mul_188 = unsqueeze_107 = None
        convert_element_type_81: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_134, torch.bfloat16);  add_134 = None
        relu_17: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_81);  convert_element_type_81 = None
        convert_element_type_82: "bf16[116, 1, 3, 3][9, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_164, torch.bfloat16);  primals_164 = None
        convolution_27: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_17, convert_element_type_82, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 116)
        add_135: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_165, 1)
        convert_element_type_83: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_27, torch.float32)
        var_mean_27 = torch.ops.aten.var_mean.correction(convert_element_type_83, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_83 = None
        getitem_68: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_27[0]
        getitem_69: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_27[1];  var_mean_27 = None
        add_136: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_68, 1e-05)
        rsqrt_27: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
        sub_27: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_27, getitem_69)
        mul_189: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
        squeeze_81: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
        squeeze_82: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
        mul_190: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
        mul_191: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_166, 0.9)
        add_137: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
        squeeze_83: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
        mul_192: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0000398612827361);  squeeze_83 = None
        mul_193: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_192, 0.1);  mul_192 = None
        mul_194: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_167, 0.9)
        add_138: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_193, mul_194);  mul_193 = mul_194 = None
        unsqueeze_108: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_168, -1)
        unsqueeze_109: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
        mul_195: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_189, unsqueeze_109);  mul_189 = unsqueeze_109 = None
        unsqueeze_110: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_169, -1);  primals_169 = None
        unsqueeze_111: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
        add_139: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_195, unsqueeze_111);  mul_195 = unsqueeze_111 = None
        convert_element_type_84: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_139, torch.bfloat16);  add_139 = None
        convert_element_type_85: "bf16[116, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_170, torch.bfloat16);  primals_170 = None
        convolution_28: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_84, convert_element_type_85, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_140: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_171, 1)
        convert_element_type_86: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_28, torch.float32)
        var_mean_28 = torch.ops.aten.var_mean.correction(convert_element_type_86, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_86 = None
        getitem_70: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_28[0]
        getitem_71: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_28[1];  var_mean_28 = None
        add_141: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_70, 1e-05)
        rsqrt_28: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
        sub_28: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_28, getitem_71)
        mul_196: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
        squeeze_84: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3])
        mul_197: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1);  squeeze_84 = None
        mul_198: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_172, 0.9)
        add_142: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
        squeeze_86: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
        mul_199: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0000398612827361);  squeeze_86 = None
        mul_200: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_199, 0.1);  mul_199 = None
        mul_201: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_173, 0.9)
        add_143: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_200, mul_201);  mul_200 = mul_201 = None
        unsqueeze_112: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_174, -1)
        unsqueeze_113: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        mul_202: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_113);  mul_196 = unsqueeze_113 = None
        unsqueeze_114: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_175, -1)
        unsqueeze_115: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        add_144: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_115);  mul_202 = unsqueeze_115 = None
        convert_element_type_87: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_144, torch.bfloat16);  add_144 = None
        relu_18: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_87);  convert_element_type_87 = None
        cat_7: "bf16[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([getitem_64, relu_18], 1);  getitem_64 = relu_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        view_14: "bf16[128, 2, 116, 14, 14][45472, 22736, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(cat_7, [128, 2, 116, 14, 14]);  cat_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_7: "bf16[128, 116, 2, 14, 14][45472, 196, 22736, 14, 1]cuda:0" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3, 4]);  view_14 = None
        clone_7: "bf16[128, 116, 2, 14, 14][45472, 392, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_15: "bf16[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(clone_7, [128, 232, 14, 14]);  clone_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        split_6 = torch.ops.aten.split.Tensor(view_15, 116, 1);  view_15 = None
        getitem_72: "bf16[128, 116, 14, 14][45472, 196, 14, 1]cuda:0" = split_6[0]
        getitem_73: "bf16[128, 116, 14, 14][45472, 196, 14, 1]cuda:0" = split_6[1];  split_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        convert_element_type_88: "bf16[116, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_176, torch.bfloat16);  primals_176 = None
        convolution_29: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(getitem_73, convert_element_type_88, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_145: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_177, 1)
        convert_element_type_89: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_29, torch.float32)
        var_mean_29 = torch.ops.aten.var_mean.correction(convert_element_type_89, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_89 = None
        getitem_74: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_29[0]
        getitem_75: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_29[1];  var_mean_29 = None
        add_146: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_74, 1e-05)
        rsqrt_29: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
        sub_29: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_29, getitem_75)
        mul_203: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
        squeeze_87: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
        squeeze_88: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
        mul_204: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
        mul_205: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_178, 0.9)
        add_147: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None
        squeeze_89: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
        mul_206: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0000398612827361);  squeeze_89 = None
        mul_207: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_206, 0.1);  mul_206 = None
        mul_208: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_179, 0.9)
        add_148: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_207, mul_208);  mul_207 = mul_208 = None
        unsqueeze_116: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_180, -1)
        unsqueeze_117: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        mul_209: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_117);  mul_203 = unsqueeze_117 = None
        unsqueeze_118: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_181, -1);  primals_181 = None
        unsqueeze_119: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
        add_149: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_119);  mul_209 = unsqueeze_119 = None
        convert_element_type_90: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_149, torch.bfloat16);  add_149 = None
        relu_19: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_90);  convert_element_type_90 = None
        convert_element_type_91: "bf16[116, 1, 3, 3][9, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_182, torch.bfloat16);  primals_182 = None
        convolution_30: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_19, convert_element_type_91, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 116)
        add_150: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_183, 1)
        convert_element_type_92: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_30, torch.float32)
        var_mean_30 = torch.ops.aten.var_mean.correction(convert_element_type_92, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_92 = None
        getitem_76: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_30[0]
        getitem_77: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_30[1];  var_mean_30 = None
        add_151: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_76, 1e-05)
        rsqrt_30: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
        sub_30: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_30, getitem_77)
        mul_210: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
        squeeze_90: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
        squeeze_91: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
        mul_211: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
        mul_212: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_184, 0.9)
        add_152: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_211, mul_212);  mul_211 = mul_212 = None
        squeeze_92: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
        mul_213: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0000398612827361);  squeeze_92 = None
        mul_214: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_213, 0.1);  mul_213 = None
        mul_215: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_185, 0.9)
        add_153: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
        unsqueeze_120: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_186, -1)
        unsqueeze_121: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        mul_216: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_121);  mul_210 = unsqueeze_121 = None
        unsqueeze_122: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_187, -1);  primals_187 = None
        unsqueeze_123: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        add_154: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_216, unsqueeze_123);  mul_216 = unsqueeze_123 = None
        convert_element_type_93: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_154, torch.bfloat16);  add_154 = None
        convert_element_type_94: "bf16[116, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_188, torch.bfloat16);  primals_188 = None
        convolution_31: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_93, convert_element_type_94, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_155: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_189, 1)
        convert_element_type_95: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_31, torch.float32)
        var_mean_31 = torch.ops.aten.var_mean.correction(convert_element_type_95, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_95 = None
        getitem_78: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_31[0]
        getitem_79: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_31[1];  var_mean_31 = None
        add_156: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_78, 1e-05)
        rsqrt_31: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
        sub_31: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_31, getitem_79)
        mul_217: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
        squeeze_93: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3])
        mul_218: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1);  squeeze_93 = None
        mul_219: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_190, 0.9)
        add_157: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_218, mul_219);  mul_218 = mul_219 = None
        squeeze_95: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
        mul_220: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0000398612827361);  squeeze_95 = None
        mul_221: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_220, 0.1);  mul_220 = None
        mul_222: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_191, 0.9)
        add_158: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
        unsqueeze_124: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_192, -1)
        unsqueeze_125: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        mul_223: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_125);  mul_217 = unsqueeze_125 = None
        unsqueeze_126: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_193, -1)
        unsqueeze_127: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        add_159: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_223, unsqueeze_127);  mul_223 = unsqueeze_127 = None
        convert_element_type_96: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_159, torch.bfloat16);  add_159 = None
        relu_20: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_96);  convert_element_type_96 = None
        cat_8: "bf16[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([getitem_72, relu_20], 1);  getitem_72 = relu_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        view_16: "bf16[128, 2, 116, 14, 14][45472, 22736, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(cat_8, [128, 2, 116, 14, 14]);  cat_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_8: "bf16[128, 116, 2, 14, 14][45472, 196, 22736, 14, 1]cuda:0" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3, 4]);  view_16 = None
        clone_8: "bf16[128, 116, 2, 14, 14][45472, 392, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_17: "bf16[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(clone_8, [128, 232, 14, 14]);  clone_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        split_7 = torch.ops.aten.split.Tensor(view_17, 116, 1);  view_17 = None
        getitem_80: "bf16[128, 116, 14, 14][45472, 196, 14, 1]cuda:0" = split_7[0]
        getitem_81: "bf16[128, 116, 14, 14][45472, 196, 14, 1]cuda:0" = split_7[1];  split_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        convert_element_type_97: "bf16[116, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_194, torch.bfloat16);  primals_194 = None
        convolution_32: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(getitem_81, convert_element_type_97, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_160: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_195, 1)
        convert_element_type_98: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_32, torch.float32)
        var_mean_32 = torch.ops.aten.var_mean.correction(convert_element_type_98, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_98 = None
        getitem_82: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_32[0]
        getitem_83: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_32[1];  var_mean_32 = None
        add_161: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_82, 1e-05)
        rsqrt_32: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
        sub_32: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_32, getitem_83)
        mul_224: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
        squeeze_96: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
        squeeze_97: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
        mul_225: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
        mul_226: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_196, 0.9)
        add_162: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
        squeeze_98: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
        mul_227: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0000398612827361);  squeeze_98 = None
        mul_228: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_227, 0.1);  mul_227 = None
        mul_229: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_197, 0.9)
        add_163: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_228, mul_229);  mul_228 = mul_229 = None
        unsqueeze_128: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_198, -1)
        unsqueeze_129: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
        mul_230: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_224, unsqueeze_129);  mul_224 = unsqueeze_129 = None
        unsqueeze_130: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_199, -1);  primals_199 = None
        unsqueeze_131: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
        add_164: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_131);  mul_230 = unsqueeze_131 = None
        convert_element_type_99: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_164, torch.bfloat16);  add_164 = None
        relu_21: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_99);  convert_element_type_99 = None
        convert_element_type_100: "bf16[116, 1, 3, 3][9, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_200, torch.bfloat16);  primals_200 = None
        convolution_33: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_21, convert_element_type_100, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 116)
        add_165: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_201, 1)
        convert_element_type_101: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_33, torch.float32)
        var_mean_33 = torch.ops.aten.var_mean.correction(convert_element_type_101, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_101 = None
        getitem_84: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_33[0]
        getitem_85: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_33[1];  var_mean_33 = None
        add_166: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_84, 1e-05)
        rsqrt_33: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
        sub_33: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_33, getitem_85)
        mul_231: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
        squeeze_99: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
        squeeze_100: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
        mul_232: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
        mul_233: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_202, 0.9)
        add_167: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
        squeeze_101: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
        mul_234: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0000398612827361);  squeeze_101 = None
        mul_235: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_234, 0.1);  mul_234 = None
        mul_236: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_203, 0.9)
        add_168: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_235, mul_236);  mul_235 = mul_236 = None
        unsqueeze_132: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_204, -1)
        unsqueeze_133: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
        mul_237: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_231, unsqueeze_133);  mul_231 = unsqueeze_133 = None
        unsqueeze_134: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_205, -1);  primals_205 = None
        unsqueeze_135: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
        add_169: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_237, unsqueeze_135);  mul_237 = unsqueeze_135 = None
        convert_element_type_102: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_169, torch.bfloat16);  add_169 = None
        convert_element_type_103: "bf16[116, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_206, torch.bfloat16);  primals_206 = None
        convolution_34: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_102, convert_element_type_103, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_170: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_207, 1)
        convert_element_type_104: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_34, torch.float32)
        var_mean_34 = torch.ops.aten.var_mean.correction(convert_element_type_104, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_104 = None
        getitem_86: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_34[0]
        getitem_87: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_34[1];  var_mean_34 = None
        add_171: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_86, 1e-05)
        rsqrt_34: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
        sub_34: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_34, getitem_87)
        mul_238: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
        squeeze_102: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3])
        mul_239: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1);  squeeze_102 = None
        mul_240: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_208, 0.9)
        add_172: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_239, mul_240);  mul_239 = mul_240 = None
        squeeze_104: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
        mul_241: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0000398612827361);  squeeze_104 = None
        mul_242: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_241, 0.1);  mul_241 = None
        mul_243: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_209, 0.9)
        add_173: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
        unsqueeze_136: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_210, -1)
        unsqueeze_137: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
        mul_244: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_137);  mul_238 = unsqueeze_137 = None
        unsqueeze_138: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_211, -1)
        unsqueeze_139: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
        add_174: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_244, unsqueeze_139);  mul_244 = unsqueeze_139 = None
        convert_element_type_105: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_174, torch.bfloat16);  add_174 = None
        relu_22: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_105);  convert_element_type_105 = None
        cat_9: "bf16[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([getitem_80, relu_22], 1);  getitem_80 = relu_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        view_18: "bf16[128, 2, 116, 14, 14][45472, 22736, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(cat_9, [128, 2, 116, 14, 14]);  cat_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_9: "bf16[128, 116, 2, 14, 14][45472, 196, 22736, 14, 1]cuda:0" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3, 4]);  view_18 = None
        clone_9: "bf16[128, 116, 2, 14, 14][45472, 392, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format);  permute_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_19: "bf16[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(clone_9, [128, 232, 14, 14]);  clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        split_8 = torch.ops.aten.split.Tensor(view_19, 116, 1);  view_19 = None
        getitem_88: "bf16[128, 116, 14, 14][45472, 196, 14, 1]cuda:0" = split_8[0]
        getitem_89: "bf16[128, 116, 14, 14][45472, 196, 14, 1]cuda:0" = split_8[1];  split_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        convert_element_type_106: "bf16[116, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_212, torch.bfloat16);  primals_212 = None
        convolution_35: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(getitem_89, convert_element_type_106, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_175: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_213, 1)
        convert_element_type_107: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_35, torch.float32)
        var_mean_35 = torch.ops.aten.var_mean.correction(convert_element_type_107, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_107 = None
        getitem_90: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_35[0]
        getitem_91: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_35[1];  var_mean_35 = None
        add_176: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_90, 1e-05)
        rsqrt_35: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_176);  add_176 = None
        sub_35: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_35, getitem_91)
        mul_245: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
        squeeze_105: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
        squeeze_106: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
        mul_246: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
        mul_247: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_214, 0.9)
        add_177: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
        squeeze_107: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
        mul_248: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_107, 1.0000398612827361);  squeeze_107 = None
        mul_249: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_248, 0.1);  mul_248 = None
        mul_250: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_215, 0.9)
        add_178: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
        unsqueeze_140: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_216, -1)
        unsqueeze_141: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
        mul_251: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_141);  mul_245 = unsqueeze_141 = None
        unsqueeze_142: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_217, -1);  primals_217 = None
        unsqueeze_143: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
        add_179: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_251, unsqueeze_143);  mul_251 = unsqueeze_143 = None
        convert_element_type_108: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_179, torch.bfloat16);  add_179 = None
        relu_23: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_108);  convert_element_type_108 = None
        convert_element_type_109: "bf16[116, 1, 3, 3][9, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_218, torch.bfloat16);  primals_218 = None
        convolution_36: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_23, convert_element_type_109, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 116)
        add_180: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_219, 1)
        convert_element_type_110: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_36, torch.float32)
        var_mean_36 = torch.ops.aten.var_mean.correction(convert_element_type_110, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_110 = None
        getitem_92: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_36[0]
        getitem_93: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_36[1];  var_mean_36 = None
        add_181: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_92, 1e-05)
        rsqrt_36: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
        sub_36: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_36, getitem_93)
        mul_252: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
        squeeze_108: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3]);  getitem_93 = None
        squeeze_109: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
        mul_253: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
        mul_254: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_220, 0.9)
        add_182: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_253, mul_254);  mul_253 = mul_254 = None
        squeeze_110: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_92, [0, 2, 3]);  getitem_92 = None
        mul_255: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0000398612827361);  squeeze_110 = None
        mul_256: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_255, 0.1);  mul_255 = None
        mul_257: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_221, 0.9)
        add_183: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
        unsqueeze_144: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_222, -1)
        unsqueeze_145: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
        mul_258: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_252, unsqueeze_145);  mul_252 = unsqueeze_145 = None
        unsqueeze_146: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_223, -1);  primals_223 = None
        unsqueeze_147: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
        add_184: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_258, unsqueeze_147);  mul_258 = unsqueeze_147 = None
        convert_element_type_111: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_184, torch.bfloat16);  add_184 = None
        convert_element_type_112: "bf16[116, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_224, torch.bfloat16);  primals_224 = None
        convolution_37: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_111, convert_element_type_112, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_185: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_225, 1)
        convert_element_type_113: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_37, torch.float32)
        var_mean_37 = torch.ops.aten.var_mean.correction(convert_element_type_113, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_113 = None
        getitem_94: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_37[0]
        getitem_95: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_37[1];  var_mean_37 = None
        add_186: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_94, 1e-05)
        rsqrt_37: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
        sub_37: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_37, getitem_95)
        mul_259: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
        squeeze_111: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3])
        mul_260: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1);  squeeze_111 = None
        mul_261: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_226, 0.9)
        add_187: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_260, mul_261);  mul_260 = mul_261 = None
        squeeze_113: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_94, [0, 2, 3]);  getitem_94 = None
        mul_262: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0000398612827361);  squeeze_113 = None
        mul_263: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_262, 0.1);  mul_262 = None
        mul_264: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_227, 0.9)
        add_188: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_263, mul_264);  mul_263 = mul_264 = None
        unsqueeze_148: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_228, -1)
        unsqueeze_149: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
        mul_265: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_149);  mul_259 = unsqueeze_149 = None
        unsqueeze_150: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_229, -1)
        unsqueeze_151: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
        add_189: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_265, unsqueeze_151);  mul_265 = unsqueeze_151 = None
        convert_element_type_114: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_189, torch.bfloat16);  add_189 = None
        relu_24: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_114);  convert_element_type_114 = None
        cat_10: "bf16[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([getitem_88, relu_24], 1);  getitem_88 = relu_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        view_20: "bf16[128, 2, 116, 14, 14][45472, 22736, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(cat_10, [128, 2, 116, 14, 14]);  cat_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_10: "bf16[128, 116, 2, 14, 14][45472, 196, 22736, 14, 1]cuda:0" = torch.ops.aten.permute.default(view_20, [0, 2, 1, 3, 4]);  view_20 = None
        clone_10: "bf16[128, 116, 2, 14, 14][45472, 392, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format);  permute_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_21: "bf16[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(clone_10, [128, 232, 14, 14]);  clone_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        split_9 = torch.ops.aten.split.Tensor(view_21, 116, 1);  view_21 = None
        getitem_96: "bf16[128, 116, 14, 14][45472, 196, 14, 1]cuda:0" = split_9[0]
        getitem_97: "bf16[128, 116, 14, 14][45472, 196, 14, 1]cuda:0" = split_9[1];  split_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        convert_element_type_115: "bf16[116, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_230, torch.bfloat16);  primals_230 = None
        convolution_38: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(getitem_97, convert_element_type_115, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_190: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_231, 1)
        convert_element_type_116: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_38, torch.float32)
        var_mean_38 = torch.ops.aten.var_mean.correction(convert_element_type_116, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_116 = None
        getitem_98: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_38[0]
        getitem_99: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_38[1];  var_mean_38 = None
        add_191: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_98, 1e-05)
        rsqrt_38: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
        sub_38: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_38, getitem_99)
        mul_266: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
        squeeze_114: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
        squeeze_115: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
        mul_267: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
        mul_268: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_232, 0.9)
        add_192: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_267, mul_268);  mul_267 = mul_268 = None
        squeeze_116: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_98, [0, 2, 3]);  getitem_98 = None
        mul_269: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0000398612827361);  squeeze_116 = None
        mul_270: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_269, 0.1);  mul_269 = None
        mul_271: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_233, 0.9)
        add_193: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_270, mul_271);  mul_270 = mul_271 = None
        unsqueeze_152: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_234, -1)
        unsqueeze_153: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
        mul_272: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_266, unsqueeze_153);  mul_266 = unsqueeze_153 = None
        unsqueeze_154: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_235, -1);  primals_235 = None
        unsqueeze_155: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
        add_194: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_272, unsqueeze_155);  mul_272 = unsqueeze_155 = None
        convert_element_type_117: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_194, torch.bfloat16);  add_194 = None
        relu_25: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_117);  convert_element_type_117 = None
        convert_element_type_118: "bf16[116, 1, 3, 3][9, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_236, torch.bfloat16);  primals_236 = None
        convolution_39: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_25, convert_element_type_118, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 116)
        add_195: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_237, 1)
        convert_element_type_119: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_39, torch.float32)
        var_mean_39 = torch.ops.aten.var_mean.correction(convert_element_type_119, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_119 = None
        getitem_100: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_39[0]
        getitem_101: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_39[1];  var_mean_39 = None
        add_196: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_100, 1e-05)
        rsqrt_39: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_196);  add_196 = None
        sub_39: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_39, getitem_101)
        mul_273: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
        squeeze_117: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2, 3]);  getitem_101 = None
        squeeze_118: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
        mul_274: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1)
        mul_275: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_238, 0.9)
        add_197: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_274, mul_275);  mul_274 = mul_275 = None
        squeeze_119: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_100, [0, 2, 3]);  getitem_100 = None
        mul_276: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_119, 1.0000398612827361);  squeeze_119 = None
        mul_277: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_276, 0.1);  mul_276 = None
        mul_278: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_239, 0.9)
        add_198: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
        unsqueeze_156: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_240, -1)
        unsqueeze_157: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
        mul_279: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_273, unsqueeze_157);  mul_273 = unsqueeze_157 = None
        unsqueeze_158: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_241, -1);  primals_241 = None
        unsqueeze_159: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
        add_199: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_279, unsqueeze_159);  mul_279 = unsqueeze_159 = None
        convert_element_type_120: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_199, torch.bfloat16);  add_199 = None
        convert_element_type_121: "bf16[116, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_242, torch.bfloat16);  primals_242 = None
        convolution_40: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_120, convert_element_type_121, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_200: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_243, 1)
        convert_element_type_122: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_40, torch.float32)
        var_mean_40 = torch.ops.aten.var_mean.correction(convert_element_type_122, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_122 = None
        getitem_102: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_40[0]
        getitem_103: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = var_mean_40[1];  var_mean_40 = None
        add_201: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_102, 1e-05)
        rsqrt_40: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_201);  add_201 = None
        sub_40: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_40, getitem_103)
        mul_280: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
        squeeze_120: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3])
        mul_281: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1);  squeeze_120 = None
        mul_282: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_244, 0.9)
        add_202: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_281, mul_282);  mul_281 = mul_282 = None
        squeeze_122: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_102, [0, 2, 3]);  getitem_102 = None
        mul_283: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_122, 1.0000398612827361);  squeeze_122 = None
        mul_284: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_283, 0.1);  mul_283 = None
        mul_285: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_245, 0.9)
        add_203: "f32[116][1]cuda:0" = torch.ops.aten.add.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
        unsqueeze_160: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_246, -1)
        unsqueeze_161: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
        mul_286: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_161);  mul_280 = unsqueeze_161 = None
        unsqueeze_162: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_247, -1)
        unsqueeze_163: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
        add_204: "f32[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_286, unsqueeze_163);  mul_286 = unsqueeze_163 = None
        convert_element_type_123: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_204, torch.bfloat16);  add_204 = None
        relu_26: "bf16[128, 116, 14, 14][22736, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_123);  convert_element_type_123 = None
        cat_11: "bf16[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([getitem_96, relu_26], 1);  getitem_96 = relu_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        view_22: "bf16[128, 2, 116, 14, 14][45472, 22736, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(cat_11, [128, 2, 116, 14, 14]);  cat_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_11: "bf16[128, 116, 2, 14, 14][45472, 196, 22736, 14, 1]cuda:0" = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3, 4]);  view_22 = None
        clone_11: "bf16[128, 116, 2, 14, 14][45472, 392, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_23: "bf16[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(clone_11, [128, 232, 14, 14]);  clone_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:97 in forward, code: out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        convert_element_type_124: "bf16[232, 1, 3, 3][9, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_248, torch.bfloat16);  primals_248 = None
        convolution_41: "bf16[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(view_23, convert_element_type_124, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 232)
        add_205: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_249, 1)
        convert_element_type_125: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_41, torch.float32)
        var_mean_41 = torch.ops.aten.var_mean.correction(convert_element_type_125, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_125 = None
        getitem_104: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = var_mean_41[0]
        getitem_105: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = var_mean_41[1];  var_mean_41 = None
        add_206: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_104, 1e-05)
        rsqrt_41: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_206);  add_206 = None
        sub_41: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_41, getitem_105)
        mul_287: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
        squeeze_123: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_105, [0, 2, 3]);  getitem_105 = None
        squeeze_124: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
        mul_288: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
        mul_289: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_250, 0.9)
        add_207: "f32[232][1]cuda:0" = torch.ops.aten.add.Tensor(mul_288, mul_289);  mul_288 = mul_289 = None
        squeeze_125: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_104, [0, 2, 3]);  getitem_104 = None
        mul_290: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_125, 1.0001594642002871);  squeeze_125 = None
        mul_291: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_290, 0.1);  mul_290 = None
        mul_292: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_251, 0.9)
        add_208: "f32[232][1]cuda:0" = torch.ops.aten.add.Tensor(mul_291, mul_292);  mul_291 = mul_292 = None
        unsqueeze_164: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_252, -1)
        unsqueeze_165: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
        mul_293: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_287, unsqueeze_165);  mul_287 = unsqueeze_165 = None
        unsqueeze_166: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_253, -1);  primals_253 = None
        unsqueeze_167: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
        add_209: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_293, unsqueeze_167);  mul_293 = unsqueeze_167 = None
        convert_element_type_126: "bf16[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_209, torch.bfloat16);  add_209 = None
        convert_element_type_127: "bf16[232, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_254, torch.bfloat16);  primals_254 = None
        convolution_42: "bf16[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_126, convert_element_type_127, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_210: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_255, 1)
        convert_element_type_128: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_42, torch.float32)
        var_mean_42 = torch.ops.aten.var_mean.correction(convert_element_type_128, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_128 = None
        getitem_106: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = var_mean_42[0]
        getitem_107: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = var_mean_42[1];  var_mean_42 = None
        add_211: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_106, 1e-05)
        rsqrt_42: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_211);  add_211 = None
        sub_42: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_42, getitem_107)
        mul_294: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
        squeeze_126: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3])
        mul_295: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1);  squeeze_126 = None
        mul_296: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_256, 0.9)
        add_212: "f32[232][1]cuda:0" = torch.ops.aten.add.Tensor(mul_295, mul_296);  mul_295 = mul_296 = None
        squeeze_128: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_106, [0, 2, 3]);  getitem_106 = None
        mul_297: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_128, 1.0001594642002871);  squeeze_128 = None
        mul_298: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_297, 0.1);  mul_297 = None
        mul_299: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_257, 0.9)
        add_213: "f32[232][1]cuda:0" = torch.ops.aten.add.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None
        unsqueeze_168: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_258, -1)
        unsqueeze_169: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
        mul_300: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_294, unsqueeze_169);  mul_294 = unsqueeze_169 = None
        unsqueeze_170: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_259, -1)
        unsqueeze_171: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
        add_214: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_300, unsqueeze_171);  mul_300 = unsqueeze_171 = None
        convert_element_type_129: "bf16[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_214, torch.bfloat16);  add_214 = None
        relu_27: "bf16[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_129);  convert_element_type_129 = None
        convert_element_type_130: "bf16[232, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_260, torch.bfloat16);  primals_260 = None
        convolution_43: "bf16[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(view_23, convert_element_type_130, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_215: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_261, 1)
        convert_element_type_131: "f32[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_43, torch.float32)
        var_mean_43 = torch.ops.aten.var_mean.correction(convert_element_type_131, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_131 = None
        getitem_108: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = var_mean_43[0]
        getitem_109: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = var_mean_43[1];  var_mean_43 = None
        add_216: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_108, 1e-05)
        rsqrt_43: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_216);  add_216 = None
        sub_43: "f32[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_43, getitem_109)
        mul_301: "f32[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
        squeeze_129: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_109, [0, 2, 3]);  getitem_109 = None
        squeeze_130: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
        mul_302: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1)
        mul_303: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_262, 0.9)
        add_217: "f32[232][1]cuda:0" = torch.ops.aten.add.Tensor(mul_302, mul_303);  mul_302 = mul_303 = None
        squeeze_131: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_108, [0, 2, 3]);  getitem_108 = None
        mul_304: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_131, 1.0000398612827361);  squeeze_131 = None
        mul_305: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_304, 0.1);  mul_304 = None
        mul_306: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_263, 0.9)
        add_218: "f32[232][1]cuda:0" = torch.ops.aten.add.Tensor(mul_305, mul_306);  mul_305 = mul_306 = None
        unsqueeze_172: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_264, -1)
        unsqueeze_173: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
        mul_307: "f32[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_173);  mul_301 = unsqueeze_173 = None
        unsqueeze_174: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_265, -1);  primals_265 = None
        unsqueeze_175: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
        add_219: "f32[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_307, unsqueeze_175);  mul_307 = unsqueeze_175 = None
        convert_element_type_132: "bf16[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_219, torch.bfloat16);  add_219 = None
        relu_28: "bf16[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_132);  convert_element_type_132 = None
        convert_element_type_133: "bf16[232, 1, 3, 3][9, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_266, torch.bfloat16);  primals_266 = None
        convolution_44: "bf16[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_28, convert_element_type_133, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 232)
        add_220: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_267, 1)
        convert_element_type_134: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_44, torch.float32)
        var_mean_44 = torch.ops.aten.var_mean.correction(convert_element_type_134, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_134 = None
        getitem_110: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = var_mean_44[0]
        getitem_111: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = var_mean_44[1];  var_mean_44 = None
        add_221: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_110, 1e-05)
        rsqrt_44: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_221);  add_221 = None
        sub_44: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_44, getitem_111)
        mul_308: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
        squeeze_132: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_111, [0, 2, 3]);  getitem_111 = None
        squeeze_133: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
        mul_309: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_132, 0.1)
        mul_310: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_268, 0.9)
        add_222: "f32[232][1]cuda:0" = torch.ops.aten.add.Tensor(mul_309, mul_310);  mul_309 = mul_310 = None
        squeeze_134: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_110, [0, 2, 3]);  getitem_110 = None
        mul_311: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_134, 1.0001594642002871);  squeeze_134 = None
        mul_312: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_311, 0.1);  mul_311 = None
        mul_313: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_269, 0.9)
        add_223: "f32[232][1]cuda:0" = torch.ops.aten.add.Tensor(mul_312, mul_313);  mul_312 = mul_313 = None
        unsqueeze_176: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_270, -1)
        unsqueeze_177: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
        mul_314: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_308, unsqueeze_177);  mul_308 = unsqueeze_177 = None
        unsqueeze_178: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_271, -1);  primals_271 = None
        unsqueeze_179: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
        add_224: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_314, unsqueeze_179);  mul_314 = unsqueeze_179 = None
        convert_element_type_135: "bf16[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_224, torch.bfloat16);  add_224 = None
        convert_element_type_136: "bf16[232, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_272, torch.bfloat16);  primals_272 = None
        convolution_45: "bf16[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_135, convert_element_type_136, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_225: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_273, 1)
        convert_element_type_137: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_45, torch.float32)
        var_mean_45 = torch.ops.aten.var_mean.correction(convert_element_type_137, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_137 = None
        getitem_112: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = var_mean_45[0]
        getitem_113: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = var_mean_45[1];  var_mean_45 = None
        add_226: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_112, 1e-05)
        rsqrt_45: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_226);  add_226 = None
        sub_45: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_45, getitem_113)
        mul_315: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
        squeeze_135: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2, 3])
        mul_316: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_135, 0.1);  squeeze_135 = None
        mul_317: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_274, 0.9)
        add_227: "f32[232][1]cuda:0" = torch.ops.aten.add.Tensor(mul_316, mul_317);  mul_316 = mul_317 = None
        squeeze_137: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_112, [0, 2, 3]);  getitem_112 = None
        mul_318: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_137, 1.0001594642002871);  squeeze_137 = None
        mul_319: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_318, 0.1);  mul_318 = None
        mul_320: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_275, 0.9)
        add_228: "f32[232][1]cuda:0" = torch.ops.aten.add.Tensor(mul_319, mul_320);  mul_319 = mul_320 = None
        unsqueeze_180: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_276, -1)
        unsqueeze_181: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
        mul_321: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_315, unsqueeze_181);  mul_315 = unsqueeze_181 = None
        unsqueeze_182: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_277, -1)
        unsqueeze_183: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
        add_229: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_321, unsqueeze_183);  mul_321 = unsqueeze_183 = None
        convert_element_type_138: "bf16[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_229, torch.bfloat16);  add_229 = None
        relu_29: "bf16[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_138);  convert_element_type_138 = None
        cat_12: "bf16[128, 464, 7, 7][22736, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([relu_27, relu_29], 1);  relu_27 = relu_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        view_24: "bf16[128, 2, 232, 7, 7][22736, 11368, 49, 7, 1]cuda:0" = torch.ops.aten.reshape.default(cat_12, [128, 2, 232, 7, 7]);  cat_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_12: "bf16[128, 232, 2, 7, 7][22736, 49, 11368, 7, 1]cuda:0" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3, 4]);  view_24 = None
        clone_12: "bf16[128, 232, 2, 7, 7][22736, 98, 49, 7, 1]cuda:0" = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_25: "bf16[128, 464, 7, 7][22736, 49, 7, 1]cuda:0" = torch.ops.aten.reshape.default(clone_12, [128, 464, 7, 7]);  clone_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        split_10 = torch.ops.aten.split.Tensor(view_25, 232, 1);  view_25 = None
        getitem_114: "bf16[128, 232, 7, 7][22736, 49, 7, 1]cuda:0" = split_10[0]
        getitem_115: "bf16[128, 232, 7, 7][22736, 49, 7, 1]cuda:0" = split_10[1];  split_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        convert_element_type_139: "bf16[232, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_278, torch.bfloat16);  primals_278 = None
        convolution_46: "bf16[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(getitem_115, convert_element_type_139, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_230: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_279, 1)
        convert_element_type_140: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_46, torch.float32)
        var_mean_46 = torch.ops.aten.var_mean.correction(convert_element_type_140, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_140 = None
        getitem_116: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = var_mean_46[0]
        getitem_117: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = var_mean_46[1];  var_mean_46 = None
        add_231: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_116, 1e-05)
        rsqrt_46: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_231);  add_231 = None
        sub_46: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_46, getitem_117)
        mul_322: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
        squeeze_138: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_117, [0, 2, 3]);  getitem_117 = None
        squeeze_139: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
        mul_323: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_138, 0.1)
        mul_324: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_280, 0.9)
        add_232: "f32[232][1]cuda:0" = torch.ops.aten.add.Tensor(mul_323, mul_324);  mul_323 = mul_324 = None
        squeeze_140: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_116, [0, 2, 3]);  getitem_116 = None
        mul_325: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_140, 1.0001594642002871);  squeeze_140 = None
        mul_326: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_325, 0.1);  mul_325 = None
        mul_327: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_281, 0.9)
        add_233: "f32[232][1]cuda:0" = torch.ops.aten.add.Tensor(mul_326, mul_327);  mul_326 = mul_327 = None
        unsqueeze_184: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_282, -1)
        unsqueeze_185: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
        mul_328: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_322, unsqueeze_185);  mul_322 = unsqueeze_185 = None
        unsqueeze_186: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_283, -1);  primals_283 = None
        unsqueeze_187: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
        add_234: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_328, unsqueeze_187);  mul_328 = unsqueeze_187 = None
        convert_element_type_141: "bf16[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_234, torch.bfloat16);  add_234 = None
        relu_30: "bf16[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_141);  convert_element_type_141 = None
        convert_element_type_142: "bf16[232, 1, 3, 3][9, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_284, torch.bfloat16);  primals_284 = None
        convolution_47: "bf16[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_30, convert_element_type_142, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 232)
        add_235: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_285, 1)
        convert_element_type_143: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_47, torch.float32)
        var_mean_47 = torch.ops.aten.var_mean.correction(convert_element_type_143, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_143 = None
        getitem_118: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = var_mean_47[0]
        getitem_119: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = var_mean_47[1];  var_mean_47 = None
        add_236: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_118, 1e-05)
        rsqrt_47: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_236);  add_236 = None
        sub_47: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_47, getitem_119)
        mul_329: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
        squeeze_141: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3]);  getitem_119 = None
        squeeze_142: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
        mul_330: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_141, 0.1)
        mul_331: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_286, 0.9)
        add_237: "f32[232][1]cuda:0" = torch.ops.aten.add.Tensor(mul_330, mul_331);  mul_330 = mul_331 = None
        squeeze_143: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_118, [0, 2, 3]);  getitem_118 = None
        mul_332: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_143, 1.0001594642002871);  squeeze_143 = None
        mul_333: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_332, 0.1);  mul_332 = None
        mul_334: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_287, 0.9)
        add_238: "f32[232][1]cuda:0" = torch.ops.aten.add.Tensor(mul_333, mul_334);  mul_333 = mul_334 = None
        unsqueeze_188: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_288, -1)
        unsqueeze_189: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
        mul_335: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_329, unsqueeze_189);  mul_329 = unsqueeze_189 = None
        unsqueeze_190: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_289, -1);  primals_289 = None
        unsqueeze_191: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
        add_239: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_335, unsqueeze_191);  mul_335 = unsqueeze_191 = None
        convert_element_type_144: "bf16[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_239, torch.bfloat16);  add_239 = None
        convert_element_type_145: "bf16[232, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_290, torch.bfloat16);  primals_290 = None
        convolution_48: "bf16[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_144, convert_element_type_145, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_240: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_291, 1)
        convert_element_type_146: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_48, torch.float32)
        var_mean_48 = torch.ops.aten.var_mean.correction(convert_element_type_146, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_146 = None
        getitem_120: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = var_mean_48[0]
        getitem_121: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = var_mean_48[1];  var_mean_48 = None
        add_241: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_120, 1e-05)
        rsqrt_48: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_241);  add_241 = None
        sub_48: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_48, getitem_121)
        mul_336: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
        squeeze_144: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3])
        mul_337: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_144, 0.1);  squeeze_144 = None
        mul_338: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_292, 0.9)
        add_242: "f32[232][1]cuda:0" = torch.ops.aten.add.Tensor(mul_337, mul_338);  mul_337 = mul_338 = None
        squeeze_146: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_120, [0, 2, 3]);  getitem_120 = None
        mul_339: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_146, 1.0001594642002871);  squeeze_146 = None
        mul_340: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_339, 0.1);  mul_339 = None
        mul_341: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_293, 0.9)
        add_243: "f32[232][1]cuda:0" = torch.ops.aten.add.Tensor(mul_340, mul_341);  mul_340 = mul_341 = None
        unsqueeze_192: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_294, -1)
        unsqueeze_193: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
        mul_342: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_336, unsqueeze_193);  mul_336 = unsqueeze_193 = None
        unsqueeze_194: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_295, -1)
        unsqueeze_195: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
        add_244: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_342, unsqueeze_195);  mul_342 = unsqueeze_195 = None
        convert_element_type_147: "bf16[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_244, torch.bfloat16);  add_244 = None
        relu_31: "bf16[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_147);  convert_element_type_147 = None
        cat_13: "bf16[128, 464, 7, 7][22736, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([getitem_114, relu_31], 1);  getitem_114 = relu_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        view_26: "bf16[128, 2, 232, 7, 7][22736, 11368, 49, 7, 1]cuda:0" = torch.ops.aten.reshape.default(cat_13, [128, 2, 232, 7, 7]);  cat_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_13: "bf16[128, 232, 2, 7, 7][22736, 49, 11368, 7, 1]cuda:0" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3, 4]);  view_26 = None
        clone_13: "bf16[128, 232, 2, 7, 7][22736, 98, 49, 7, 1]cuda:0" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_27: "bf16[128, 464, 7, 7][22736, 49, 7, 1]cuda:0" = torch.ops.aten.reshape.default(clone_13, [128, 464, 7, 7]);  clone_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        split_11 = torch.ops.aten.split.Tensor(view_27, 232, 1);  view_27 = None
        getitem_122: "bf16[128, 232, 7, 7][22736, 49, 7, 1]cuda:0" = split_11[0]
        getitem_123: "bf16[128, 232, 7, 7][22736, 49, 7, 1]cuda:0" = split_11[1];  split_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        convert_element_type_148: "bf16[232, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_296, torch.bfloat16);  primals_296 = None
        convolution_49: "bf16[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(getitem_123, convert_element_type_148, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_245: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_297, 1)
        convert_element_type_149: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_49, torch.float32)
        var_mean_49 = torch.ops.aten.var_mean.correction(convert_element_type_149, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_149 = None
        getitem_124: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = var_mean_49[0]
        getitem_125: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = var_mean_49[1];  var_mean_49 = None
        add_246: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_124, 1e-05)
        rsqrt_49: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_246);  add_246 = None
        sub_49: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_49, getitem_125)
        mul_343: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
        squeeze_147: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_125, [0, 2, 3]);  getitem_125 = None
        squeeze_148: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
        mul_344: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_147, 0.1)
        mul_345: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_298, 0.9)
        add_247: "f32[232][1]cuda:0" = torch.ops.aten.add.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
        squeeze_149: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_124, [0, 2, 3]);  getitem_124 = None
        mul_346: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_149, 1.0001594642002871);  squeeze_149 = None
        mul_347: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_346, 0.1);  mul_346 = None
        mul_348: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_299, 0.9)
        add_248: "f32[232][1]cuda:0" = torch.ops.aten.add.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
        unsqueeze_196: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_300, -1)
        unsqueeze_197: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
        mul_349: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_197);  mul_343 = unsqueeze_197 = None
        unsqueeze_198: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_301, -1);  primals_301 = None
        unsqueeze_199: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
        add_249: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_349, unsqueeze_199);  mul_349 = unsqueeze_199 = None
        convert_element_type_150: "bf16[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_249, torch.bfloat16);  add_249 = None
        relu_32: "bf16[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_150);  convert_element_type_150 = None
        convert_element_type_151: "bf16[232, 1, 3, 3][9, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_302, torch.bfloat16);  primals_302 = None
        convolution_50: "bf16[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_32, convert_element_type_151, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 232)
        add_250: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_303, 1)
        convert_element_type_152: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_50, torch.float32)
        var_mean_50 = torch.ops.aten.var_mean.correction(convert_element_type_152, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_152 = None
        getitem_126: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = var_mean_50[0]
        getitem_127: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = var_mean_50[1];  var_mean_50 = None
        add_251: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_126, 1e-05)
        rsqrt_50: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_251);  add_251 = None
        sub_50: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_50, getitem_127)
        mul_350: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
        squeeze_150: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_127, [0, 2, 3]);  getitem_127 = None
        squeeze_151: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
        mul_351: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_150, 0.1)
        mul_352: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_304, 0.9)
        add_252: "f32[232][1]cuda:0" = torch.ops.aten.add.Tensor(mul_351, mul_352);  mul_351 = mul_352 = None
        squeeze_152: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_126, [0, 2, 3]);  getitem_126 = None
        mul_353: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_152, 1.0001594642002871);  squeeze_152 = None
        mul_354: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_353, 0.1);  mul_353 = None
        mul_355: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_305, 0.9)
        add_253: "f32[232][1]cuda:0" = torch.ops.aten.add.Tensor(mul_354, mul_355);  mul_354 = mul_355 = None
        unsqueeze_200: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_306, -1)
        unsqueeze_201: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
        mul_356: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_350, unsqueeze_201);  mul_350 = unsqueeze_201 = None
        unsqueeze_202: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_307, -1);  primals_307 = None
        unsqueeze_203: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
        add_254: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_356, unsqueeze_203);  mul_356 = unsqueeze_203 = None
        convert_element_type_153: "bf16[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_254, torch.bfloat16);  add_254 = None
        convert_element_type_154: "bf16[232, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_308, torch.bfloat16);  primals_308 = None
        convolution_51: "bf16[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_153, convert_element_type_154, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_255: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_309, 1)
        convert_element_type_155: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_51, torch.float32)
        var_mean_51 = torch.ops.aten.var_mean.correction(convert_element_type_155, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_155 = None
        getitem_128: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = var_mean_51[0]
        getitem_129: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = var_mean_51[1];  var_mean_51 = None
        add_256: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_128, 1e-05)
        rsqrt_51: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_256);  add_256 = None
        sub_51: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_51, getitem_129)
        mul_357: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
        squeeze_153: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_129, [0, 2, 3])
        mul_358: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_153, 0.1);  squeeze_153 = None
        mul_359: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_310, 0.9)
        add_257: "f32[232][1]cuda:0" = torch.ops.aten.add.Tensor(mul_358, mul_359);  mul_358 = mul_359 = None
        squeeze_155: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_128, [0, 2, 3]);  getitem_128 = None
        mul_360: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_155, 1.0001594642002871);  squeeze_155 = None
        mul_361: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_360, 0.1);  mul_360 = None
        mul_362: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_311, 0.9)
        add_258: "f32[232][1]cuda:0" = torch.ops.aten.add.Tensor(mul_361, mul_362);  mul_361 = mul_362 = None
        unsqueeze_204: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_312, -1)
        unsqueeze_205: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
        mul_363: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_357, unsqueeze_205);  mul_357 = unsqueeze_205 = None
        unsqueeze_206: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_313, -1)
        unsqueeze_207: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
        add_259: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_363, unsqueeze_207);  mul_363 = unsqueeze_207 = None
        convert_element_type_156: "bf16[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_259, torch.bfloat16);  add_259 = None
        relu_33: "bf16[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_156);  convert_element_type_156 = None
        cat_14: "bf16[128, 464, 7, 7][22736, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([getitem_122, relu_33], 1);  getitem_122 = relu_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        view_28: "bf16[128, 2, 232, 7, 7][22736, 11368, 49, 7, 1]cuda:0" = torch.ops.aten.reshape.default(cat_14, [128, 2, 232, 7, 7]);  cat_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_14: "bf16[128, 232, 2, 7, 7][22736, 49, 11368, 7, 1]cuda:0" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3, 4]);  view_28 = None
        clone_14: "bf16[128, 232, 2, 7, 7][22736, 98, 49, 7, 1]cuda:0" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_29: "bf16[128, 464, 7, 7][22736, 49, 7, 1]cuda:0" = torch.ops.aten.reshape.default(clone_14, [128, 464, 7, 7]);  clone_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        split_12 = torch.ops.aten.split.Tensor(view_29, 232, 1);  view_29 = None
        getitem_130: "bf16[128, 232, 7, 7][22736, 49, 7, 1]cuda:0" = split_12[0]
        getitem_131: "bf16[128, 232, 7, 7][22736, 49, 7, 1]cuda:0" = split_12[1];  split_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        convert_element_type_157: "bf16[232, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_314, torch.bfloat16);  primals_314 = None
        convolution_52: "bf16[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(getitem_131, convert_element_type_157, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_260: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_315, 1)
        convert_element_type_158: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_52, torch.float32)
        var_mean_52 = torch.ops.aten.var_mean.correction(convert_element_type_158, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_158 = None
        getitem_132: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = var_mean_52[0]
        getitem_133: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = var_mean_52[1];  var_mean_52 = None
        add_261: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_132, 1e-05)
        rsqrt_52: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_261);  add_261 = None
        sub_52: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_52, getitem_133)
        mul_364: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
        squeeze_156: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_133, [0, 2, 3]);  getitem_133 = None
        squeeze_157: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
        mul_365: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_156, 0.1)
        mul_366: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_316, 0.9)
        add_262: "f32[232][1]cuda:0" = torch.ops.aten.add.Tensor(mul_365, mul_366);  mul_365 = mul_366 = None
        squeeze_158: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_132, [0, 2, 3]);  getitem_132 = None
        mul_367: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_158, 1.0001594642002871);  squeeze_158 = None
        mul_368: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_367, 0.1);  mul_367 = None
        mul_369: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_317, 0.9)
        add_263: "f32[232][1]cuda:0" = torch.ops.aten.add.Tensor(mul_368, mul_369);  mul_368 = mul_369 = None
        unsqueeze_208: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_318, -1)
        unsqueeze_209: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
        mul_370: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_364, unsqueeze_209);  mul_364 = unsqueeze_209 = None
        unsqueeze_210: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_319, -1);  primals_319 = None
        unsqueeze_211: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
        add_264: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_370, unsqueeze_211);  mul_370 = unsqueeze_211 = None
        convert_element_type_159: "bf16[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_264, torch.bfloat16);  add_264 = None
        relu_34: "bf16[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_159);  convert_element_type_159 = None
        convert_element_type_160: "bf16[232, 1, 3, 3][9, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_320, torch.bfloat16);  primals_320 = None
        convolution_53: "bf16[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_34, convert_element_type_160, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 232)
        add_265: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_321, 1)
        convert_element_type_161: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_53, torch.float32)
        var_mean_53 = torch.ops.aten.var_mean.correction(convert_element_type_161, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_161 = None
        getitem_134: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = var_mean_53[0]
        getitem_135: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = var_mean_53[1];  var_mean_53 = None
        add_266: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_134, 1e-05)
        rsqrt_53: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_266);  add_266 = None
        sub_53: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_53, getitem_135)
        mul_371: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = None
        squeeze_159: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_135, [0, 2, 3]);  getitem_135 = None
        squeeze_160: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2, 3]);  rsqrt_53 = None
        mul_372: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_159, 0.1)
        mul_373: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_322, 0.9)
        add_267: "f32[232][1]cuda:0" = torch.ops.aten.add.Tensor(mul_372, mul_373);  mul_372 = mul_373 = None
        squeeze_161: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_134, [0, 2, 3]);  getitem_134 = None
        mul_374: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_161, 1.0001594642002871);  squeeze_161 = None
        mul_375: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_374, 0.1);  mul_374 = None
        mul_376: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_323, 0.9)
        add_268: "f32[232][1]cuda:0" = torch.ops.aten.add.Tensor(mul_375, mul_376);  mul_375 = mul_376 = None
        unsqueeze_212: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_324, -1)
        unsqueeze_213: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
        mul_377: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_371, unsqueeze_213);  mul_371 = unsqueeze_213 = None
        unsqueeze_214: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_325, -1);  primals_325 = None
        unsqueeze_215: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
        add_269: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_377, unsqueeze_215);  mul_377 = unsqueeze_215 = None
        convert_element_type_162: "bf16[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_269, torch.bfloat16);  add_269 = None
        convert_element_type_163: "bf16[232, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_326, torch.bfloat16);  primals_326 = None
        convolution_54: "bf16[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_162, convert_element_type_163, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_270: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_327, 1)
        convert_element_type_164: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_54, torch.float32)
        var_mean_54 = torch.ops.aten.var_mean.correction(convert_element_type_164, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_164 = None
        getitem_136: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = var_mean_54[0]
        getitem_137: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = var_mean_54[1];  var_mean_54 = None
        add_271: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_136, 1e-05)
        rsqrt_54: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_271);  add_271 = None
        sub_54: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_54, getitem_137)
        mul_378: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = None
        squeeze_162: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_137, [0, 2, 3])
        mul_379: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_162, 0.1);  squeeze_162 = None
        mul_380: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_328, 0.9)
        add_272: "f32[232][1]cuda:0" = torch.ops.aten.add.Tensor(mul_379, mul_380);  mul_379 = mul_380 = None
        squeeze_164: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_136, [0, 2, 3]);  getitem_136 = None
        mul_381: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_164, 1.0001594642002871);  squeeze_164 = None
        mul_382: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_381, 0.1);  mul_381 = None
        mul_383: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_329, 0.9)
        add_273: "f32[232][1]cuda:0" = torch.ops.aten.add.Tensor(mul_382, mul_383);  mul_382 = mul_383 = None
        unsqueeze_216: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_330, -1)
        unsqueeze_217: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
        mul_384: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_378, unsqueeze_217);  mul_378 = unsqueeze_217 = None
        unsqueeze_218: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_331, -1)
        unsqueeze_219: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
        add_274: "f32[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_384, unsqueeze_219);  mul_384 = unsqueeze_219 = None
        convert_element_type_165: "bf16[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_274, torch.bfloat16);  add_274 = None
        relu_35: "bf16[128, 232, 7, 7][11368, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_165);  convert_element_type_165 = None
        cat_15: "bf16[128, 464, 7, 7][22736, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([getitem_130, relu_35], 1);  getitem_130 = relu_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        view_30: "bf16[128, 2, 232, 7, 7][22736, 11368, 49, 7, 1]cuda:0" = torch.ops.aten.reshape.default(cat_15, [128, 2, 232, 7, 7]);  cat_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_15: "bf16[128, 232, 2, 7, 7][22736, 49, 11368, 7, 1]cuda:0" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3, 4]);  view_30 = None
        clone_15: "bf16[128, 232, 2, 7, 7][22736, 98, 49, 7, 1]cuda:0" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_31: "bf16[128, 464, 7, 7][22736, 49, 7, 1]cuda:0" = torch.ops.aten.reshape.default(clone_15, [128, 464, 7, 7]);  clone_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:160 in _forward_impl, code: x = self.conv5(x)
        convert_element_type_166: "bf16[1024, 464, 1, 1][464, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_332, torch.bfloat16);  primals_332 = None
        convolution_55: "bf16[128, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(view_31, convert_element_type_166, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_275: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_333, 1)
        convert_element_type_167: "f32[128, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_55, torch.float32)
        var_mean_55 = torch.ops.aten.var_mean.correction(convert_element_type_167, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_167 = None
        getitem_138: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_55[0]
        getitem_139: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_55[1];  var_mean_55 = None
        add_276: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_138, 1e-05)
        rsqrt_55: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_276);  add_276 = None
        sub_55: "f32[128, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_55, getitem_139)
        mul_385: "f32[128, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = None
        squeeze_165: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_139, [0, 2, 3])
        mul_386: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_165, 0.1);  squeeze_165 = None
        mul_387: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_334, 0.9)
        add_277: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_386, mul_387);  mul_386 = mul_387 = None
        squeeze_167: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_138, [0, 2, 3]);  getitem_138 = None
        mul_388: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_167, 1.0001594642002871);  squeeze_167 = None
        mul_389: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_388, 0.1);  mul_388 = None
        mul_390: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_335, 0.9)
        add_278: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_389, mul_390);  mul_389 = mul_390 = None
        unsqueeze_220: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_336, -1)
        unsqueeze_221: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
        mul_391: "f32[128, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_385, unsqueeze_221);  mul_385 = unsqueeze_221 = None
        unsqueeze_222: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_337, -1)
        unsqueeze_223: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
        add_279: "f32[128, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_391, unsqueeze_223);  mul_391 = unsqueeze_223 = None
        convert_element_type_168: "bf16[128, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_279, torch.bfloat16);  add_279 = None
        relu_36: "bf16[128, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_168);  convert_element_type_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:161 in _forward_impl, code: x = x.mean([2, 3])  # globalpool
        mean: "bf16[128, 1024][1024, 1]cuda:0" = torch.ops.aten.mean.dim(relu_36, [2, 3]);  relu_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:162 in _forward_impl, code: x = self.fc(x)
        convert_element_type_169: "bf16[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_339, torch.bfloat16);  primals_339 = None
        convert_element_type_170: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_338, torch.bfloat16);  primals_338 = None
        permute_16: "bf16[1024, 1000][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_170, [1, 0]);  convert_element_type_170 = None
        addmm: "bf16[128, 1000][1000, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_169, mean, permute_16);  convert_element_type_169 = None
        permute_17: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        unsqueeze_250: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_159, 0);  squeeze_159 = None
        unsqueeze_251: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 2);  unsqueeze_250 = None
        unsqueeze_252: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 3);  unsqueeze_251 = None
        unsqueeze_262: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_156, 0);  squeeze_156 = None
        unsqueeze_263: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 2);  unsqueeze_262 = None
        unsqueeze_264: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 3);  unsqueeze_263 = None
        unsqueeze_286: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_150, 0);  squeeze_150 = None
        unsqueeze_287: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 2);  unsqueeze_286 = None
        unsqueeze_288: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 3);  unsqueeze_287 = None
        unsqueeze_298: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_147, 0);  squeeze_147 = None
        unsqueeze_299: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 2);  unsqueeze_298 = None
        unsqueeze_300: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 3);  unsqueeze_299 = None
        unsqueeze_322: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_141, 0);  squeeze_141 = None
        unsqueeze_323: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 2);  unsqueeze_322 = None
        unsqueeze_324: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 3);  unsqueeze_323 = None
        unsqueeze_334: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
        unsqueeze_335: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 2);  unsqueeze_334 = None
        unsqueeze_336: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 3);  unsqueeze_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:97 in forward, code: out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        unsqueeze_358: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
        unsqueeze_359: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 2);  unsqueeze_358 = None
        unsqueeze_360: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 3);  unsqueeze_359 = None
        unsqueeze_370: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
        unsqueeze_371: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 2);  unsqueeze_370 = None
        unsqueeze_372: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 3);  unsqueeze_371 = None
        unsqueeze_394: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
        unsqueeze_395: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 2);  unsqueeze_394 = None
        unsqueeze_396: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 3);  unsqueeze_395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        unsqueeze_418: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
        unsqueeze_419: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 2);  unsqueeze_418 = None
        unsqueeze_420: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 3);  unsqueeze_419 = None
        unsqueeze_430: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
        unsqueeze_431: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 2);  unsqueeze_430 = None
        unsqueeze_432: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 3);  unsqueeze_431 = None
        unsqueeze_454: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
        unsqueeze_455: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_454, 2);  unsqueeze_454 = None
        unsqueeze_456: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 3);  unsqueeze_455 = None
        unsqueeze_466: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
        unsqueeze_467: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 2);  unsqueeze_466 = None
        unsqueeze_468: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 3);  unsqueeze_467 = None
        unsqueeze_490: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
        unsqueeze_491: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 2);  unsqueeze_490 = None
        unsqueeze_492: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 3);  unsqueeze_491 = None
        unsqueeze_502: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
        unsqueeze_503: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_502, 2);  unsqueeze_502 = None
        unsqueeze_504: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 3);  unsqueeze_503 = None
        unsqueeze_526: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
        unsqueeze_527: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 2);  unsqueeze_526 = None
        unsqueeze_528: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 3);  unsqueeze_527 = None
        unsqueeze_538: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
        unsqueeze_539: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 2);  unsqueeze_538 = None
        unsqueeze_540: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 3);  unsqueeze_539 = None
        unsqueeze_562: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
        unsqueeze_563: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_562, 2);  unsqueeze_562 = None
        unsqueeze_564: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 3);  unsqueeze_563 = None
        unsqueeze_574: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
        unsqueeze_575: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_574, 2);  unsqueeze_574 = None
        unsqueeze_576: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 3);  unsqueeze_575 = None
        unsqueeze_598: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
        unsqueeze_599: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_598, 2);  unsqueeze_598 = None
        unsqueeze_600: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 3);  unsqueeze_599 = None
        unsqueeze_610: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
        unsqueeze_611: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_610, 2);  unsqueeze_610 = None
        unsqueeze_612: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 3);  unsqueeze_611 = None
        unsqueeze_634: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
        unsqueeze_635: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_634, 2);  unsqueeze_634 = None
        unsqueeze_636: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 3);  unsqueeze_635 = None
        unsqueeze_646: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
        unsqueeze_647: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_646, 2);  unsqueeze_646 = None
        unsqueeze_648: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 3);  unsqueeze_647 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:97 in forward, code: out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        unsqueeze_670: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
        unsqueeze_671: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_670, 2);  unsqueeze_670 = None
        unsqueeze_672: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 3);  unsqueeze_671 = None
        unsqueeze_682: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
        unsqueeze_683: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_682, 2);  unsqueeze_682 = None
        unsqueeze_684: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 3);  unsqueeze_683 = None
        unsqueeze_706: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
        unsqueeze_707: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_706, 2);  unsqueeze_706 = None
        unsqueeze_708: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 3);  unsqueeze_707 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        unsqueeze_730: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
        unsqueeze_731: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_730, 2);  unsqueeze_730 = None
        unsqueeze_732: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 3);  unsqueeze_731 = None
        unsqueeze_742: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
        unsqueeze_743: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_742, 2);  unsqueeze_742 = None
        unsqueeze_744: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_743, 3);  unsqueeze_743 = None
        unsqueeze_766: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
        unsqueeze_767: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_766, 2);  unsqueeze_766 = None
        unsqueeze_768: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_767, 3);  unsqueeze_767 = None
        unsqueeze_778: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
        unsqueeze_779: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_778, 2);  unsqueeze_778 = None
        unsqueeze_780: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_779, 3);  unsqueeze_779 = None
        unsqueeze_802: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
        unsqueeze_803: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_802, 2);  unsqueeze_802 = None
        unsqueeze_804: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_803, 3);  unsqueeze_803 = None
        unsqueeze_814: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
        unsqueeze_815: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_814, 2);  unsqueeze_814 = None
        unsqueeze_816: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_815, 3);  unsqueeze_815 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:97 in forward, code: out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        unsqueeze_838: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
        unsqueeze_839: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_838, 2);  unsqueeze_838 = None
        unsqueeze_840: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_839, 3);  unsqueeze_839 = None
        unsqueeze_850: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
        unsqueeze_851: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_850, 2);  unsqueeze_850 = None
        unsqueeze_852: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_851, 3);  unsqueeze_851 = None
        unsqueeze_874: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
        unsqueeze_875: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_874, 2);  unsqueeze_874 = None
        unsqueeze_876: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_875, 3);  unsqueeze_875 = None

        # No stacktrace found for following nodes
        copy_: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_3, add);  primals_3 = add = copy_ = None
        copy__1: "f32[24][1]cuda:0" = torch.ops.aten.copy_.default(primals_4, add_2);  primals_4 = add_2 = copy__1 = None
        copy__2: "f32[24][1]cuda:0" = torch.ops.aten.copy_.default(primals_5, add_3);  primals_5 = add_3 = copy__2 = None
        copy__3: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_9, add_5);  primals_9 = add_5 = copy__3 = None
        copy__4: "f32[24][1]cuda:0" = torch.ops.aten.copy_.default(primals_10, add_7);  primals_10 = add_7 = copy__4 = None
        copy__5: "f32[24][1]cuda:0" = torch.ops.aten.copy_.default(primals_11, add_8);  primals_11 = add_8 = copy__5 = None
        copy__6: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_15, add_10);  primals_15 = add_10 = copy__6 = None
        copy__7: "f32[58][1]cuda:0" = torch.ops.aten.copy_.default(primals_16, add_12);  primals_16 = add_12 = copy__7 = None
        copy__8: "f32[58][1]cuda:0" = torch.ops.aten.copy_.default(primals_17, add_13);  primals_17 = add_13 = copy__8 = None
        copy__9: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_21, add_15);  primals_21 = add_15 = copy__9 = None
        copy__10: "f32[58][1]cuda:0" = torch.ops.aten.copy_.default(primals_22, add_17);  primals_22 = add_17 = copy__10 = None
        copy__11: "f32[58][1]cuda:0" = torch.ops.aten.copy_.default(primals_23, add_18);  primals_23 = add_18 = copy__11 = None
        copy__12: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_27, add_20);  primals_27 = add_20 = copy__12 = None
        copy__13: "f32[58][1]cuda:0" = torch.ops.aten.copy_.default(primals_28, add_22);  primals_28 = add_22 = copy__13 = None
        copy__14: "f32[58][1]cuda:0" = torch.ops.aten.copy_.default(primals_29, add_23);  primals_29 = add_23 = copy__14 = None
        copy__15: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_33, add_25);  primals_33 = add_25 = copy__15 = None
        copy__16: "f32[58][1]cuda:0" = torch.ops.aten.copy_.default(primals_34, add_27);  primals_34 = add_27 = copy__16 = None
        copy__17: "f32[58][1]cuda:0" = torch.ops.aten.copy_.default(primals_35, add_28);  primals_35 = add_28 = copy__17 = None
        copy__18: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_39, add_30);  primals_39 = add_30 = copy__18 = None
        copy__19: "f32[58][1]cuda:0" = torch.ops.aten.copy_.default(primals_40, add_32);  primals_40 = add_32 = copy__19 = None
        copy__20: "f32[58][1]cuda:0" = torch.ops.aten.copy_.default(primals_41, add_33);  primals_41 = add_33 = copy__20 = None
        copy__21: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_45, add_35);  primals_45 = add_35 = copy__21 = None
        copy__22: "f32[58][1]cuda:0" = torch.ops.aten.copy_.default(primals_46, add_37);  primals_46 = add_37 = copy__22 = None
        copy__23: "f32[58][1]cuda:0" = torch.ops.aten.copy_.default(primals_47, add_38);  primals_47 = add_38 = copy__23 = None
        copy__24: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_51, add_40);  primals_51 = add_40 = copy__24 = None
        copy__25: "f32[58][1]cuda:0" = torch.ops.aten.copy_.default(primals_52, add_42);  primals_52 = add_42 = copy__25 = None
        copy__26: "f32[58][1]cuda:0" = torch.ops.aten.copy_.default(primals_53, add_43);  primals_53 = add_43 = copy__26 = None
        copy__27: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_57, add_45);  primals_57 = add_45 = copy__27 = None
        copy__28: "f32[58][1]cuda:0" = torch.ops.aten.copy_.default(primals_58, add_47);  primals_58 = add_47 = copy__28 = None
        copy__29: "f32[58][1]cuda:0" = torch.ops.aten.copy_.default(primals_59, add_48);  primals_59 = add_48 = copy__29 = None
        copy__30: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_63, add_50);  primals_63 = add_50 = copy__30 = None
        copy__31: "f32[58][1]cuda:0" = torch.ops.aten.copy_.default(primals_64, add_52);  primals_64 = add_52 = copy__31 = None
        copy__32: "f32[58][1]cuda:0" = torch.ops.aten.copy_.default(primals_65, add_53);  primals_65 = add_53 = copy__32 = None
        copy__33: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_69, add_55);  primals_69 = add_55 = copy__33 = None
        copy__34: "f32[58][1]cuda:0" = torch.ops.aten.copy_.default(primals_70, add_57);  primals_70 = add_57 = copy__34 = None
        copy__35: "f32[58][1]cuda:0" = torch.ops.aten.copy_.default(primals_71, add_58);  primals_71 = add_58 = copy__35 = None
        copy__36: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_75, add_60);  primals_75 = add_60 = copy__36 = None
        copy__37: "f32[58][1]cuda:0" = torch.ops.aten.copy_.default(primals_76, add_62);  primals_76 = add_62 = copy__37 = None
        copy__38: "f32[58][1]cuda:0" = torch.ops.aten.copy_.default(primals_77, add_63);  primals_77 = add_63 = copy__38 = None
        copy__39: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_81, add_65);  primals_81 = add_65 = copy__39 = None
        copy__40: "f32[58][1]cuda:0" = torch.ops.aten.copy_.default(primals_82, add_67);  primals_82 = add_67 = copy__40 = None
        copy__41: "f32[58][1]cuda:0" = torch.ops.aten.copy_.default(primals_83, add_68);  primals_83 = add_68 = copy__41 = None
        copy__42: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_87, add_70);  primals_87 = add_70 = copy__42 = None
        copy__43: "f32[58][1]cuda:0" = torch.ops.aten.copy_.default(primals_88, add_72);  primals_88 = add_72 = copy__43 = None
        copy__44: "f32[58][1]cuda:0" = torch.ops.aten.copy_.default(primals_89, add_73);  primals_89 = add_73 = copy__44 = None
        copy__45: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_93, add_75);  primals_93 = add_75 = copy__45 = None
        copy__46: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_94, add_77);  primals_94 = add_77 = copy__46 = None
        copy__47: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_95, add_78);  primals_95 = add_78 = copy__47 = None
        copy__48: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_99, add_80);  primals_99 = add_80 = copy__48 = None
        copy__49: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_100, add_82);  primals_100 = add_82 = copy__49 = None
        copy__50: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_101, add_83);  primals_101 = add_83 = copy__50 = None
        copy__51: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_105, add_85);  primals_105 = add_85 = copy__51 = None
        copy__52: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_106, add_87);  primals_106 = add_87 = copy__52 = None
        copy__53: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_107, add_88);  primals_107 = add_88 = copy__53 = None
        copy__54: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_111, add_90);  primals_111 = add_90 = copy__54 = None
        copy__55: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_112, add_92);  primals_112 = add_92 = copy__55 = None
        copy__56: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_113, add_93);  primals_113 = add_93 = copy__56 = None
        copy__57: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_117, add_95);  primals_117 = add_95 = copy__57 = None
        copy__58: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_118, add_97);  primals_118 = add_97 = copy__58 = None
        copy__59: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_119, add_98);  primals_119 = add_98 = copy__59 = None
        copy__60: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_123, add_100);  primals_123 = add_100 = copy__60 = None
        copy__61: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_124, add_102);  primals_124 = add_102 = copy__61 = None
        copy__62: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_125, add_103);  primals_125 = add_103 = copy__62 = None
        copy__63: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_129, add_105);  primals_129 = add_105 = copy__63 = None
        copy__64: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_130, add_107);  primals_130 = add_107 = copy__64 = None
        copy__65: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_131, add_108);  primals_131 = add_108 = copy__65 = None
        copy__66: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_135, add_110);  primals_135 = add_110 = copy__66 = None
        copy__67: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_136, add_112);  primals_136 = add_112 = copy__67 = None
        copy__68: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_137, add_113);  primals_137 = add_113 = copy__68 = None
        copy__69: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_141, add_115);  primals_141 = add_115 = copy__69 = None
        copy__70: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_142, add_117);  primals_142 = add_117 = copy__70 = None
        copy__71: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_143, add_118);  primals_143 = add_118 = copy__71 = None
        copy__72: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_147, add_120);  primals_147 = add_120 = copy__72 = None
        copy__73: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_148, add_122);  primals_148 = add_122 = copy__73 = None
        copy__74: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_149, add_123);  primals_149 = add_123 = copy__74 = None
        copy__75: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_153, add_125);  primals_153 = add_125 = copy__75 = None
        copy__76: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_154, add_127);  primals_154 = add_127 = copy__76 = None
        copy__77: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_155, add_128);  primals_155 = add_128 = copy__77 = None
        copy__78: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_159, add_130);  primals_159 = add_130 = copy__78 = None
        copy__79: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_160, add_132);  primals_160 = add_132 = copy__79 = None
        copy__80: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_161, add_133);  primals_161 = add_133 = copy__80 = None
        copy__81: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_165, add_135);  primals_165 = add_135 = copy__81 = None
        copy__82: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_166, add_137);  primals_166 = add_137 = copy__82 = None
        copy__83: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_167, add_138);  primals_167 = add_138 = copy__83 = None
        copy__84: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_171, add_140);  primals_171 = add_140 = copy__84 = None
        copy__85: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_172, add_142);  primals_172 = add_142 = copy__85 = None
        copy__86: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_173, add_143);  primals_173 = add_143 = copy__86 = None
        copy__87: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_177, add_145);  primals_177 = add_145 = copy__87 = None
        copy__88: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_178, add_147);  primals_178 = add_147 = copy__88 = None
        copy__89: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_179, add_148);  primals_179 = add_148 = copy__89 = None
        copy__90: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_183, add_150);  primals_183 = add_150 = copy__90 = None
        copy__91: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_184, add_152);  primals_184 = add_152 = copy__91 = None
        copy__92: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_185, add_153);  primals_185 = add_153 = copy__92 = None
        copy__93: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_189, add_155);  primals_189 = add_155 = copy__93 = None
        copy__94: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_190, add_157);  primals_190 = add_157 = copy__94 = None
        copy__95: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_191, add_158);  primals_191 = add_158 = copy__95 = None
        copy__96: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_195, add_160);  primals_195 = add_160 = copy__96 = None
        copy__97: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_196, add_162);  primals_196 = add_162 = copy__97 = None
        copy__98: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_197, add_163);  primals_197 = add_163 = copy__98 = None
        copy__99: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_201, add_165);  primals_201 = add_165 = copy__99 = None
        copy__100: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_202, add_167);  primals_202 = add_167 = copy__100 = None
        copy__101: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_203, add_168);  primals_203 = add_168 = copy__101 = None
        copy__102: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_207, add_170);  primals_207 = add_170 = copy__102 = None
        copy__103: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_208, add_172);  primals_208 = add_172 = copy__103 = None
        copy__104: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_209, add_173);  primals_209 = add_173 = copy__104 = None
        copy__105: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_213, add_175);  primals_213 = add_175 = copy__105 = None
        copy__106: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_214, add_177);  primals_214 = add_177 = copy__106 = None
        copy__107: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_215, add_178);  primals_215 = add_178 = copy__107 = None
        copy__108: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_219, add_180);  primals_219 = add_180 = copy__108 = None
        copy__109: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_220, add_182);  primals_220 = add_182 = copy__109 = None
        copy__110: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_221, add_183);  primals_221 = add_183 = copy__110 = None
        copy__111: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_225, add_185);  primals_225 = add_185 = copy__111 = None
        copy__112: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_226, add_187);  primals_226 = add_187 = copy__112 = None
        copy__113: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_227, add_188);  primals_227 = add_188 = copy__113 = None
        copy__114: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_231, add_190);  primals_231 = add_190 = copy__114 = None
        copy__115: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_232, add_192);  primals_232 = add_192 = copy__115 = None
        copy__116: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_233, add_193);  primals_233 = add_193 = copy__116 = None
        copy__117: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_237, add_195);  primals_237 = add_195 = copy__117 = None
        copy__118: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_238, add_197);  primals_238 = add_197 = copy__118 = None
        copy__119: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_239, add_198);  primals_239 = add_198 = copy__119 = None
        copy__120: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_243, add_200);  primals_243 = add_200 = copy__120 = None
        copy__121: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_244, add_202);  primals_244 = add_202 = copy__121 = None
        copy__122: "f32[116][1]cuda:0" = torch.ops.aten.copy_.default(primals_245, add_203);  primals_245 = add_203 = copy__122 = None
        copy__123: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_249, add_205);  primals_249 = add_205 = copy__123 = None
        copy__124: "f32[232][1]cuda:0" = torch.ops.aten.copy_.default(primals_250, add_207);  primals_250 = add_207 = copy__124 = None
        copy__125: "f32[232][1]cuda:0" = torch.ops.aten.copy_.default(primals_251, add_208);  primals_251 = add_208 = copy__125 = None
        copy__126: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_255, add_210);  primals_255 = add_210 = copy__126 = None
        copy__127: "f32[232][1]cuda:0" = torch.ops.aten.copy_.default(primals_256, add_212);  primals_256 = add_212 = copy__127 = None
        copy__128: "f32[232][1]cuda:0" = torch.ops.aten.copy_.default(primals_257, add_213);  primals_257 = add_213 = copy__128 = None
        copy__129: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_261, add_215);  primals_261 = add_215 = copy__129 = None
        copy__130: "f32[232][1]cuda:0" = torch.ops.aten.copy_.default(primals_262, add_217);  primals_262 = add_217 = copy__130 = None
        copy__131: "f32[232][1]cuda:0" = torch.ops.aten.copy_.default(primals_263, add_218);  primals_263 = add_218 = copy__131 = None
        copy__132: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_267, add_220);  primals_267 = add_220 = copy__132 = None
        copy__133: "f32[232][1]cuda:0" = torch.ops.aten.copy_.default(primals_268, add_222);  primals_268 = add_222 = copy__133 = None
        copy__134: "f32[232][1]cuda:0" = torch.ops.aten.copy_.default(primals_269, add_223);  primals_269 = add_223 = copy__134 = None
        copy__135: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_273, add_225);  primals_273 = add_225 = copy__135 = None
        copy__136: "f32[232][1]cuda:0" = torch.ops.aten.copy_.default(primals_274, add_227);  primals_274 = add_227 = copy__136 = None
        copy__137: "f32[232][1]cuda:0" = torch.ops.aten.copy_.default(primals_275, add_228);  primals_275 = add_228 = copy__137 = None
        copy__138: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_279, add_230);  primals_279 = add_230 = copy__138 = None
        copy__139: "f32[232][1]cuda:0" = torch.ops.aten.copy_.default(primals_280, add_232);  primals_280 = add_232 = copy__139 = None
        copy__140: "f32[232][1]cuda:0" = torch.ops.aten.copy_.default(primals_281, add_233);  primals_281 = add_233 = copy__140 = None
        copy__141: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_285, add_235);  primals_285 = add_235 = copy__141 = None
        copy__142: "f32[232][1]cuda:0" = torch.ops.aten.copy_.default(primals_286, add_237);  primals_286 = add_237 = copy__142 = None
        copy__143: "f32[232][1]cuda:0" = torch.ops.aten.copy_.default(primals_287, add_238);  primals_287 = add_238 = copy__143 = None
        copy__144: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_291, add_240);  primals_291 = add_240 = copy__144 = None
        copy__145: "f32[232][1]cuda:0" = torch.ops.aten.copy_.default(primals_292, add_242);  primals_292 = add_242 = copy__145 = None
        copy__146: "f32[232][1]cuda:0" = torch.ops.aten.copy_.default(primals_293, add_243);  primals_293 = add_243 = copy__146 = None
        copy__147: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_297, add_245);  primals_297 = add_245 = copy__147 = None
        copy__148: "f32[232][1]cuda:0" = torch.ops.aten.copy_.default(primals_298, add_247);  primals_298 = add_247 = copy__148 = None
        copy__149: "f32[232][1]cuda:0" = torch.ops.aten.copy_.default(primals_299, add_248);  primals_299 = add_248 = copy__149 = None
        copy__150: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_303, add_250);  primals_303 = add_250 = copy__150 = None
        copy__151: "f32[232][1]cuda:0" = torch.ops.aten.copy_.default(primals_304, add_252);  primals_304 = add_252 = copy__151 = None
        copy__152: "f32[232][1]cuda:0" = torch.ops.aten.copy_.default(primals_305, add_253);  primals_305 = add_253 = copy__152 = None
        copy__153: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_309, add_255);  primals_309 = add_255 = copy__153 = None
        copy__154: "f32[232][1]cuda:0" = torch.ops.aten.copy_.default(primals_310, add_257);  primals_310 = add_257 = copy__154 = None
        copy__155: "f32[232][1]cuda:0" = torch.ops.aten.copy_.default(primals_311, add_258);  primals_311 = add_258 = copy__155 = None
        copy__156: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_315, add_260);  primals_315 = add_260 = copy__156 = None
        copy__157: "f32[232][1]cuda:0" = torch.ops.aten.copy_.default(primals_316, add_262);  primals_316 = add_262 = copy__157 = None
        copy__158: "f32[232][1]cuda:0" = torch.ops.aten.copy_.default(primals_317, add_263);  primals_317 = add_263 = copy__158 = None
        copy__159: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_321, add_265);  primals_321 = add_265 = copy__159 = None
        copy__160: "f32[232][1]cuda:0" = torch.ops.aten.copy_.default(primals_322, add_267);  primals_322 = add_267 = copy__160 = None
        copy__161: "f32[232][1]cuda:0" = torch.ops.aten.copy_.default(primals_323, add_268);  primals_323 = add_268 = copy__161 = None
        copy__162: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_327, add_270);  primals_327 = add_270 = copy__162 = None
        copy__163: "f32[232][1]cuda:0" = torch.ops.aten.copy_.default(primals_328, add_272);  primals_328 = add_272 = copy__163 = None
        copy__164: "f32[232][1]cuda:0" = torch.ops.aten.copy_.default(primals_329, add_273);  primals_329 = add_273 = copy__164 = None
        copy__165: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_333, add_275);  primals_333 = add_275 = copy__165 = None
        copy__166: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_334, add_277);  primals_334 = add_277 = copy__166 = None
        copy__167: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_335, add_278);  primals_335 = add_278 = copy__167 = None
        return (addmm, primals_6, primals_7, primals_12, primals_18, primals_19, primals_24, primals_30, primals_36, primals_37, primals_42, primals_48, primals_54, primals_55, primals_60, primals_66, primals_72, primals_73, primals_78, primals_84, primals_90, primals_91, primals_96, primals_102, primals_103, primals_108, primals_114, primals_120, primals_121, primals_126, primals_132, primals_138, primals_139, primals_144, primals_150, primals_156, primals_157, primals_162, primals_168, primals_174, primals_175, primals_180, primals_186, primals_192, primals_193, primals_198, primals_204, primals_210, primals_211, primals_216, primals_222, primals_228, primals_229, primals_234, primals_240, primals_246, primals_247, primals_252, primals_258, primals_259, primals_264, primals_270, primals_276, primals_277, primals_282, primals_288, primals_294, primals_295, primals_300, primals_306, primals_312, primals_313, primals_318, primals_324, primals_330, primals_331, primals_336, primals_337, convert_element_type, convert_element_type_1, convolution, getitem_1, rsqrt, getitem_2, getitem_3, convert_element_type_4, convolution_1, squeeze_4, convert_element_type_6, convert_element_type_7, convolution_2, getitem_7, rsqrt_2, convert_element_type_10, convolution_3, squeeze_10, relu_2, convert_element_type_13, convolution_4, squeeze_13, convert_element_type_15, convert_element_type_16, convolution_5, getitem_13, rsqrt_5, getitem_15, convert_element_type_19, convolution_6, squeeze_19, relu_4, convert_element_type_22, convolution_7, squeeze_22, convert_element_type_24, convert_element_type_25, convolution_8, getitem_21, rsqrt_8, getitem_23, convert_element_type_28, convolution_9, squeeze_28, relu_6, convert_element_type_31, convolution_10, squeeze_31, convert_element_type_33, convert_element_type_34, convolution_11, getitem_29, rsqrt_11, getitem_31, convert_element_type_37, convolution_12, squeeze_37, relu_8, convert_element_type_40, convolution_13, squeeze_40, convert_element_type_42, convert_element_type_43, convolution_14, getitem_37, rsqrt_14, view_7, convert_element_type_46, convolution_15, squeeze_46, convert_element_type_48, convert_element_type_49, convolution_16, getitem_41, rsqrt_16, convert_element_type_52, convolution_17, squeeze_52, relu_11, convert_element_type_55, convolution_18, squeeze_55, convert_element_type_57, convert_element_type_58, convolution_19, getitem_47, rsqrt_19, getitem_49, convert_element_type_61, convolution_20, squeeze_61, relu_13, convert_element_type_64, convolution_21, squeeze_64, convert_element_type_66, convert_element_type_67, convolution_22, getitem_55, rsqrt_22, getitem_57, convert_element_type_70, convolution_23, squeeze_70, relu_15, convert_element_type_73, convolution_24, squeeze_73, convert_element_type_75, convert_element_type_76, convolution_25, getitem_63, rsqrt_25, getitem_65, convert_element_type_79, convolution_26, squeeze_79, relu_17, convert_element_type_82, convolution_27, squeeze_82, convert_element_type_84, convert_element_type_85, convolution_28, getitem_71, rsqrt_28, getitem_73, convert_element_type_88, convolution_29, squeeze_88, relu_19, convert_element_type_91, convolution_30, squeeze_91, convert_element_type_93, convert_element_type_94, convolution_31, getitem_79, rsqrt_31, getitem_81, convert_element_type_97, convolution_32, squeeze_97, relu_21, convert_element_type_100, convolution_33, squeeze_100, convert_element_type_102, convert_element_type_103, convolution_34, getitem_87, rsqrt_34, getitem_89, convert_element_type_106, convolution_35, squeeze_106, relu_23, convert_element_type_109, convolution_36, squeeze_109, convert_element_type_111, convert_element_type_112, convolution_37, getitem_95, rsqrt_37, getitem_97, convert_element_type_115, convolution_38, squeeze_115, relu_25, convert_element_type_118, convolution_39, squeeze_118, convert_element_type_120, convert_element_type_121, convolution_40, getitem_103, rsqrt_40, view_23, convert_element_type_124, convolution_41, squeeze_124, convert_element_type_126, convert_element_type_127, convolution_42, getitem_107, rsqrt_42, convert_element_type_130, convolution_43, squeeze_130, relu_28, convert_element_type_133, convolution_44, squeeze_133, convert_element_type_135, convert_element_type_136, convolution_45, getitem_113, rsqrt_45, getitem_115, convert_element_type_139, convolution_46, squeeze_139, relu_30, convert_element_type_142, convolution_47, squeeze_142, convert_element_type_144, convert_element_type_145, convolution_48, getitem_121, rsqrt_48, getitem_123, convert_element_type_148, convolution_49, squeeze_148, relu_32, convert_element_type_151, convolution_50, squeeze_151, convert_element_type_153, convert_element_type_154, convolution_51, getitem_129, rsqrt_51, getitem_131, convert_element_type_157, convolution_52, squeeze_157, relu_34, convert_element_type_160, convolution_53, squeeze_160, convert_element_type_162, convert_element_type_163, convolution_54, getitem_137, rsqrt_54, view_31, convert_element_type_166, convolution_55, getitem_139, rsqrt_55, mean, permute_17, unsqueeze_252, unsqueeze_264, unsqueeze_288, unsqueeze_300, unsqueeze_324, unsqueeze_336, unsqueeze_360, unsqueeze_372, unsqueeze_396, unsqueeze_420, unsqueeze_432, unsqueeze_456, unsqueeze_468, unsqueeze_492, unsqueeze_504, unsqueeze_528, unsqueeze_540, unsqueeze_564, unsqueeze_576, unsqueeze_600, unsqueeze_612, unsqueeze_636, unsqueeze_648, unsqueeze_672, unsqueeze_684, unsqueeze_708, unsqueeze_732, unsqueeze_744, unsqueeze_768, unsqueeze_780, unsqueeze_804, unsqueeze_816, unsqueeze_840, unsqueeze_852, unsqueeze_876)
