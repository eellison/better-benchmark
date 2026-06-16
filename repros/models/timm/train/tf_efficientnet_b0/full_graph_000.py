class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[32, 3, 3, 3][27, 1, 9, 3]cuda:0", primals_2: "f32[128, 3, 224, 224][150528, 1, 672, 3]cuda:0", primals_3: "i64[][]cuda:0", primals_4: "f32[32][1]cuda:0", primals_5: "f32[32][1]cuda:0", primals_6: "f32[32][1]cuda:0", primals_7: "f32[32][1]cuda:0", primals_8: "f32[32, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_9: "i64[][]cuda:0", primals_10: "f32[32][1]cuda:0", primals_11: "f32[32][1]cuda:0", primals_12: "f32[32][1]cuda:0", primals_13: "f32[32][1]cuda:0", primals_14: "f32[8, 32, 1, 1][32, 1, 32, 32]cuda:0", primals_15: "f32[8][1]cuda:0", primals_16: "f32[32, 8, 1, 1][8, 1, 8, 8]cuda:0", primals_17: "f32[32][1]cuda:0", primals_18: "f32[16, 32, 1, 1][32, 1, 32, 32]cuda:0", primals_19: "i64[][]cuda:0", primals_20: "f32[16][1]cuda:0", primals_21: "f32[16][1]cuda:0", primals_22: "f32[16][1]cuda:0", primals_23: "f32[16][1]cuda:0", primals_24: "f32[96, 16, 1, 1][16, 1, 16, 16]cuda:0", primals_25: "i64[][]cuda:0", primals_26: "f32[96][1]cuda:0", primals_27: "f32[96][1]cuda:0", primals_28: "f32[96][1]cuda:0", primals_29: "f32[96][1]cuda:0", primals_30: "f32[96, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_31: "i64[][]cuda:0", primals_32: "f32[96][1]cuda:0", primals_33: "f32[96][1]cuda:0", primals_34: "f32[96][1]cuda:0", primals_35: "f32[96][1]cuda:0", primals_36: "f32[4, 96, 1, 1][96, 1, 96, 96]cuda:0", primals_37: "f32[4][1]cuda:0", primals_38: "f32[96, 4, 1, 1][4, 1, 4, 4]cuda:0", primals_39: "f32[96][1]cuda:0", primals_40: "f32[24, 96, 1, 1][96, 1, 96, 96]cuda:0", primals_41: "i64[][]cuda:0", primals_42: "f32[24][1]cuda:0", primals_43: "f32[24][1]cuda:0", primals_44: "f32[24][1]cuda:0", primals_45: "f32[24][1]cuda:0", primals_46: "f32[144, 24, 1, 1][24, 1, 24, 24]cuda:0", primals_47: "i64[][]cuda:0", primals_48: "f32[144][1]cuda:0", primals_49: "f32[144][1]cuda:0", primals_50: "f32[144][1]cuda:0", primals_51: "f32[144][1]cuda:0", primals_52: "f32[144, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_53: "i64[][]cuda:0", primals_54: "f32[144][1]cuda:0", primals_55: "f32[144][1]cuda:0", primals_56: "f32[144][1]cuda:0", primals_57: "f32[144][1]cuda:0", primals_58: "f32[6, 144, 1, 1][144, 1, 144, 144]cuda:0", primals_59: "f32[6][1]cuda:0", primals_60: "f32[144, 6, 1, 1][6, 1, 6, 6]cuda:0", primals_61: "f32[144][1]cuda:0", primals_62: "f32[24, 144, 1, 1][144, 1, 144, 144]cuda:0", primals_63: "i64[][]cuda:0", primals_64: "f32[24][1]cuda:0", primals_65: "f32[24][1]cuda:0", primals_66: "f32[24][1]cuda:0", primals_67: "f32[24][1]cuda:0", primals_68: "f32[144, 24, 1, 1][24, 1, 24, 24]cuda:0", primals_69: "i64[][]cuda:0", primals_70: "f32[144][1]cuda:0", primals_71: "f32[144][1]cuda:0", primals_72: "f32[144][1]cuda:0", primals_73: "f32[144][1]cuda:0", primals_74: "f32[144, 1, 5, 5][25, 1, 5, 1]cuda:0", primals_75: "i64[][]cuda:0", primals_76: "f32[144][1]cuda:0", primals_77: "f32[144][1]cuda:0", primals_78: "f32[144][1]cuda:0", primals_79: "f32[144][1]cuda:0", primals_80: "f32[6, 144, 1, 1][144, 1, 144, 144]cuda:0", primals_81: "f32[6][1]cuda:0", primals_82: "f32[144, 6, 1, 1][6, 1, 6, 6]cuda:0", primals_83: "f32[144][1]cuda:0", primals_84: "f32[40, 144, 1, 1][144, 1, 144, 144]cuda:0", primals_85: "i64[][]cuda:0", primals_86: "f32[40][1]cuda:0", primals_87: "f32[40][1]cuda:0", primals_88: "f32[40][1]cuda:0", primals_89: "f32[40][1]cuda:0", primals_90: "f32[240, 40, 1, 1][40, 1, 40, 40]cuda:0", primals_91: "i64[][]cuda:0", primals_92: "f32[240][1]cuda:0", primals_93: "f32[240][1]cuda:0", primals_94: "f32[240][1]cuda:0", primals_95: "f32[240][1]cuda:0", primals_96: "f32[240, 1, 5, 5][25, 1, 5, 1]cuda:0", primals_97: "i64[][]cuda:0", primals_98: "f32[240][1]cuda:0", primals_99: "f32[240][1]cuda:0", primals_100: "f32[240][1]cuda:0", primals_101: "f32[240][1]cuda:0", primals_102: "f32[10, 240, 1, 1][240, 1, 240, 240]cuda:0", primals_103: "f32[10][1]cuda:0", primals_104: "f32[240, 10, 1, 1][10, 1, 10, 10]cuda:0", primals_105: "f32[240][1]cuda:0", primals_106: "f32[40, 240, 1, 1][240, 1, 240, 240]cuda:0", primals_107: "i64[][]cuda:0", primals_108: "f32[40][1]cuda:0", primals_109: "f32[40][1]cuda:0", primals_110: "f32[40][1]cuda:0", primals_111: "f32[40][1]cuda:0", primals_112: "f32[240, 40, 1, 1][40, 1, 40, 40]cuda:0", primals_113: "i64[][]cuda:0", primals_114: "f32[240][1]cuda:0", primals_115: "f32[240][1]cuda:0", primals_116: "f32[240][1]cuda:0", primals_117: "f32[240][1]cuda:0", primals_118: "f32[240, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_119: "i64[][]cuda:0", primals_120: "f32[240][1]cuda:0", primals_121: "f32[240][1]cuda:0", primals_122: "f32[240][1]cuda:0", primals_123: "f32[240][1]cuda:0", primals_124: "f32[10, 240, 1, 1][240, 1, 240, 240]cuda:0", primals_125: "f32[10][1]cuda:0", primals_126: "f32[240, 10, 1, 1][10, 1, 10, 10]cuda:0", primals_127: "f32[240][1]cuda:0", primals_128: "f32[80, 240, 1, 1][240, 1, 240, 240]cuda:0", primals_129: "i64[][]cuda:0", primals_130: "f32[80][1]cuda:0", primals_131: "f32[80][1]cuda:0", primals_132: "f32[80][1]cuda:0", primals_133: "f32[80][1]cuda:0", primals_134: "f32[480, 80, 1, 1][80, 1, 80, 80]cuda:0", primals_135: "i64[][]cuda:0", primals_136: "f32[480][1]cuda:0", primals_137: "f32[480][1]cuda:0", primals_138: "f32[480][1]cuda:0", primals_139: "f32[480][1]cuda:0", primals_140: "f32[480, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_141: "i64[][]cuda:0", primals_142: "f32[480][1]cuda:0", primals_143: "f32[480][1]cuda:0", primals_144: "f32[480][1]cuda:0", primals_145: "f32[480][1]cuda:0", primals_146: "f32[20, 480, 1, 1][480, 1, 480, 480]cuda:0", primals_147: "f32[20][1]cuda:0", primals_148: "f32[480, 20, 1, 1][20, 1, 20, 20]cuda:0", primals_149: "f32[480][1]cuda:0", primals_150: "f32[80, 480, 1, 1][480, 1, 480, 480]cuda:0", primals_151: "i64[][]cuda:0", primals_152: "f32[80][1]cuda:0", primals_153: "f32[80][1]cuda:0", primals_154: "f32[80][1]cuda:0", primals_155: "f32[80][1]cuda:0", primals_156: "f32[480, 80, 1, 1][80, 1, 80, 80]cuda:0", primals_157: "i64[][]cuda:0", primals_158: "f32[480][1]cuda:0", primals_159: "f32[480][1]cuda:0", primals_160: "f32[480][1]cuda:0", primals_161: "f32[480][1]cuda:0", primals_162: "f32[480, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_163: "i64[][]cuda:0", primals_164: "f32[480][1]cuda:0", primals_165: "f32[480][1]cuda:0", primals_166: "f32[480][1]cuda:0", primals_167: "f32[480][1]cuda:0", primals_168: "f32[20, 480, 1, 1][480, 1, 480, 480]cuda:0", primals_169: "f32[20][1]cuda:0", primals_170: "f32[480, 20, 1, 1][20, 1, 20, 20]cuda:0", primals_171: "f32[480][1]cuda:0", primals_172: "f32[80, 480, 1, 1][480, 1, 480, 480]cuda:0", primals_173: "i64[][]cuda:0", primals_174: "f32[80][1]cuda:0", primals_175: "f32[80][1]cuda:0", primals_176: "f32[80][1]cuda:0", primals_177: "f32[80][1]cuda:0", primals_178: "f32[480, 80, 1, 1][80, 1, 80, 80]cuda:0", primals_179: "i64[][]cuda:0", primals_180: "f32[480][1]cuda:0", primals_181: "f32[480][1]cuda:0", primals_182: "f32[480][1]cuda:0", primals_183: "f32[480][1]cuda:0", primals_184: "f32[480, 1, 5, 5][25, 1, 5, 1]cuda:0", primals_185: "i64[][]cuda:0", primals_186: "f32[480][1]cuda:0", primals_187: "f32[480][1]cuda:0", primals_188: "f32[480][1]cuda:0", primals_189: "f32[480][1]cuda:0", primals_190: "f32[20, 480, 1, 1][480, 1, 480, 480]cuda:0", primals_191: "f32[20][1]cuda:0", primals_192: "f32[480, 20, 1, 1][20, 1, 20, 20]cuda:0", primals_193: "f32[480][1]cuda:0", primals_194: "f32[112, 480, 1, 1][480, 1, 480, 480]cuda:0", primals_195: "i64[][]cuda:0", primals_196: "f32[112][1]cuda:0", primals_197: "f32[112][1]cuda:0", primals_198: "f32[112][1]cuda:0", primals_199: "f32[112][1]cuda:0", primals_200: "f32[672, 112, 1, 1][112, 1, 112, 112]cuda:0", primals_201: "i64[][]cuda:0", primals_202: "f32[672][1]cuda:0", primals_203: "f32[672][1]cuda:0", primals_204: "f32[672][1]cuda:0", primals_205: "f32[672][1]cuda:0", primals_206: "f32[672, 1, 5, 5][25, 1, 5, 1]cuda:0", primals_207: "i64[][]cuda:0", primals_208: "f32[672][1]cuda:0", primals_209: "f32[672][1]cuda:0", primals_210: "f32[672][1]cuda:0", primals_211: "f32[672][1]cuda:0", primals_212: "f32[28, 672, 1, 1][672, 1, 672, 672]cuda:0", primals_213: "f32[28][1]cuda:0", primals_214: "f32[672, 28, 1, 1][28, 1, 28, 28]cuda:0", primals_215: "f32[672][1]cuda:0", primals_216: "f32[112, 672, 1, 1][672, 1, 672, 672]cuda:0", primals_217: "i64[][]cuda:0", primals_218: "f32[112][1]cuda:0", primals_219: "f32[112][1]cuda:0", primals_220: "f32[112][1]cuda:0", primals_221: "f32[112][1]cuda:0", primals_222: "f32[672, 112, 1, 1][112, 1, 112, 112]cuda:0", primals_223: "i64[][]cuda:0", primals_224: "f32[672][1]cuda:0", primals_225: "f32[672][1]cuda:0", primals_226: "f32[672][1]cuda:0", primals_227: "f32[672][1]cuda:0", primals_228: "f32[672, 1, 5, 5][25, 1, 5, 1]cuda:0", primals_229: "i64[][]cuda:0", primals_230: "f32[672][1]cuda:0", primals_231: "f32[672][1]cuda:0", primals_232: "f32[672][1]cuda:0", primals_233: "f32[672][1]cuda:0", primals_234: "f32[28, 672, 1, 1][672, 1, 672, 672]cuda:0", primals_235: "f32[28][1]cuda:0", primals_236: "f32[672, 28, 1, 1][28, 1, 28, 28]cuda:0", primals_237: "f32[672][1]cuda:0", primals_238: "f32[112, 672, 1, 1][672, 1, 672, 672]cuda:0", primals_239: "i64[][]cuda:0", primals_240: "f32[112][1]cuda:0", primals_241: "f32[112][1]cuda:0", primals_242: "f32[112][1]cuda:0", primals_243: "f32[112][1]cuda:0", primals_244: "f32[672, 112, 1, 1][112, 1, 112, 112]cuda:0", primals_245: "i64[][]cuda:0", primals_246: "f32[672][1]cuda:0", primals_247: "f32[672][1]cuda:0", primals_248: "f32[672][1]cuda:0", primals_249: "f32[672][1]cuda:0", primals_250: "f32[672, 1, 5, 5][25, 1, 5, 1]cuda:0", primals_251: "i64[][]cuda:0", primals_252: "f32[672][1]cuda:0", primals_253: "f32[672][1]cuda:0", primals_254: "f32[672][1]cuda:0", primals_255: "f32[672][1]cuda:0", primals_256: "f32[28, 672, 1, 1][672, 1, 672, 672]cuda:0", primals_257: "f32[28][1]cuda:0", primals_258: "f32[672, 28, 1, 1][28, 1, 28, 28]cuda:0", primals_259: "f32[672][1]cuda:0", primals_260: "f32[192, 672, 1, 1][672, 1, 672, 672]cuda:0", primals_261: "i64[][]cuda:0", primals_262: "f32[192][1]cuda:0", primals_263: "f32[192][1]cuda:0", primals_264: "f32[192][1]cuda:0", primals_265: "f32[192][1]cuda:0", primals_266: "f32[1152, 192, 1, 1][192, 1, 192, 192]cuda:0", primals_267: "i64[][]cuda:0", primals_268: "f32[1152][1]cuda:0", primals_269: "f32[1152][1]cuda:0", primals_270: "f32[1152][1]cuda:0", primals_271: "f32[1152][1]cuda:0", primals_272: "f32[1152, 1, 5, 5][25, 1, 5, 1]cuda:0", primals_273: "i64[][]cuda:0", primals_274: "f32[1152][1]cuda:0", primals_275: "f32[1152][1]cuda:0", primals_276: "f32[1152][1]cuda:0", primals_277: "f32[1152][1]cuda:0", primals_278: "f32[48, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", primals_279: "f32[48][1]cuda:0", primals_280: "f32[1152, 48, 1, 1][48, 1, 48, 48]cuda:0", primals_281: "f32[1152][1]cuda:0", primals_282: "f32[192, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", primals_283: "i64[][]cuda:0", primals_284: "f32[192][1]cuda:0", primals_285: "f32[192][1]cuda:0", primals_286: "f32[192][1]cuda:0", primals_287: "f32[192][1]cuda:0", primals_288: "f32[1152, 192, 1, 1][192, 1, 192, 192]cuda:0", primals_289: "i64[][]cuda:0", primals_290: "f32[1152][1]cuda:0", primals_291: "f32[1152][1]cuda:0", primals_292: "f32[1152][1]cuda:0", primals_293: "f32[1152][1]cuda:0", primals_294: "f32[1152, 1, 5, 5][25, 1, 5, 1]cuda:0", primals_295: "i64[][]cuda:0", primals_296: "f32[1152][1]cuda:0", primals_297: "f32[1152][1]cuda:0", primals_298: "f32[1152][1]cuda:0", primals_299: "f32[1152][1]cuda:0", primals_300: "f32[48, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", primals_301: "f32[48][1]cuda:0", primals_302: "f32[1152, 48, 1, 1][48, 1, 48, 48]cuda:0", primals_303: "f32[1152][1]cuda:0", primals_304: "f32[192, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", primals_305: "i64[][]cuda:0", primals_306: "f32[192][1]cuda:0", primals_307: "f32[192][1]cuda:0", primals_308: "f32[192][1]cuda:0", primals_309: "f32[192][1]cuda:0", primals_310: "f32[1152, 192, 1, 1][192, 1, 192, 192]cuda:0", primals_311: "i64[][]cuda:0", primals_312: "f32[1152][1]cuda:0", primals_313: "f32[1152][1]cuda:0", primals_314: "f32[1152][1]cuda:0", primals_315: "f32[1152][1]cuda:0", primals_316: "f32[1152, 1, 5, 5][25, 1, 5, 1]cuda:0", primals_317: "i64[][]cuda:0", primals_318: "f32[1152][1]cuda:0", primals_319: "f32[1152][1]cuda:0", primals_320: "f32[1152][1]cuda:0", primals_321: "f32[1152][1]cuda:0", primals_322: "f32[48, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", primals_323: "f32[48][1]cuda:0", primals_324: "f32[1152, 48, 1, 1][48, 1, 48, 48]cuda:0", primals_325: "f32[1152][1]cuda:0", primals_326: "f32[192, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", primals_327: "i64[][]cuda:0", primals_328: "f32[192][1]cuda:0", primals_329: "f32[192][1]cuda:0", primals_330: "f32[192][1]cuda:0", primals_331: "f32[192][1]cuda:0", primals_332: "f32[1152, 192, 1, 1][192, 1, 192, 192]cuda:0", primals_333: "i64[][]cuda:0", primals_334: "f32[1152][1]cuda:0", primals_335: "f32[1152][1]cuda:0", primals_336: "f32[1152][1]cuda:0", primals_337: "f32[1152][1]cuda:0", primals_338: "f32[1152, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_339: "i64[][]cuda:0", primals_340: "f32[1152][1]cuda:0", primals_341: "f32[1152][1]cuda:0", primals_342: "f32[1152][1]cuda:0", primals_343: "f32[1152][1]cuda:0", primals_344: "f32[48, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", primals_345: "f32[48][1]cuda:0", primals_346: "f32[1152, 48, 1, 1][48, 1, 48, 48]cuda:0", primals_347: "f32[1152][1]cuda:0", primals_348: "f32[320, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", primals_349: "i64[][]cuda:0", primals_350: "f32[320][1]cuda:0", primals_351: "f32[320][1]cuda:0", primals_352: "f32[320][1]cuda:0", primals_353: "f32[320][1]cuda:0", primals_354: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", primals_355: "i64[][]cuda:0", primals_356: "f32[1280][1]cuda:0", primals_357: "f32[1280][1]cuda:0", primals_358: "f32[1280][1]cuda:0", primals_359: "f32[1280][1]cuda:0", primals_360: "f32[1000, 1280][1280, 1]cuda:0", primals_361: "f32[1000][1]cuda:0"):
        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "f32[128, 3, 225, 225][151875, 1, 675, 3]cuda:0" = torch.ops.aten.constant_pad_nd.default(primals_2, [0, 1, 0, 1], 0.0);  primals_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv2d_same.py:28 in conv2d_same, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
        convert_element_type: "bf16[32, 3, 3, 3][27, 1, 9, 3]cuda:0" = torch.ops.prims.convert_element_type.default(primals_1, torch.bfloat16);  primals_1 = None
        convert_element_type_1: "bf16[128, 3, 225, 225][151875, 1, 675, 3]cuda:0" = torch.ops.prims.convert_element_type.default(constant_pad_nd, torch.bfloat16);  constant_pad_nd = None
        convolution: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_1, convert_element_type, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_2: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(convolution, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_2, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_2 = None
        getitem: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        add_1: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 0.001)
        rsqrt: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze: "f32[32][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3])
        mul_1: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze, 0.1);  squeeze = None
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
        unsqueeze_2: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_7, -1)
        unsqueeze_3: "f32[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_4: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        convert_element_type_3: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_4: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        neg: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.neg.default(convert_element_type_4)
        exp: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.exp.default(neg);  neg = None
        add_5: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_4, add_5);  convert_element_type_4 = add_5 = None
        convert_element_type_5: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:223 in forward, code: x = self.conv_dw(x)
        convert_element_type_6: "bf16[32, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_8, torch.bfloat16);  primals_8 = None
        convolution_1: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_5, convert_element_type_6, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_6: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_9, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_7: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_7, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_7 = None
        getitem_2: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_1[0]
        getitem_3: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        add_7: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_2, 0.001)
        rsqrt_1: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        sub_1: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
        mul_7: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        squeeze_3: "f32[32][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3])
        mul_8: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1);  squeeze_3 = None
        mul_9: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_10, 0.9)
        add_8: "f32[32][1]cuda:0" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        squeeze_5: "f32[32][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_10: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000006228081046);  squeeze_5 = None
        mul_11: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
        mul_12: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_11, 0.9)
        add_9: "f32[32][1]cuda:0" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        unsqueeze_4: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_12, -1)
        unsqueeze_5: "f32[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_13: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
        unsqueeze_6: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_13, -1)
        unsqueeze_7: "f32[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_10: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
        convert_element_type_8: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(add_10, torch.bfloat16);  add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_9: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_8, torch.float32);  convert_element_type_8 = None
        neg_1: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.neg.default(convert_element_type_9)
        exp_1: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.exp.default(neg_1);  neg_1 = None
        add_11: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.add.Tensor(exp_1, 1);  exp_1 = None
        div_1: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_9, add_11);  convert_element_type_9 = add_11 = None
        convert_element_type_10: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean: "bf16[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_10, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convert_element_type_11: "bf16[8][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16);  primals_15 = None
        convert_element_type_12: "bf16[8, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
        convolution_2: "bf16[128, 8, 1, 1][8, 1, 8, 8]cuda:0" = torch.ops.aten.convolution.default(mean, convert_element_type_12, convert_element_type_11, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_13: "f32[128, 8, 1, 1][8, 1, 8, 8]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_2, torch.float32)
        neg_2: "f32[128, 8, 1, 1][8, 1, 8, 8]cuda:0" = torch.ops.aten.neg.default(convert_element_type_13)
        exp_2: "f32[128, 8, 1, 1][8, 1, 8, 8]cuda:0" = torch.ops.aten.exp.default(neg_2);  neg_2 = None
        add_12: "f32[128, 8, 1, 1][8, 1, 8, 8]cuda:0" = torch.ops.aten.add.Tensor(exp_2, 1);  exp_2 = None
        div_2: "f32[128, 8, 1, 1][8, 1, 8, 8]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_13, add_12);  convert_element_type_13 = add_12 = None
        convert_element_type_14: "bf16[128, 8, 1, 1][8, 1, 8, 8]cuda:0" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convert_element_type_15: "bf16[32][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_17, torch.bfloat16);  primals_17 = None
        convert_element_type_16: "bf16[32, 8, 1, 1][8, 1, 8, 8]cuda:0" = torch.ops.prims.convert_element_type.default(primals_16, torch.bfloat16);  primals_16 = None
        convolution_3: "bf16[128, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_14, convert_element_type_16, convert_element_type_15, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid: "bf16[128, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.aten.sigmoid.default(convolution_3)
        mul_14: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_10, sigmoid);  convert_element_type_10 = sigmoid = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:227 in forward, code: x = self.conv_pw(x)
        convert_element_type_17: "bf16[16, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.prims.convert_element_type.default(primals_18, torch.bfloat16);  primals_18 = None
        convolution_4: "bf16[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.convolution.default(mul_14, convert_element_type_17, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_13: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_19, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_18: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_4, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_18, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_18 = None
        getitem_4: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = var_mean_2[0]
        getitem_5: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        add_14: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 0.001)
        rsqrt_2: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        sub_2: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.sub.Tensor(convolution_4, getitem_5)
        mul_15: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        squeeze_6: "f32[16][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        squeeze_7: "f32[16][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_16: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
        mul_17: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_20, 0.9)
        add_15: "f32[16][1]cuda:0" = torch.ops.aten.add.Tensor(mul_16, mul_17);  mul_16 = mul_17 = None
        squeeze_8: "f32[16][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_18: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0000006228081046);  squeeze_8 = None
        mul_19: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_18, 0.1);  mul_18 = None
        mul_20: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_21, 0.9)
        add_16: "f32[16][1]cuda:0" = torch.ops.aten.add.Tensor(mul_19, mul_20);  mul_19 = mul_20 = None
        unsqueeze_8: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_22, -1)
        unsqueeze_9: "f32[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        mul_21: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(mul_15, unsqueeze_9);  mul_15 = unsqueeze_9 = None
        unsqueeze_10: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_23, -1);  primals_23 = None
        unsqueeze_11: "f32[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        add_17: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.add.Tensor(mul_21, unsqueeze_11);  mul_21 = unsqueeze_11 = None
        convert_element_type_19: "bf16[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.prims.convert_element_type.default(add_17, torch.bfloat16);  add_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convert_element_type_20: "bf16[96, 16, 1, 1][16, 1, 16, 16]cuda:0" = torch.ops.prims.convert_element_type.default(primals_24, torch.bfloat16);  primals_24 = None
        convolution_5: "bf16[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_19, convert_element_type_20, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_18: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_25, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_21: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_5, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_21, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_21 = None
        getitem_6: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = var_mean_3[0]
        getitem_7: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        add_19: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_6, 0.001)
        rsqrt_3: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        sub_3: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_5, getitem_7)
        mul_22: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        squeeze_9: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3])
        mul_23: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1);  squeeze_9 = None
        mul_24: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_26, 0.9)
        add_20: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(mul_23, mul_24);  mul_23 = mul_24 = None
        squeeze_11: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_25: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0000006228081046);  squeeze_11 = None
        mul_26: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_25, 0.1);  mul_25 = None
        mul_27: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_27, 0.9)
        add_21: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(mul_26, mul_27);  mul_26 = mul_27 = None
        unsqueeze_12: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_28, -1)
        unsqueeze_13: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_28: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_13);  mul_22 = unsqueeze_13 = None
        unsqueeze_14: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_29, -1)
        unsqueeze_15: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_22: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_28, unsqueeze_15);  mul_28 = unsqueeze_15 = None
        convert_element_type_22: "bf16[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_22, torch.bfloat16);  add_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_23: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_22, torch.float32);  convert_element_type_22 = None
        neg_3: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.neg.default(convert_element_type_23)
        exp_3: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.exp.default(neg_3);  neg_3 = None
        add_23: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.add.Tensor(exp_3, 1);  exp_3 = None
        div_3: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_23, add_23);  convert_element_type_23 = add_23 = None
        convert_element_type_24: "bf16[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_1: "bf16[128, 96, 113, 113][1225824, 1, 10848, 96]cuda:0" = torch.ops.aten.constant_pad_nd.default(convert_element_type_24, [0, 1, 0, 1], 0.0);  convert_element_type_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv2d_same.py:28 in conv2d_same, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
        convert_element_type_25: "bf16[96, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_30, torch.bfloat16);  primals_30 = None
        convolution_6: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.convolution.default(constant_pad_nd_1, convert_element_type_25, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 96)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_24: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_31, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_26: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_6, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_26, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_26 = None
        getitem_8: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = var_mean_4[0]
        getitem_9: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        add_25: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 0.001)
        rsqrt_4: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        sub_4: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_6, getitem_9)
        mul_29: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        squeeze_12: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3])
        mul_30: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1);  squeeze_12 = None
        mul_31: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_32, 0.9)
        add_26: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(mul_30, mul_31);  mul_30 = mul_31 = None
        squeeze_14: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_32: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000024912370735);  squeeze_14 = None
        mul_33: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_32, 0.1);  mul_32 = None
        mul_34: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_33, 0.9)
        add_27: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(mul_33, mul_34);  mul_33 = mul_34 = None
        unsqueeze_16: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_34, -1)
        unsqueeze_17: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        mul_35: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_29, unsqueeze_17);  mul_29 = unsqueeze_17 = None
        unsqueeze_18: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_35, -1)
        unsqueeze_19: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        add_28: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_35, unsqueeze_19);  mul_35 = unsqueeze_19 = None
        convert_element_type_27: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_28, torch.bfloat16);  add_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_28: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_27, torch.float32);  convert_element_type_27 = None
        neg_4: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.neg.default(convert_element_type_28)
        exp_4: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.exp.default(neg_4);  neg_4 = None
        add_29: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.add.Tensor(exp_4, 1);  exp_4 = None
        div_4: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_28, add_29);  convert_element_type_28 = add_29 = None
        convert_element_type_29: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_1: "bf16[128, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_29, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convert_element_type_30: "bf16[4][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_37, torch.bfloat16);  primals_37 = None
        convert_element_type_31: "bf16[4, 96, 1, 1][96, 1, 96, 96]cuda:0" = torch.ops.prims.convert_element_type.default(primals_36, torch.bfloat16);  primals_36 = None
        convolution_7: "bf16[128, 4, 1, 1][4, 1, 4, 4]cuda:0" = torch.ops.aten.convolution.default(mean_1, convert_element_type_31, convert_element_type_30, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_32: "f32[128, 4, 1, 1][4, 1, 4, 4]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_7, torch.float32)
        neg_5: "f32[128, 4, 1, 1][4, 1, 4, 4]cuda:0" = torch.ops.aten.neg.default(convert_element_type_32)
        exp_5: "f32[128, 4, 1, 1][4, 1, 4, 4]cuda:0" = torch.ops.aten.exp.default(neg_5);  neg_5 = None
        add_30: "f32[128, 4, 1, 1][4, 1, 4, 4]cuda:0" = torch.ops.aten.add.Tensor(exp_5, 1);  exp_5 = None
        div_5: "f32[128, 4, 1, 1][4, 1, 4, 4]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_32, add_30);  convert_element_type_32 = add_30 = None
        convert_element_type_33: "bf16[128, 4, 1, 1][4, 1, 4, 4]cuda:0" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convert_element_type_34: "bf16[96][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_39, torch.bfloat16);  primals_39 = None
        convert_element_type_35: "bf16[96, 4, 1, 1][4, 1, 4, 4]cuda:0" = torch.ops.prims.convert_element_type.default(primals_38, torch.bfloat16);  primals_38 = None
        convolution_8: "bf16[128, 96, 1, 1][96, 1, 96, 96]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_33, convert_element_type_35, convert_element_type_34, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_1: "bf16[128, 96, 1, 1][96, 1, 96, 96]cuda:0" = torch.ops.aten.sigmoid.default(convolution_8)
        mul_36: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_29, sigmoid_1);  convert_element_type_29 = sigmoid_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convert_element_type_36: "bf16[24, 96, 1, 1][96, 1, 96, 96]cuda:0" = torch.ops.prims.convert_element_type.default(primals_40, torch.bfloat16);  primals_40 = None
        convolution_9: "bf16[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.convolution.default(mul_36, convert_element_type_36, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_31: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_41, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_37: "f32[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_9, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_37, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_37 = None
        getitem_10: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = var_mean_5[0]
        getitem_11: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        add_32: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_10, 0.001)
        rsqrt_5: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        sub_5: "f32[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.sub.Tensor(convolution_9, getitem_11)
        mul_37: "f32[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        squeeze_15: "f32[24][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        squeeze_16: "f32[24][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_38: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
        mul_39: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_42, 0.9)
        add_33: "f32[24][1]cuda:0" = torch.ops.aten.add.Tensor(mul_38, mul_39);  mul_38 = mul_39 = None
        squeeze_17: "f32[24][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_40: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0000024912370735);  squeeze_17 = None
        mul_41: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_40, 0.1);  mul_40 = None
        mul_42: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_43, 0.9)
        add_34: "f32[24][1]cuda:0" = torch.ops.aten.add.Tensor(mul_41, mul_42);  mul_41 = mul_42 = None
        unsqueeze_20: "f32[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_44, -1)
        unsqueeze_21: "f32[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_43: "f32[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.mul.Tensor(mul_37, unsqueeze_21);  mul_37 = unsqueeze_21 = None
        unsqueeze_22: "f32[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_45, -1);  primals_45 = None
        unsqueeze_23: "f32[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_35: "f32[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.add.Tensor(mul_43, unsqueeze_23);  mul_43 = unsqueeze_23 = None
        convert_element_type_38: "bf16[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.prims.convert_element_type.default(add_35, torch.bfloat16);  add_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convert_element_type_39: "bf16[144, 24, 1, 1][24, 1, 24, 24]cuda:0" = torch.ops.prims.convert_element_type.default(primals_46, torch.bfloat16);  primals_46 = None
        convolution_10: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_38, convert_element_type_39, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_36: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_47, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_40: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_10, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_40, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_40 = None
        getitem_12: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = var_mean_6[0]
        getitem_13: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        add_37: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_12, 0.001)
        rsqrt_6: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        sub_6: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sub.Tensor(convolution_10, getitem_13)
        mul_44: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        squeeze_18: "f32[144][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3])
        mul_45: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1);  squeeze_18 = None
        mul_46: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_48, 0.9)
        add_38: "f32[144][1]cuda:0" = torch.ops.aten.add.Tensor(mul_45, mul_46);  mul_45 = mul_46 = None
        squeeze_20: "f32[144][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
        mul_47: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0000024912370735);  squeeze_20 = None
        mul_48: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_47, 0.1);  mul_47 = None
        mul_49: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_49, 0.9)
        add_39: "f32[144][1]cuda:0" = torch.ops.aten.add.Tensor(mul_48, mul_49);  mul_48 = mul_49 = None
        unsqueeze_24: "f32[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_50, -1)
        unsqueeze_25: "f32[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        mul_50: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(mul_44, unsqueeze_25);  mul_44 = unsqueeze_25 = None
        unsqueeze_26: "f32[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_51, -1)
        unsqueeze_27: "f32[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        add_40: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.add.Tensor(mul_50, unsqueeze_27);  mul_50 = unsqueeze_27 = None
        convert_element_type_41: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(add_40, torch.bfloat16);  add_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_42: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_41, torch.float32);  convert_element_type_41 = None
        neg_6: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.neg.default(convert_element_type_42)
        exp_6: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.exp.default(neg_6);  neg_6 = None
        add_41: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.add.Tensor(exp_6, 1);  exp_6 = None
        div_6: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_42, add_41);  convert_element_type_42 = add_41 = None
        convert_element_type_43: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convert_element_type_44: "bf16[144, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_52, torch.bfloat16);  primals_52 = None
        convolution_11: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_43, convert_element_type_44, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 144)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_42: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_53, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_45: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_11, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_45, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_45 = None
        getitem_14: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = var_mean_7[0]
        getitem_15: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        add_43: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 0.001)
        rsqrt_7: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        sub_7: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sub.Tensor(convolution_11, getitem_15)
        mul_51: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        squeeze_21: "f32[144][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3])
        mul_52: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1);  squeeze_21 = None
        mul_53: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_54, 0.9)
        add_44: "f32[144][1]cuda:0" = torch.ops.aten.add.Tensor(mul_52, mul_53);  mul_52 = mul_53 = None
        squeeze_23: "f32[144][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
        mul_54: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0000024912370735);  squeeze_23 = None
        mul_55: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_54, 0.1);  mul_54 = None
        mul_56: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_55, 0.9)
        add_45: "f32[144][1]cuda:0" = torch.ops.aten.add.Tensor(mul_55, mul_56);  mul_55 = mul_56 = None
        unsqueeze_28: "f32[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_56, -1)
        unsqueeze_29: "f32[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_57: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(mul_51, unsqueeze_29);  mul_51 = unsqueeze_29 = None
        unsqueeze_30: "f32[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_57, -1)
        unsqueeze_31: "f32[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_46: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.add.Tensor(mul_57, unsqueeze_31);  mul_57 = unsqueeze_31 = None
        convert_element_type_46: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(add_46, torch.bfloat16);  add_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_47: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_46, torch.float32);  convert_element_type_46 = None
        neg_7: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.neg.default(convert_element_type_47)
        exp_7: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.exp.default(neg_7);  neg_7 = None
        add_47: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.add.Tensor(exp_7, 1);  exp_7 = None
        div_7: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_47, add_47);  convert_element_type_47 = add_47 = None
        convert_element_type_48: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_2: "bf16[128, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_48, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convert_element_type_49: "bf16[6][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_59, torch.bfloat16);  primals_59 = None
        convert_element_type_50: "bf16[6, 144, 1, 1][144, 1, 144, 144]cuda:0" = torch.ops.prims.convert_element_type.default(primals_58, torch.bfloat16);  primals_58 = None
        convolution_12: "bf16[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.aten.convolution.default(mean_2, convert_element_type_50, convert_element_type_49, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_51: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_12, torch.float32)
        neg_8: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.aten.neg.default(convert_element_type_51)
        exp_8: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.aten.exp.default(neg_8);  neg_8 = None
        add_48: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.aten.add.Tensor(exp_8, 1);  exp_8 = None
        div_8: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_51, add_48);  convert_element_type_51 = add_48 = None
        convert_element_type_52: "bf16[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convert_element_type_53: "bf16[144][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_61, torch.bfloat16);  primals_61 = None
        convert_element_type_54: "bf16[144, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.prims.convert_element_type.default(primals_60, torch.bfloat16);  primals_60 = None
        convolution_13: "bf16[128, 144, 1, 1][144, 1, 144, 144]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_52, convert_element_type_54, convert_element_type_53, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_2: "bf16[128, 144, 1, 1][144, 1, 144, 144]cuda:0" = torch.ops.aten.sigmoid.default(convolution_13)
        mul_58: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_48, sigmoid_2);  convert_element_type_48 = sigmoid_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convert_element_type_55: "bf16[24, 144, 1, 1][144, 1, 144, 144]cuda:0" = torch.ops.prims.convert_element_type.default(primals_62, torch.bfloat16);  primals_62 = None
        convolution_14: "bf16[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.convolution.default(mul_58, convert_element_type_55, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_49: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_63, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_56: "f32[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_14, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_56, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_56 = None
        getitem_16: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = var_mean_8[0]
        getitem_17: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        add_50: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 0.001)
        rsqrt_8: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_8: "f32[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.sub.Tensor(convolution_14, getitem_17)
        mul_59: "f32[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        squeeze_24: "f32[24][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
        squeeze_25: "f32[24][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_60: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
        mul_61: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_64, 0.9)
        add_51: "f32[24][1]cuda:0" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
        squeeze_26: "f32[24][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
        mul_62: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0000024912370735);  squeeze_26 = None
        mul_63: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_62, 0.1);  mul_62 = None
        mul_64: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_65, 0.9)
        add_52: "f32[24][1]cuda:0" = torch.ops.aten.add.Tensor(mul_63, mul_64);  mul_63 = mul_64 = None
        unsqueeze_32: "f32[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_66, -1)
        unsqueeze_33: "f32[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        mul_65: "f32[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.mul.Tensor(mul_59, unsqueeze_33);  mul_59 = unsqueeze_33 = None
        unsqueeze_34: "f32[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_67, -1);  primals_67 = None
        unsqueeze_35: "f32[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        add_53: "f32[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.add.Tensor(mul_65, unsqueeze_35);  mul_65 = unsqueeze_35 = None
        convert_element_type_57: "bf16[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.prims.convert_element_type.default(add_53, torch.bfloat16);  add_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:338 in forward, code: x = self.drop_path(x) + shortcut
        add_54: "bf16[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_57, convert_element_type_38);  convert_element_type_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convert_element_type_58: "bf16[144, 24, 1, 1][24, 1, 24, 24]cuda:0" = torch.ops.prims.convert_element_type.default(primals_68, torch.bfloat16);  primals_68 = None
        convolution_15: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.convolution.default(add_54, convert_element_type_58, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_55: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_69, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_59: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_15, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_59, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_59 = None
        getitem_18: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = var_mean_9[0]
        getitem_19: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        add_56: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 0.001)
        rsqrt_9: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        sub_9: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sub.Tensor(convolution_15, getitem_19)
        mul_66: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        squeeze_27: "f32[144][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3])
        mul_67: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1);  squeeze_27 = None
        mul_68: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_70, 0.9)
        add_57: "f32[144][1]cuda:0" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
        squeeze_29: "f32[144][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
        mul_69: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0000024912370735);  squeeze_29 = None
        mul_70: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_69, 0.1);  mul_69 = None
        mul_71: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_71, 0.9)
        add_58: "f32[144][1]cuda:0" = torch.ops.aten.add.Tensor(mul_70, mul_71);  mul_70 = mul_71 = None
        unsqueeze_36: "f32[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_72, -1)
        unsqueeze_37: "f32[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_72: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, unsqueeze_37);  mul_66 = unsqueeze_37 = None
        unsqueeze_38: "f32[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_73, -1)
        unsqueeze_39: "f32[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_59: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.add.Tensor(mul_72, unsqueeze_39);  mul_72 = unsqueeze_39 = None
        convert_element_type_60: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(add_59, torch.bfloat16);  add_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_61: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_60, torch.float32);  convert_element_type_60 = None
        neg_9: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.neg.default(convert_element_type_61)
        exp_9: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.exp.default(neg_9);  neg_9 = None
        add_60: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.add.Tensor(exp_9, 1);  exp_9 = None
        div_9: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_61, add_60);  convert_element_type_61 = add_60 = None
        convert_element_type_62: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_2: "bf16[128, 144, 59, 59][501264, 1, 8496, 144]cuda:0" = torch.ops.aten.constant_pad_nd.default(convert_element_type_62, [1, 2, 1, 2], 0.0);  convert_element_type_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv2d_same.py:28 in conv2d_same, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
        convert_element_type_63: "bf16[144, 1, 5, 5][25, 1, 5, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_74, torch.bfloat16);  primals_74 = None
        convolution_16: "bf16[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.convolution.default(constant_pad_nd_2, convert_element_type_63, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 144)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_61: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_75, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_64: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_16, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_64, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_64 = None
        getitem_20: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = var_mean_10[0]
        getitem_21: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        add_62: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 0.001)
        rsqrt_10: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        sub_10: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.sub.Tensor(convolution_16, getitem_21)
        mul_73: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        squeeze_30: "f32[144][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3])
        mul_74: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1);  squeeze_30 = None
        mul_75: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_76, 0.9)
        add_63: "f32[144][1]cuda:0" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
        squeeze_32: "f32[144][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        mul_76: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_32, 1.00000996502277);  squeeze_32 = None
        mul_77: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_76, 0.1);  mul_76 = None
        mul_78: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_77, 0.9)
        add_64: "f32[144][1]cuda:0" = torch.ops.aten.add.Tensor(mul_77, mul_78);  mul_77 = mul_78 = None
        unsqueeze_40: "f32[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_78, -1)
        unsqueeze_41: "f32[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        mul_79: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.mul.Tensor(mul_73, unsqueeze_41);  mul_73 = unsqueeze_41 = None
        unsqueeze_42: "f32[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_79, -1)
        unsqueeze_43: "f32[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        add_65: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.add.Tensor(mul_79, unsqueeze_43);  mul_79 = unsqueeze_43 = None
        convert_element_type_65: "bf16[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.prims.convert_element_type.default(add_65, torch.bfloat16);  add_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_66: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_65, torch.float32);  convert_element_type_65 = None
        neg_10: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.neg.default(convert_element_type_66)
        exp_10: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.exp.default(neg_10);  neg_10 = None
        add_66: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.add.Tensor(exp_10, 1);  exp_10 = None
        div_10: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_66, add_66);  convert_element_type_66 = add_66 = None
        convert_element_type_67: "bf16[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_3: "bf16[128, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_67, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convert_element_type_68: "bf16[6][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_81, torch.bfloat16);  primals_81 = None
        convert_element_type_69: "bf16[6, 144, 1, 1][144, 1, 144, 144]cuda:0" = torch.ops.prims.convert_element_type.default(primals_80, torch.bfloat16);  primals_80 = None
        convolution_17: "bf16[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.aten.convolution.default(mean_3, convert_element_type_69, convert_element_type_68, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_70: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_17, torch.float32)
        neg_11: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.aten.neg.default(convert_element_type_70)
        exp_11: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.aten.exp.default(neg_11);  neg_11 = None
        add_67: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.aten.add.Tensor(exp_11, 1);  exp_11 = None
        div_11: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_70, add_67);  convert_element_type_70 = add_67 = None
        convert_element_type_71: "bf16[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convert_element_type_72: "bf16[144][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_83, torch.bfloat16);  primals_83 = None
        convert_element_type_73: "bf16[144, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.prims.convert_element_type.default(primals_82, torch.bfloat16);  primals_82 = None
        convolution_18: "bf16[128, 144, 1, 1][144, 1, 144, 144]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_71, convert_element_type_73, convert_element_type_72, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_3: "bf16[128, 144, 1, 1][144, 1, 144, 144]cuda:0" = torch.ops.aten.sigmoid.default(convolution_18)
        mul_80: "bf16[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_67, sigmoid_3);  convert_element_type_67 = sigmoid_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convert_element_type_74: "bf16[40, 144, 1, 1][144, 1, 144, 144]cuda:0" = torch.ops.prims.convert_element_type.default(primals_84, torch.bfloat16);  primals_84 = None
        convolution_19: "bf16[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.convolution.default(mul_80, convert_element_type_74, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_68: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_85, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_75: "f32[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_19, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_75, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_75 = None
        getitem_22: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = var_mean_11[0]
        getitem_23: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        add_69: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_22, 0.001)
        rsqrt_11: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_11: "f32[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.sub.Tensor(convolution_19, getitem_23)
        mul_81: "f32[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        squeeze_33: "f32[40][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
        squeeze_34: "f32[40][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
        mul_82: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
        mul_83: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_86, 0.9)
        add_70: "f32[40][1]cuda:0" = torch.ops.aten.add.Tensor(mul_82, mul_83);  mul_82 = mul_83 = None
        squeeze_35: "f32[40][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
        mul_84: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_35, 1.00000996502277);  squeeze_35 = None
        mul_85: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_84, 0.1);  mul_84 = None
        mul_86: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_87, 0.9)
        add_71: "f32[40][1]cuda:0" = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
        unsqueeze_44: "f32[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_88, -1)
        unsqueeze_45: "f32[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_87: "f32[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.mul.Tensor(mul_81, unsqueeze_45);  mul_81 = unsqueeze_45 = None
        unsqueeze_46: "f32[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_89, -1);  primals_89 = None
        unsqueeze_47: "f32[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_72: "f32[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.add.Tensor(mul_87, unsqueeze_47);  mul_87 = unsqueeze_47 = None
        convert_element_type_76: "bf16[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.prims.convert_element_type.default(add_72, torch.bfloat16);  add_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convert_element_type_77: "bf16[240, 40, 1, 1][40, 1, 40, 40]cuda:0" = torch.ops.prims.convert_element_type.default(primals_90, torch.bfloat16);  primals_90 = None
        convolution_20: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_76, convert_element_type_77, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_73: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_91, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_78: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_20, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_78, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_78 = None
        getitem_24: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = var_mean_12[0]
        getitem_25: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        add_74: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 0.001)
        rsqrt_12: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_12: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.sub.Tensor(convolution_20, getitem_25)
        mul_88: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        squeeze_36: "f32[240][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3])
        mul_89: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1);  squeeze_36 = None
        mul_90: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_92, 0.9)
        add_75: "f32[240][1]cuda:0" = torch.ops.aten.add.Tensor(mul_89, mul_90);  mul_89 = mul_90 = None
        squeeze_38: "f32[240][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
        mul_91: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_38, 1.00000996502277);  squeeze_38 = None
        mul_92: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_91, 0.1);  mul_91 = None
        mul_93: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_93, 0.9)
        add_76: "f32[240][1]cuda:0" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
        unsqueeze_48: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_94, -1)
        unsqueeze_49: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        mul_94: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(mul_88, unsqueeze_49);  mul_88 = unsqueeze_49 = None
        unsqueeze_50: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_95, -1)
        unsqueeze_51: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        add_77: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.add.Tensor(mul_94, unsqueeze_51);  mul_94 = unsqueeze_51 = None
        convert_element_type_79: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(add_77, torch.bfloat16);  add_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_80: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_79, torch.float32);  convert_element_type_79 = None
        neg_12: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.neg.default(convert_element_type_80)
        exp_12: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.exp.default(neg_12);  neg_12 = None
        add_78: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.add.Tensor(exp_12, 1);  exp_12 = None
        div_12: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_80, add_78);  convert_element_type_80 = add_78 = None
        convert_element_type_81: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(div_12, torch.bfloat16);  div_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convert_element_type_82: "bf16[240, 1, 5, 5][25, 1, 5, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_96, torch.bfloat16);  primals_96 = None
        convolution_21: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_81, convert_element_type_82, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 240)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_79: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_97, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_83: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_21, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_83, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_83 = None
        getitem_26: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = var_mean_13[0]
        getitem_27: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        add_80: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 0.001)
        rsqrt_13: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        sub_13: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.sub.Tensor(convolution_21, getitem_27)
        mul_95: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        squeeze_39: "f32[240][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3])
        mul_96: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1);  squeeze_39 = None
        mul_97: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_98, 0.9)
        add_81: "f32[240][1]cuda:0" = torch.ops.aten.add.Tensor(mul_96, mul_97);  mul_96 = mul_97 = None
        squeeze_41: "f32[240][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        mul_98: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_41, 1.00000996502277);  squeeze_41 = None
        mul_99: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_98, 0.1);  mul_98 = None
        mul_100: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_99, 0.9)
        add_82: "f32[240][1]cuda:0" = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
        unsqueeze_52: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_100, -1)
        unsqueeze_53: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_101: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(mul_95, unsqueeze_53);  mul_95 = unsqueeze_53 = None
        unsqueeze_54: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_101, -1)
        unsqueeze_55: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_83: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.add.Tensor(mul_101, unsqueeze_55);  mul_101 = unsqueeze_55 = None
        convert_element_type_84: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(add_83, torch.bfloat16);  add_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_85: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_84, torch.float32);  convert_element_type_84 = None
        neg_13: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.neg.default(convert_element_type_85)
        exp_13: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.exp.default(neg_13);  neg_13 = None
        add_84: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.add.Tensor(exp_13, 1);  exp_13 = None
        div_13: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_85, add_84);  convert_element_type_85 = add_84 = None
        convert_element_type_86: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_4: "bf16[128, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_86, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convert_element_type_87: "bf16[10][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_103, torch.bfloat16);  primals_103 = None
        convert_element_type_88: "bf16[10, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.prims.convert_element_type.default(primals_102, torch.bfloat16);  primals_102 = None
        convolution_22: "bf16[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.aten.convolution.default(mean_4, convert_element_type_88, convert_element_type_87, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_89: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_22, torch.float32)
        neg_14: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.aten.neg.default(convert_element_type_89)
        exp_14: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.aten.exp.default(neg_14);  neg_14 = None
        add_85: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.aten.add.Tensor(exp_14, 1);  exp_14 = None
        div_14: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_89, add_85);  convert_element_type_89 = add_85 = None
        convert_element_type_90: "bf16[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.prims.convert_element_type.default(div_14, torch.bfloat16);  div_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convert_element_type_91: "bf16[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_105, torch.bfloat16);  primals_105 = None
        convert_element_type_92: "bf16[240, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.prims.convert_element_type.default(primals_104, torch.bfloat16);  primals_104 = None
        convolution_23: "bf16[128, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_90, convert_element_type_92, convert_element_type_91, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_4: "bf16[128, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.aten.sigmoid.default(convolution_23)
        mul_102: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_86, sigmoid_4);  convert_element_type_86 = sigmoid_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convert_element_type_93: "bf16[40, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.prims.convert_element_type.default(primals_106, torch.bfloat16);  primals_106 = None
        convolution_24: "bf16[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.convolution.default(mul_102, convert_element_type_93, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_86: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_107, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_94: "f32[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_24, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_94, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_94 = None
        getitem_28: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = var_mean_14[0]
        getitem_29: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        add_87: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_28, 0.001)
        rsqrt_14: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
        sub_14: "f32[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.sub.Tensor(convolution_24, getitem_29)
        mul_103: "f32[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        squeeze_42: "f32[40][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        squeeze_43: "f32[40][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_104: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
        mul_105: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_108, 0.9)
        add_88: "f32[40][1]cuda:0" = torch.ops.aten.add.Tensor(mul_104, mul_105);  mul_104 = mul_105 = None
        squeeze_44: "f32[40][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        mul_106: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_44, 1.00000996502277);  squeeze_44 = None
        mul_107: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_106, 0.1);  mul_106 = None
        mul_108: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_109, 0.9)
        add_89: "f32[40][1]cuda:0" = torch.ops.aten.add.Tensor(mul_107, mul_108);  mul_107 = mul_108 = None
        unsqueeze_56: "f32[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_110, -1)
        unsqueeze_57: "f32[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        mul_109: "f32[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.mul.Tensor(mul_103, unsqueeze_57);  mul_103 = unsqueeze_57 = None
        unsqueeze_58: "f32[40, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_111, -1);  primals_111 = None
        unsqueeze_59: "f32[40, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        add_90: "f32[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.add.Tensor(mul_109, unsqueeze_59);  mul_109 = unsqueeze_59 = None
        convert_element_type_95: "bf16[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.prims.convert_element_type.default(add_90, torch.bfloat16);  add_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:338 in forward, code: x = self.drop_path(x) + shortcut
        add_91: "bf16[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_95, convert_element_type_76);  convert_element_type_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convert_element_type_96: "bf16[240, 40, 1, 1][40, 1, 40, 40]cuda:0" = torch.ops.prims.convert_element_type.default(primals_112, torch.bfloat16);  primals_112 = None
        convolution_25: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.convolution.default(add_91, convert_element_type_96, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_92: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_113, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_97: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_25, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_97, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_97 = None
        getitem_30: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = var_mean_15[0]
        getitem_31: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        add_93: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 0.001)
        rsqrt_15: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        sub_15: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.sub.Tensor(convolution_25, getitem_31)
        mul_110: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        squeeze_45: "f32[240][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3])
        mul_111: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1);  squeeze_45 = None
        mul_112: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_114, 0.9)
        add_94: "f32[240][1]cuda:0" = torch.ops.aten.add.Tensor(mul_111, mul_112);  mul_111 = mul_112 = None
        squeeze_47: "f32[240][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
        mul_113: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_47, 1.00000996502277);  squeeze_47 = None
        mul_114: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_113, 0.1);  mul_113 = None
        mul_115: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_115, 0.9)
        add_95: "f32[240][1]cuda:0" = torch.ops.aten.add.Tensor(mul_114, mul_115);  mul_114 = mul_115 = None
        unsqueeze_60: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_116, -1)
        unsqueeze_61: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_116: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(mul_110, unsqueeze_61);  mul_110 = unsqueeze_61 = None
        unsqueeze_62: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_117, -1)
        unsqueeze_63: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_96: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.add.Tensor(mul_116, unsqueeze_63);  mul_116 = unsqueeze_63 = None
        convert_element_type_98: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(add_96, torch.bfloat16);  add_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_99: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_98, torch.float32);  convert_element_type_98 = None
        neg_15: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.neg.default(convert_element_type_99)
        exp_15: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.exp.default(neg_15);  neg_15 = None
        add_97: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.add.Tensor(exp_15, 1);  exp_15 = None
        div_15: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_99, add_97);  convert_element_type_99 = add_97 = None
        convert_element_type_100: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_3: "bf16[128, 240, 29, 29][201840, 1, 6960, 240]cuda:0" = torch.ops.aten.constant_pad_nd.default(convert_element_type_100, [0, 1, 0, 1], 0.0);  convert_element_type_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv2d_same.py:28 in conv2d_same, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
        convert_element_type_101: "bf16[240, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_118, torch.bfloat16);  primals_118 = None
        convolution_26: "bf16[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.convolution.default(constant_pad_nd_3, convert_element_type_101, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 240)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_98: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_119, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_102: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_26, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_102, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_102 = None
        getitem_32: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = var_mean_16[0]
        getitem_33: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        add_99: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 0.001)
        rsqrt_16: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
        sub_16: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.sub.Tensor(convolution_26, getitem_33)
        mul_117: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        squeeze_48: "f32[240][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3])
        mul_118: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1);  squeeze_48 = None
        mul_119: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_120, 0.9)
        add_100: "f32[240][1]cuda:0" = torch.ops.aten.add.Tensor(mul_118, mul_119);  mul_118 = mul_119 = None
        squeeze_50: "f32[240][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
        mul_120: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0000398612827361);  squeeze_50 = None
        mul_121: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_120, 0.1);  mul_120 = None
        mul_122: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_121, 0.9)
        add_101: "f32[240][1]cuda:0" = torch.ops.aten.add.Tensor(mul_121, mul_122);  mul_121 = mul_122 = None
        unsqueeze_64: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_122, -1)
        unsqueeze_65: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        mul_123: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(mul_117, unsqueeze_65);  mul_117 = unsqueeze_65 = None
        unsqueeze_66: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_123, -1)
        unsqueeze_67: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        add_102: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.add.Tensor(mul_123, unsqueeze_67);  mul_123 = unsqueeze_67 = None
        convert_element_type_103: "bf16[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.prims.convert_element_type.default(add_102, torch.bfloat16);  add_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_104: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_103, torch.float32);  convert_element_type_103 = None
        neg_16: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.neg.default(convert_element_type_104)
        exp_16: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.exp.default(neg_16);  neg_16 = None
        add_103: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.add.Tensor(exp_16, 1);  exp_16 = None
        div_16: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_104, add_103);  convert_element_type_104 = add_103 = None
        convert_element_type_105: "bf16[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.prims.convert_element_type.default(div_16, torch.bfloat16);  div_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_5: "bf16[128, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_105, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convert_element_type_106: "bf16[10][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_125, torch.bfloat16);  primals_125 = None
        convert_element_type_107: "bf16[10, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.prims.convert_element_type.default(primals_124, torch.bfloat16);  primals_124 = None
        convolution_27: "bf16[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.aten.convolution.default(mean_5, convert_element_type_107, convert_element_type_106, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_108: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_27, torch.float32)
        neg_17: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.aten.neg.default(convert_element_type_108)
        exp_17: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.aten.exp.default(neg_17);  neg_17 = None
        add_104: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.aten.add.Tensor(exp_17, 1);  exp_17 = None
        div_17: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_108, add_104);  convert_element_type_108 = add_104 = None
        convert_element_type_109: "bf16[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.prims.convert_element_type.default(div_17, torch.bfloat16);  div_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convert_element_type_110: "bf16[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_127, torch.bfloat16);  primals_127 = None
        convert_element_type_111: "bf16[240, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.prims.convert_element_type.default(primals_126, torch.bfloat16);  primals_126 = None
        convolution_28: "bf16[128, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_109, convert_element_type_111, convert_element_type_110, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_5: "bf16[128, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.aten.sigmoid.default(convolution_28)
        mul_124: "bf16[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_105, sigmoid_5);  convert_element_type_105 = sigmoid_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convert_element_type_112: "bf16[80, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.prims.convert_element_type.default(primals_128, torch.bfloat16);  primals_128 = None
        convolution_29: "bf16[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.convolution.default(mul_124, convert_element_type_112, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_105: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_129, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_113: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_29, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_113, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_113 = None
        getitem_34: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = var_mean_17[0]
        getitem_35: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        add_106: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 0.001)
        rsqrt_17: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        sub_17: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.sub.Tensor(convolution_29, getitem_35)
        mul_125: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        squeeze_51: "f32[80][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
        squeeze_52: "f32[80][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
        mul_126: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
        mul_127: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_130, 0.9)
        add_107: "f32[80][1]cuda:0" = torch.ops.aten.add.Tensor(mul_126, mul_127);  mul_126 = mul_127 = None
        squeeze_53: "f32[80][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
        mul_128: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0000398612827361);  squeeze_53 = None
        mul_129: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_128, 0.1);  mul_128 = None
        mul_130: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_131, 0.9)
        add_108: "f32[80][1]cuda:0" = torch.ops.aten.add.Tensor(mul_129, mul_130);  mul_129 = mul_130 = None
        unsqueeze_68: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_132, -1)
        unsqueeze_69: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_131: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(mul_125, unsqueeze_69);  mul_125 = unsqueeze_69 = None
        unsqueeze_70: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_133, -1);  primals_133 = None
        unsqueeze_71: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_109: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.add.Tensor(mul_131, unsqueeze_71);  mul_131 = unsqueeze_71 = None
        convert_element_type_114: "bf16[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(add_109, torch.bfloat16);  add_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convert_element_type_115: "bf16[480, 80, 1, 1][80, 1, 80, 80]cuda:0" = torch.ops.prims.convert_element_type.default(primals_134, torch.bfloat16);  primals_134 = None
        convolution_30: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_114, convert_element_type_115, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_110: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_135, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_116: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_30, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_116, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_116 = None
        getitem_36: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = var_mean_18[0]
        getitem_37: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        add_111: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_36, 0.001)
        rsqrt_18: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
        sub_18: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convolution_30, getitem_37)
        mul_132: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        squeeze_54: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3])
        mul_133: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1);  squeeze_54 = None
        mul_134: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_136, 0.9)
        add_112: "f32[480][1]cuda:0" = torch.ops.aten.add.Tensor(mul_133, mul_134);  mul_133 = mul_134 = None
        squeeze_56: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
        mul_135: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0000398612827361);  squeeze_56 = None
        mul_136: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_135, 0.1);  mul_135 = None
        mul_137: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_137, 0.9)
        add_113: "f32[480][1]cuda:0" = torch.ops.aten.add.Tensor(mul_136, mul_137);  mul_136 = mul_137 = None
        unsqueeze_72: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_138, -1)
        unsqueeze_73: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        mul_138: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(mul_132, unsqueeze_73);  mul_132 = unsqueeze_73 = None
        unsqueeze_74: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_139, -1)
        unsqueeze_75: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        add_114: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_138, unsqueeze_75);  mul_138 = unsqueeze_75 = None
        convert_element_type_117: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(add_114, torch.bfloat16);  add_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_118: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_117, torch.float32);  convert_element_type_117 = None
        neg_18: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.neg.default(convert_element_type_118)
        exp_18: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.exp.default(neg_18);  neg_18 = None
        add_115: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(exp_18, 1);  exp_18 = None
        div_18: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_118, add_115);  convert_element_type_118 = add_115 = None
        convert_element_type_119: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(div_18, torch.bfloat16);  div_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convert_element_type_120: "bf16[480, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_140, torch.bfloat16);  primals_140 = None
        convolution_31: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_119, convert_element_type_120, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_116: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_141, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_121: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_31, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_121, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_121 = None
        getitem_38: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = var_mean_19[0]
        getitem_39: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        add_117: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_38, 0.001)
        rsqrt_19: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
        sub_19: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convolution_31, getitem_39)
        mul_139: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        squeeze_57: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3])
        mul_140: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1);  squeeze_57 = None
        mul_141: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_142, 0.9)
        add_118: "f32[480][1]cuda:0" = torch.ops.aten.add.Tensor(mul_140, mul_141);  mul_140 = mul_141 = None
        squeeze_59: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
        mul_142: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0000398612827361);  squeeze_59 = None
        mul_143: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_142, 0.1);  mul_142 = None
        mul_144: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_143, 0.9)
        add_119: "f32[480][1]cuda:0" = torch.ops.aten.add.Tensor(mul_143, mul_144);  mul_143 = mul_144 = None
        unsqueeze_76: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_144, -1)
        unsqueeze_77: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_145: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(mul_139, unsqueeze_77);  mul_139 = unsqueeze_77 = None
        unsqueeze_78: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_145, -1)
        unsqueeze_79: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_120: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_145, unsqueeze_79);  mul_145 = unsqueeze_79 = None
        convert_element_type_122: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(add_120, torch.bfloat16);  add_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_123: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_122, torch.float32);  convert_element_type_122 = None
        neg_19: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.neg.default(convert_element_type_123)
        exp_19: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.exp.default(neg_19);  neg_19 = None
        add_121: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(exp_19, 1);  exp_19 = None
        div_19: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_123, add_121);  convert_element_type_123 = add_121 = None
        convert_element_type_124: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16);  div_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_6: "bf16[128, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_124, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convert_element_type_125: "bf16[20][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_147, torch.bfloat16);  primals_147 = None
        convert_element_type_126: "bf16[20, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.prims.convert_element_type.default(primals_146, torch.bfloat16);  primals_146 = None
        convolution_32: "bf16[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.convolution.default(mean_6, convert_element_type_126, convert_element_type_125, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_127: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_32, torch.float32)
        neg_20: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.neg.default(convert_element_type_127)
        exp_20: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.exp.default(neg_20);  neg_20 = None
        add_122: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.add.Tensor(exp_20, 1);  exp_20 = None
        div_20: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_127, add_122);  convert_element_type_127 = add_122 = None
        convert_element_type_128: "bf16[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.prims.convert_element_type.default(div_20, torch.bfloat16);  div_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convert_element_type_129: "bf16[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_149, torch.bfloat16);  primals_149 = None
        convert_element_type_130: "bf16[480, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.prims.convert_element_type.default(primals_148, torch.bfloat16);  primals_148 = None
        convolution_33: "bf16[128, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_128, convert_element_type_130, convert_element_type_129, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_6: "bf16[128, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.sigmoid.default(convolution_33)
        mul_146: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_124, sigmoid_6);  convert_element_type_124 = sigmoid_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convert_element_type_131: "bf16[80, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.prims.convert_element_type.default(primals_150, torch.bfloat16);  primals_150 = None
        convolution_34: "bf16[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.convolution.default(mul_146, convert_element_type_131, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_123: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_151, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_132: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_34, torch.float32)
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_132, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_132 = None
        getitem_40: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = var_mean_20[0]
        getitem_41: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None
        add_124: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_40, 0.001)
        rsqrt_20: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
        sub_20: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.sub.Tensor(convolution_34, getitem_41)
        mul_147: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
        squeeze_60: "f32[80][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
        squeeze_61: "f32[80][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
        mul_148: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
        mul_149: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_152, 0.9)
        add_125: "f32[80][1]cuda:0" = torch.ops.aten.add.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None
        squeeze_62: "f32[80][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
        mul_150: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0000398612827361);  squeeze_62 = None
        mul_151: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_150, 0.1);  mul_150 = None
        mul_152: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_153, 0.9)
        add_126: "f32[80][1]cuda:0" = torch.ops.aten.add.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
        unsqueeze_80: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_154, -1)
        unsqueeze_81: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        mul_153: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(mul_147, unsqueeze_81);  mul_147 = unsqueeze_81 = None
        unsqueeze_82: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_155, -1);  primals_155 = None
        unsqueeze_83: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        add_127: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.add.Tensor(mul_153, unsqueeze_83);  mul_153 = unsqueeze_83 = None
        convert_element_type_133: "bf16[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(add_127, torch.bfloat16);  add_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:338 in forward, code: x = self.drop_path(x) + shortcut
        add_128: "bf16[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_133, convert_element_type_114);  convert_element_type_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convert_element_type_134: "bf16[480, 80, 1, 1][80, 1, 80, 80]cuda:0" = torch.ops.prims.convert_element_type.default(primals_156, torch.bfloat16);  primals_156 = None
        convolution_35: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.convolution.default(add_128, convert_element_type_134, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_129: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_157, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_135: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_35, torch.float32)
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_135, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_135 = None
        getitem_42: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = var_mean_21[0]
        getitem_43: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        add_130: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_42, 0.001)
        rsqrt_21: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        sub_21: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convolution_35, getitem_43)
        mul_154: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
        squeeze_63: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3])
        mul_155: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1);  squeeze_63 = None
        mul_156: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_158, 0.9)
        add_131: "f32[480][1]cuda:0" = torch.ops.aten.add.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
        squeeze_65: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
        mul_157: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0000398612827361);  squeeze_65 = None
        mul_158: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_157, 0.1);  mul_157 = None
        mul_159: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_159, 0.9)
        add_132: "f32[480][1]cuda:0" = torch.ops.aten.add.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None
        unsqueeze_84: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_160, -1)
        unsqueeze_85: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_160: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_85);  mul_154 = unsqueeze_85 = None
        unsqueeze_86: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_161, -1)
        unsqueeze_87: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_133: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_160, unsqueeze_87);  mul_160 = unsqueeze_87 = None
        convert_element_type_136: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(add_133, torch.bfloat16);  add_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_137: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_136, torch.float32);  convert_element_type_136 = None
        neg_21: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.neg.default(convert_element_type_137)
        exp_21: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.exp.default(neg_21);  neg_21 = None
        add_134: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(exp_21, 1);  exp_21 = None
        div_21: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_137, add_134);  convert_element_type_137 = add_134 = None
        convert_element_type_138: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16);  div_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convert_element_type_139: "bf16[480, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_162, torch.bfloat16);  primals_162 = None
        convolution_36: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_138, convert_element_type_139, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_135: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_163, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_140: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_36, torch.float32)
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_140, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_140 = None
        getitem_44: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = var_mean_22[0]
        getitem_45: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None
        add_136: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_44, 0.001)
        rsqrt_22: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
        sub_22: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convolution_36, getitem_45)
        mul_161: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        squeeze_66: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3])
        mul_162: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1);  squeeze_66 = None
        mul_163: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_164, 0.9)
        add_137: "f32[480][1]cuda:0" = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
        squeeze_68: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
        mul_164: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0000398612827361);  squeeze_68 = None
        mul_165: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_164, 0.1);  mul_164 = None
        mul_166: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_165, 0.9)
        add_138: "f32[480][1]cuda:0" = torch.ops.aten.add.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
        unsqueeze_88: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_166, -1)
        unsqueeze_89: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        mul_167: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_89);  mul_161 = unsqueeze_89 = None
        unsqueeze_90: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_167, -1)
        unsqueeze_91: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        add_139: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_91);  mul_167 = unsqueeze_91 = None
        convert_element_type_141: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(add_139, torch.bfloat16);  add_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_142: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_141, torch.float32);  convert_element_type_141 = None
        neg_22: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.neg.default(convert_element_type_142)
        exp_22: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.exp.default(neg_22);  neg_22 = None
        add_140: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(exp_22, 1);  exp_22 = None
        div_22: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_142, add_140);  convert_element_type_142 = add_140 = None
        convert_element_type_143: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(div_22, torch.bfloat16);  div_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_7: "bf16[128, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_143, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convert_element_type_144: "bf16[20][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_169, torch.bfloat16);  primals_169 = None
        convert_element_type_145: "bf16[20, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.prims.convert_element_type.default(primals_168, torch.bfloat16);  primals_168 = None
        convolution_37: "bf16[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.convolution.default(mean_7, convert_element_type_145, convert_element_type_144, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_146: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_37, torch.float32)
        neg_23: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.neg.default(convert_element_type_146)
        exp_23: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.exp.default(neg_23);  neg_23 = None
        add_141: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.add.Tensor(exp_23, 1);  exp_23 = None
        div_23: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_146, add_141);  convert_element_type_146 = add_141 = None
        convert_element_type_147: "bf16[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.prims.convert_element_type.default(div_23, torch.bfloat16);  div_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convert_element_type_148: "bf16[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_171, torch.bfloat16);  primals_171 = None
        convert_element_type_149: "bf16[480, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.prims.convert_element_type.default(primals_170, torch.bfloat16);  primals_170 = None
        convolution_38: "bf16[128, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_147, convert_element_type_149, convert_element_type_148, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_7: "bf16[128, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.sigmoid.default(convolution_38)
        mul_168: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_143, sigmoid_7);  convert_element_type_143 = sigmoid_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convert_element_type_150: "bf16[80, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.prims.convert_element_type.default(primals_172, torch.bfloat16);  primals_172 = None
        convolution_39: "bf16[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.convolution.default(mul_168, convert_element_type_150, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_142: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_173, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_151: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_39, torch.float32)
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_151, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_151 = None
        getitem_46: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = var_mean_23[0]
        getitem_47: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        add_143: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_46, 0.001)
        rsqrt_23: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
        sub_23: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.sub.Tensor(convolution_39, getitem_47)
        mul_169: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
        squeeze_69: "f32[80][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
        squeeze_70: "f32[80][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
        mul_170: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
        mul_171: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_174, 0.9)
        add_144: "f32[80][1]cuda:0" = torch.ops.aten.add.Tensor(mul_170, mul_171);  mul_170 = mul_171 = None
        squeeze_71: "f32[80][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
        mul_172: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0000398612827361);  squeeze_71 = None
        mul_173: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_172, 0.1);  mul_172 = None
        mul_174: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_175, 0.9)
        add_145: "f32[80][1]cuda:0" = torch.ops.aten.add.Tensor(mul_173, mul_174);  mul_173 = mul_174 = None
        unsqueeze_92: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_176, -1)
        unsqueeze_93: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        mul_175: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(mul_169, unsqueeze_93);  mul_169 = unsqueeze_93 = None
        unsqueeze_94: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_177, -1);  primals_177 = None
        unsqueeze_95: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        add_146: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.add.Tensor(mul_175, unsqueeze_95);  mul_175 = unsqueeze_95 = None
        convert_element_type_152: "bf16[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(add_146, torch.bfloat16);  add_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:338 in forward, code: x = self.drop_path(x) + shortcut
        add_147: "bf16[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_152, add_128);  convert_element_type_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convert_element_type_153: "bf16[480, 80, 1, 1][80, 1, 80, 80]cuda:0" = torch.ops.prims.convert_element_type.default(primals_178, torch.bfloat16);  primals_178 = None
        convolution_40: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.convolution.default(add_147, convert_element_type_153, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_148: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_179, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_154: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_40, torch.float32)
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_154, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_154 = None
        getitem_48: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = var_mean_24[0]
        getitem_49: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None
        add_149: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_48, 0.001)
        rsqrt_24: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
        sub_24: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convolution_40, getitem_49)
        mul_176: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        squeeze_72: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3])
        mul_177: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1);  squeeze_72 = None
        mul_178: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_180, 0.9)
        add_150: "f32[480][1]cuda:0" = torch.ops.aten.add.Tensor(mul_177, mul_178);  mul_177 = mul_178 = None
        squeeze_74: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
        mul_179: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0000398612827361);  squeeze_74 = None
        mul_180: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_179, 0.1);  mul_179 = None
        mul_181: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_181, 0.9)
        add_151: "f32[480][1]cuda:0" = torch.ops.aten.add.Tensor(mul_180, mul_181);  mul_180 = mul_181 = None
        unsqueeze_96: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_182, -1)
        unsqueeze_97: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        mul_182: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(mul_176, unsqueeze_97);  mul_176 = unsqueeze_97 = None
        unsqueeze_98: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_183, -1)
        unsqueeze_99: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        add_152: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_182, unsqueeze_99);  mul_182 = unsqueeze_99 = None
        convert_element_type_155: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(add_152, torch.bfloat16);  add_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_156: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_155, torch.float32);  convert_element_type_155 = None
        neg_24: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.neg.default(convert_element_type_156)
        exp_24: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.exp.default(neg_24);  neg_24 = None
        add_153: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(exp_24, 1);  exp_24 = None
        div_24: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_156, add_153);  convert_element_type_156 = add_153 = None
        convert_element_type_157: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(div_24, torch.bfloat16);  div_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convert_element_type_158: "bf16[480, 1, 5, 5][25, 1, 5, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_184, torch.bfloat16);  primals_184 = None
        convolution_41: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_157, convert_element_type_158, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 480)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_154: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_185, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_159: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_41, torch.float32)
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_159, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_159 = None
        getitem_50: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = var_mean_25[0]
        getitem_51: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = var_mean_25[1];  var_mean_25 = None
        add_155: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_50, 0.001)
        rsqrt_25: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
        sub_25: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convolution_41, getitem_51)
        mul_183: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        squeeze_75: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3])
        mul_184: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1);  squeeze_75 = None
        mul_185: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_186, 0.9)
        add_156: "f32[480][1]cuda:0" = torch.ops.aten.add.Tensor(mul_184, mul_185);  mul_184 = mul_185 = None
        squeeze_77: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
        mul_186: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0000398612827361);  squeeze_77 = None
        mul_187: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_186, 0.1);  mul_186 = None
        mul_188: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_187, 0.9)
        add_157: "f32[480][1]cuda:0" = torch.ops.aten.add.Tensor(mul_187, mul_188);  mul_187 = mul_188 = None
        unsqueeze_100: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_188, -1)
        unsqueeze_101: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_189: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(mul_183, unsqueeze_101);  mul_183 = unsqueeze_101 = None
        unsqueeze_102: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_189, -1)
        unsqueeze_103: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_158: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_189, unsqueeze_103);  mul_189 = unsqueeze_103 = None
        convert_element_type_160: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(add_158, torch.bfloat16);  add_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_161: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_160, torch.float32);  convert_element_type_160 = None
        neg_25: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.neg.default(convert_element_type_161)
        exp_25: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.exp.default(neg_25);  neg_25 = None
        add_159: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(exp_25, 1);  exp_25 = None
        div_25: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_161, add_159);  convert_element_type_161 = add_159 = None
        convert_element_type_162: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(div_25, torch.bfloat16);  div_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_8: "bf16[128, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_162, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convert_element_type_163: "bf16[20][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_191, torch.bfloat16);  primals_191 = None
        convert_element_type_164: "bf16[20, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.prims.convert_element_type.default(primals_190, torch.bfloat16);  primals_190 = None
        convolution_42: "bf16[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.convolution.default(mean_8, convert_element_type_164, convert_element_type_163, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_165: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_42, torch.float32)
        neg_26: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.neg.default(convert_element_type_165)
        exp_26: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.exp.default(neg_26);  neg_26 = None
        add_160: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.add.Tensor(exp_26, 1);  exp_26 = None
        div_26: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_165, add_160);  convert_element_type_165 = add_160 = None
        convert_element_type_166: "bf16[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.prims.convert_element_type.default(div_26, torch.bfloat16);  div_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convert_element_type_167: "bf16[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_193, torch.bfloat16);  primals_193 = None
        convert_element_type_168: "bf16[480, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.prims.convert_element_type.default(primals_192, torch.bfloat16);  primals_192 = None
        convolution_43: "bf16[128, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_166, convert_element_type_168, convert_element_type_167, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_8: "bf16[128, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.sigmoid.default(convolution_43)
        mul_190: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_162, sigmoid_8);  convert_element_type_162 = sigmoid_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convert_element_type_169: "bf16[112, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.prims.convert_element_type.default(primals_194, torch.bfloat16);  primals_194 = None
        convolution_44: "bf16[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.convolution.default(mul_190, convert_element_type_169, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_161: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_195, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_170: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_44, torch.float32)
        var_mean_26 = torch.ops.aten.var_mean.correction(convert_element_type_170, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_170 = None
        getitem_52: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = var_mean_26[0]
        getitem_53: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = var_mean_26[1];  var_mean_26 = None
        add_162: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_52, 0.001)
        rsqrt_26: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
        sub_26: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.sub.Tensor(convolution_44, getitem_53)
        mul_191: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
        squeeze_78: "f32[112][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
        squeeze_79: "f32[112][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
        mul_192: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
        mul_193: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_196, 0.9)
        add_163: "f32[112][1]cuda:0" = torch.ops.aten.add.Tensor(mul_192, mul_193);  mul_192 = mul_193 = None
        squeeze_80: "f32[112][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
        mul_194: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0000398612827361);  squeeze_80 = None
        mul_195: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_194, 0.1);  mul_194 = None
        mul_196: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_197, 0.9)
        add_164: "f32[112][1]cuda:0" = torch.ops.aten.add.Tensor(mul_195, mul_196);  mul_195 = mul_196 = None
        unsqueeze_104: "f32[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_198, -1)
        unsqueeze_105: "f32[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        mul_197: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.mul.Tensor(mul_191, unsqueeze_105);  mul_191 = unsqueeze_105 = None
        unsqueeze_106: "f32[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_199, -1);  primals_199 = None
        unsqueeze_107: "f32[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
        add_165: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.add.Tensor(mul_197, unsqueeze_107);  mul_197 = unsqueeze_107 = None
        convert_element_type_171: "bf16[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.prims.convert_element_type.default(add_165, torch.bfloat16);  add_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convert_element_type_172: "bf16[672, 112, 1, 1][112, 1, 112, 112]cuda:0" = torch.ops.prims.convert_element_type.default(primals_200, torch.bfloat16);  primals_200 = None
        convolution_45: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_171, convert_element_type_172, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_166: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_201, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_173: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_45, torch.float32)
        var_mean_27 = torch.ops.aten.var_mean.correction(convert_element_type_173, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_173 = None
        getitem_54: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = var_mean_27[0]
        getitem_55: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = var_mean_27[1];  var_mean_27 = None
        add_167: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_54, 0.001)
        rsqrt_27: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_167);  add_167 = None
        sub_27: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convolution_45, getitem_55)
        mul_198: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
        squeeze_81: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3])
        mul_199: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1);  squeeze_81 = None
        mul_200: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_202, 0.9)
        add_168: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(mul_199, mul_200);  mul_199 = mul_200 = None
        squeeze_83: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
        mul_201: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0000398612827361);  squeeze_83 = None
        mul_202: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_201, 0.1);  mul_201 = None
        mul_203: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_203, 0.9)
        add_169: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(mul_202, mul_203);  mul_202 = mul_203 = None
        unsqueeze_108: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_204, -1)
        unsqueeze_109: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
        mul_204: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(mul_198, unsqueeze_109);  mul_198 = unsqueeze_109 = None
        unsqueeze_110: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_205, -1)
        unsqueeze_111: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
        add_170: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_204, unsqueeze_111);  mul_204 = unsqueeze_111 = None
        convert_element_type_174: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(add_170, torch.bfloat16);  add_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_175: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_174, torch.float32);  convert_element_type_174 = None
        neg_27: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.neg.default(convert_element_type_175)
        exp_27: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.exp.default(neg_27);  neg_27 = None
        add_171: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(exp_27, 1);  exp_27 = None
        div_27: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_175, add_171);  convert_element_type_175 = add_171 = None
        convert_element_type_176: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(div_27, torch.bfloat16);  div_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convert_element_type_177: "bf16[672, 1, 5, 5][25, 1, 5, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_206, torch.bfloat16);  primals_206 = None
        convolution_46: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_176, convert_element_type_177, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 672)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_172: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_207, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_178: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_46, torch.float32)
        var_mean_28 = torch.ops.aten.var_mean.correction(convert_element_type_178, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_178 = None
        getitem_56: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = var_mean_28[0]
        getitem_57: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = var_mean_28[1];  var_mean_28 = None
        add_173: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_56, 0.001)
        rsqrt_28: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_173);  add_173 = None
        sub_28: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convolution_46, getitem_57)
        mul_205: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
        squeeze_84: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3])
        mul_206: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1);  squeeze_84 = None
        mul_207: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_208, 0.9)
        add_174: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(mul_206, mul_207);  mul_206 = mul_207 = None
        squeeze_86: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
        mul_208: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0000398612827361);  squeeze_86 = None
        mul_209: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_208, 0.1);  mul_208 = None
        mul_210: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_209, 0.9)
        add_175: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(mul_209, mul_210);  mul_209 = mul_210 = None
        unsqueeze_112: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_210, -1)
        unsqueeze_113: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        mul_211: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(mul_205, unsqueeze_113);  mul_205 = unsqueeze_113 = None
        unsqueeze_114: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_211, -1)
        unsqueeze_115: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        add_176: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_211, unsqueeze_115);  mul_211 = unsqueeze_115 = None
        convert_element_type_179: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(add_176, torch.bfloat16);  add_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_180: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_179, torch.float32);  convert_element_type_179 = None
        neg_28: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.neg.default(convert_element_type_180)
        exp_28: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.exp.default(neg_28);  neg_28 = None
        add_177: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(exp_28, 1);  exp_28 = None
        div_28: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_180, add_177);  convert_element_type_180 = add_177 = None
        convert_element_type_181: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(div_28, torch.bfloat16);  div_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_9: "bf16[128, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_181, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convert_element_type_182: "bf16[28][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_213, torch.bfloat16);  primals_213 = None
        convert_element_type_183: "bf16[28, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(primals_212, torch.bfloat16);  primals_212 = None
        convolution_47: "bf16[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.convolution.default(mean_9, convert_element_type_183, convert_element_type_182, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_184: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_47, torch.float32)
        neg_29: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.neg.default(convert_element_type_184)
        exp_29: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.exp.default(neg_29);  neg_29 = None
        add_178: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.add.Tensor(exp_29, 1);  exp_29 = None
        div_29: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_184, add_178);  convert_element_type_184 = add_178 = None
        convert_element_type_185: "bf16[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.prims.convert_element_type.default(div_29, torch.bfloat16);  div_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convert_element_type_186: "bf16[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_215, torch.bfloat16);  primals_215 = None
        convert_element_type_187: "bf16[672, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.prims.convert_element_type.default(primals_214, torch.bfloat16);  primals_214 = None
        convolution_48: "bf16[128, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_185, convert_element_type_187, convert_element_type_186, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_9: "bf16[128, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.sigmoid.default(convolution_48)
        mul_212: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_181, sigmoid_9);  convert_element_type_181 = sigmoid_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convert_element_type_188: "bf16[112, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(primals_216, torch.bfloat16);  primals_216 = None
        convolution_49: "bf16[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.convolution.default(mul_212, convert_element_type_188, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_179: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_217, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_189: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_49, torch.float32)
        var_mean_29 = torch.ops.aten.var_mean.correction(convert_element_type_189, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_189 = None
        getitem_58: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = var_mean_29[0]
        getitem_59: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = var_mean_29[1];  var_mean_29 = None
        add_180: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_58, 0.001)
        rsqrt_29: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_180);  add_180 = None
        sub_29: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.sub.Tensor(convolution_49, getitem_59)
        mul_213: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
        squeeze_87: "f32[112][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
        squeeze_88: "f32[112][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
        mul_214: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
        mul_215: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_218, 0.9)
        add_181: "f32[112][1]cuda:0" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
        squeeze_89: "f32[112][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
        mul_216: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0000398612827361);  squeeze_89 = None
        mul_217: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_216, 0.1);  mul_216 = None
        mul_218: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_219, 0.9)
        add_182: "f32[112][1]cuda:0" = torch.ops.aten.add.Tensor(mul_217, mul_218);  mul_217 = mul_218 = None
        unsqueeze_116: "f32[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_220, -1)
        unsqueeze_117: "f32[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        mul_219: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.mul.Tensor(mul_213, unsqueeze_117);  mul_213 = unsqueeze_117 = None
        unsqueeze_118: "f32[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_221, -1);  primals_221 = None
        unsqueeze_119: "f32[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
        add_183: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.add.Tensor(mul_219, unsqueeze_119);  mul_219 = unsqueeze_119 = None
        convert_element_type_190: "bf16[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.prims.convert_element_type.default(add_183, torch.bfloat16);  add_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:338 in forward, code: x = self.drop_path(x) + shortcut
        add_184: "bf16[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_190, convert_element_type_171);  convert_element_type_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convert_element_type_191: "bf16[672, 112, 1, 1][112, 1, 112, 112]cuda:0" = torch.ops.prims.convert_element_type.default(primals_222, torch.bfloat16);  primals_222 = None
        convolution_50: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.convolution.default(add_184, convert_element_type_191, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_185: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_223, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_192: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_50, torch.float32)
        var_mean_30 = torch.ops.aten.var_mean.correction(convert_element_type_192, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_192 = None
        getitem_60: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = var_mean_30[0]
        getitem_61: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = var_mean_30[1];  var_mean_30 = None
        add_186: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_60, 0.001)
        rsqrt_30: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
        sub_30: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convolution_50, getitem_61)
        mul_220: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
        squeeze_90: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3])
        mul_221: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1);  squeeze_90 = None
        mul_222: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_224, 0.9)
        add_187: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
        squeeze_92: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
        mul_223: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0000398612827361);  squeeze_92 = None
        mul_224: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_223, 0.1);  mul_223 = None
        mul_225: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_225, 0.9)
        add_188: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(mul_224, mul_225);  mul_224 = mul_225 = None
        unsqueeze_120: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_226, -1)
        unsqueeze_121: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        mul_226: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(mul_220, unsqueeze_121);  mul_220 = unsqueeze_121 = None
        unsqueeze_122: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_227, -1)
        unsqueeze_123: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        add_189: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_226, unsqueeze_123);  mul_226 = unsqueeze_123 = None
        convert_element_type_193: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(add_189, torch.bfloat16);  add_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_194: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_193, torch.float32);  convert_element_type_193 = None
        neg_30: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.neg.default(convert_element_type_194)
        exp_30: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.exp.default(neg_30);  neg_30 = None
        add_190: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(exp_30, 1);  exp_30 = None
        div_30: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_194, add_190);  convert_element_type_194 = add_190 = None
        convert_element_type_195: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(div_30, torch.bfloat16);  div_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convert_element_type_196: "bf16[672, 1, 5, 5][25, 1, 5, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_228, torch.bfloat16);  primals_228 = None
        convolution_51: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_195, convert_element_type_196, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 672)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_191: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_229, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_197: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_51, torch.float32)
        var_mean_31 = torch.ops.aten.var_mean.correction(convert_element_type_197, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_197 = None
        getitem_62: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = var_mean_31[0]
        getitem_63: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = var_mean_31[1];  var_mean_31 = None
        add_192: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_62, 0.001)
        rsqrt_31: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
        sub_31: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convolution_51, getitem_63)
        mul_227: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
        squeeze_93: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3])
        mul_228: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1);  squeeze_93 = None
        mul_229: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_230, 0.9)
        add_193: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(mul_228, mul_229);  mul_228 = mul_229 = None
        squeeze_95: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
        mul_230: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0000398612827361);  squeeze_95 = None
        mul_231: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_230, 0.1);  mul_230 = None
        mul_232: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_231, 0.9)
        add_194: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(mul_231, mul_232);  mul_231 = mul_232 = None
        unsqueeze_124: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_232, -1)
        unsqueeze_125: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        mul_233: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(mul_227, unsqueeze_125);  mul_227 = unsqueeze_125 = None
        unsqueeze_126: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_233, -1)
        unsqueeze_127: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        add_195: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_233, unsqueeze_127);  mul_233 = unsqueeze_127 = None
        convert_element_type_198: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(add_195, torch.bfloat16);  add_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_199: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_198, torch.float32);  convert_element_type_198 = None
        neg_31: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.neg.default(convert_element_type_199)
        exp_31: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.exp.default(neg_31);  neg_31 = None
        add_196: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(exp_31, 1);  exp_31 = None
        div_31: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_199, add_196);  convert_element_type_199 = add_196 = None
        convert_element_type_200: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(div_31, torch.bfloat16);  div_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_10: "bf16[128, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_200, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convert_element_type_201: "bf16[28][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_235, torch.bfloat16);  primals_235 = None
        convert_element_type_202: "bf16[28, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(primals_234, torch.bfloat16);  primals_234 = None
        convolution_52: "bf16[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.convolution.default(mean_10, convert_element_type_202, convert_element_type_201, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_203: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_52, torch.float32)
        neg_32: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.neg.default(convert_element_type_203)
        exp_32: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.exp.default(neg_32);  neg_32 = None
        add_197: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.add.Tensor(exp_32, 1);  exp_32 = None
        div_32: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_203, add_197);  convert_element_type_203 = add_197 = None
        convert_element_type_204: "bf16[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.prims.convert_element_type.default(div_32, torch.bfloat16);  div_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convert_element_type_205: "bf16[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_237, torch.bfloat16);  primals_237 = None
        convert_element_type_206: "bf16[672, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.prims.convert_element_type.default(primals_236, torch.bfloat16);  primals_236 = None
        convolution_53: "bf16[128, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_204, convert_element_type_206, convert_element_type_205, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_10: "bf16[128, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.sigmoid.default(convolution_53)
        mul_234: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_200, sigmoid_10);  convert_element_type_200 = sigmoid_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convert_element_type_207: "bf16[112, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(primals_238, torch.bfloat16);  primals_238 = None
        convolution_54: "bf16[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.convolution.default(mul_234, convert_element_type_207, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_198: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_239, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_208: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_54, torch.float32)
        var_mean_32 = torch.ops.aten.var_mean.correction(convert_element_type_208, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_208 = None
        getitem_64: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = var_mean_32[0]
        getitem_65: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = var_mean_32[1];  var_mean_32 = None
        add_199: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_64, 0.001)
        rsqrt_32: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
        sub_32: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.sub.Tensor(convolution_54, getitem_65)
        mul_235: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
        squeeze_96: "f32[112][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
        squeeze_97: "f32[112][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
        mul_236: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
        mul_237: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_240, 0.9)
        add_200: "f32[112][1]cuda:0" = torch.ops.aten.add.Tensor(mul_236, mul_237);  mul_236 = mul_237 = None
        squeeze_98: "f32[112][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
        mul_238: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0000398612827361);  squeeze_98 = None
        mul_239: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_238, 0.1);  mul_238 = None
        mul_240: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_241, 0.9)
        add_201: "f32[112][1]cuda:0" = torch.ops.aten.add.Tensor(mul_239, mul_240);  mul_239 = mul_240 = None
        unsqueeze_128: "f32[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_242, -1)
        unsqueeze_129: "f32[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
        mul_241: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.mul.Tensor(mul_235, unsqueeze_129);  mul_235 = unsqueeze_129 = None
        unsqueeze_130: "f32[112, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_243, -1);  primals_243 = None
        unsqueeze_131: "f32[112, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
        add_202: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.add.Tensor(mul_241, unsqueeze_131);  mul_241 = unsqueeze_131 = None
        convert_element_type_209: "bf16[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.prims.convert_element_type.default(add_202, torch.bfloat16);  add_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:338 in forward, code: x = self.drop_path(x) + shortcut
        add_203: "bf16[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_209, add_184);  convert_element_type_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convert_element_type_210: "bf16[672, 112, 1, 1][112, 1, 112, 112]cuda:0" = torch.ops.prims.convert_element_type.default(primals_244, torch.bfloat16);  primals_244 = None
        convolution_55: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.convolution.default(add_203, convert_element_type_210, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_204: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_245, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_211: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_55, torch.float32)
        var_mean_33 = torch.ops.aten.var_mean.correction(convert_element_type_211, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_211 = None
        getitem_66: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = var_mean_33[0]
        getitem_67: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = var_mean_33[1];  var_mean_33 = None
        add_205: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_66, 0.001)
        rsqrt_33: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_205);  add_205 = None
        sub_33: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convolution_55, getitem_67)
        mul_242: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
        squeeze_99: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3])
        mul_243: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1);  squeeze_99 = None
        mul_244: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_246, 0.9)
        add_206: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(mul_243, mul_244);  mul_243 = mul_244 = None
        squeeze_101: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
        mul_245: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0000398612827361);  squeeze_101 = None
        mul_246: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_245, 0.1);  mul_245 = None
        mul_247: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_247, 0.9)
        add_207: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
        unsqueeze_132: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_248, -1)
        unsqueeze_133: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
        mul_248: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(mul_242, unsqueeze_133);  mul_242 = unsqueeze_133 = None
        unsqueeze_134: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_249, -1)
        unsqueeze_135: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
        add_208: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_248, unsqueeze_135);  mul_248 = unsqueeze_135 = None
        convert_element_type_212: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(add_208, torch.bfloat16);  add_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_213: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_212, torch.float32);  convert_element_type_212 = None
        neg_33: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.neg.default(convert_element_type_213)
        exp_33: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.exp.default(neg_33);  neg_33 = None
        add_209: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(exp_33, 1);  exp_33 = None
        div_33: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_213, add_209);  convert_element_type_213 = add_209 = None
        convert_element_type_214: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(div_33, torch.bfloat16);  div_33 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_4: "bf16[128, 672, 17, 17][194208, 1, 11424, 672]cuda:0" = torch.ops.aten.constant_pad_nd.default(convert_element_type_214, [1, 2, 1, 2], 0.0);  convert_element_type_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv2d_same.py:28 in conv2d_same, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
        convert_element_type_215: "bf16[672, 1, 5, 5][25, 1, 5, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_250, torch.bfloat16);  primals_250 = None
        convolution_56: "bf16[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.convolution.default(constant_pad_nd_4, convert_element_type_215, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 672)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_210: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_251, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_216: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_56, torch.float32)
        var_mean_34 = torch.ops.aten.var_mean.correction(convert_element_type_216, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_216 = None
        getitem_68: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = var_mean_34[0]
        getitem_69: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = var_mean_34[1];  var_mean_34 = None
        add_211: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_68, 0.001)
        rsqrt_34: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_211);  add_211 = None
        sub_34: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.sub.Tensor(convolution_56, getitem_69)
        mul_249: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
        squeeze_102: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3])
        mul_250: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1);  squeeze_102 = None
        mul_251: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_252, 0.9)
        add_212: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(mul_250, mul_251);  mul_250 = mul_251 = None
        squeeze_104: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
        mul_252: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0001594642002871);  squeeze_104 = None
        mul_253: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_252, 0.1);  mul_252 = None
        mul_254: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_253, 0.9)
        add_213: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(mul_253, mul_254);  mul_253 = mul_254 = None
        unsqueeze_136: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_254, -1)
        unsqueeze_137: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
        mul_255: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(mul_249, unsqueeze_137);  mul_249 = unsqueeze_137 = None
        unsqueeze_138: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_255, -1)
        unsqueeze_139: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
        add_214: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_255, unsqueeze_139);  mul_255 = unsqueeze_139 = None
        convert_element_type_217: "bf16[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.prims.convert_element_type.default(add_214, torch.bfloat16);  add_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_218: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_217, torch.float32);  convert_element_type_217 = None
        neg_34: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.neg.default(convert_element_type_218)
        exp_34: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.exp.default(neg_34);  neg_34 = None
        add_215: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.add.Tensor(exp_34, 1);  exp_34 = None
        div_34: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_218, add_215);  convert_element_type_218 = add_215 = None
        convert_element_type_219: "bf16[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.prims.convert_element_type.default(div_34, torch.bfloat16);  div_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_11: "bf16[128, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_219, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convert_element_type_220: "bf16[28][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_257, torch.bfloat16);  primals_257 = None
        convert_element_type_221: "bf16[28, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(primals_256, torch.bfloat16);  primals_256 = None
        convolution_57: "bf16[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.convolution.default(mean_11, convert_element_type_221, convert_element_type_220, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_222: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_57, torch.float32)
        neg_35: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.neg.default(convert_element_type_222)
        exp_35: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.exp.default(neg_35);  neg_35 = None
        add_216: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.add.Tensor(exp_35, 1);  exp_35 = None
        div_35: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_222, add_216);  convert_element_type_222 = add_216 = None
        convert_element_type_223: "bf16[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.prims.convert_element_type.default(div_35, torch.bfloat16);  div_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convert_element_type_224: "bf16[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_259, torch.bfloat16);  primals_259 = None
        convert_element_type_225: "bf16[672, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.prims.convert_element_type.default(primals_258, torch.bfloat16);  primals_258 = None
        convolution_58: "bf16[128, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_223, convert_element_type_225, convert_element_type_224, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_11: "bf16[128, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.sigmoid.default(convolution_58)
        mul_256: "bf16[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_219, sigmoid_11);  convert_element_type_219 = sigmoid_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convert_element_type_226: "bf16[192, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(primals_260, torch.bfloat16);  primals_260 = None
        convolution_59: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.convolution.default(mul_256, convert_element_type_226, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_217: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_261, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_227: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_59, torch.float32)
        var_mean_35 = torch.ops.aten.var_mean.correction(convert_element_type_227, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_227 = None
        getitem_70: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_35[0]
        getitem_71: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_35[1];  var_mean_35 = None
        add_218: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_70, 0.001)
        rsqrt_35: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
        sub_35: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_59, getitem_71)
        mul_257: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
        squeeze_105: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
        squeeze_106: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
        mul_258: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
        mul_259: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_262, 0.9)
        add_219: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_258, mul_259);  mul_258 = mul_259 = None
        squeeze_107: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
        mul_260: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_107, 1.0001594642002871);  squeeze_107 = None
        mul_261: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_260, 0.1);  mul_260 = None
        mul_262: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_263, 0.9)
        add_220: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_261, mul_262);  mul_261 = mul_262 = None
        unsqueeze_140: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_264, -1)
        unsqueeze_141: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
        mul_263: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_257, unsqueeze_141);  mul_257 = unsqueeze_141 = None
        unsqueeze_142: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_265, -1);  primals_265 = None
        unsqueeze_143: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
        add_221: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_263, unsqueeze_143);  mul_263 = unsqueeze_143 = None
        convert_element_type_228: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_221, torch.bfloat16);  add_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convert_element_type_229: "bf16[1152, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(primals_266, torch.bfloat16);  primals_266 = None
        convolution_60: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_228, convert_element_type_229, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_222: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_267, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_230: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_60, torch.float32)
        var_mean_36 = torch.ops.aten.var_mean.correction(convert_element_type_230, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_230 = None
        getitem_72: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = var_mean_36[0]
        getitem_73: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = var_mean_36[1];  var_mean_36 = None
        add_223: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_72, 0.001)
        rsqrt_36: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_223);  add_223 = None
        sub_36: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convolution_60, getitem_73)
        mul_264: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
        squeeze_108: "f32[1152][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3])
        mul_265: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1);  squeeze_108 = None
        mul_266: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_268, 0.9)
        add_224: "f32[1152][1]cuda:0" = torch.ops.aten.add.Tensor(mul_265, mul_266);  mul_265 = mul_266 = None
        squeeze_110: "f32[1152][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
        mul_267: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0001594642002871);  squeeze_110 = None
        mul_268: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_267, 0.1);  mul_267 = None
        mul_269: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_269, 0.9)
        add_225: "f32[1152][1]cuda:0" = torch.ops.aten.add.Tensor(mul_268, mul_269);  mul_268 = mul_269 = None
        unsqueeze_144: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_270, -1)
        unsqueeze_145: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
        mul_270: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(mul_264, unsqueeze_145);  mul_264 = unsqueeze_145 = None
        unsqueeze_146: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_271, -1)
        unsqueeze_147: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
        add_226: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_270, unsqueeze_147);  mul_270 = unsqueeze_147 = None
        convert_element_type_231: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(add_226, torch.bfloat16);  add_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_232: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_231, torch.float32);  convert_element_type_231 = None
        neg_36: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.neg.default(convert_element_type_232)
        exp_36: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.exp.default(neg_36);  neg_36 = None
        add_227: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(exp_36, 1);  exp_36 = None
        div_36: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_232, add_227);  convert_element_type_232 = add_227 = None
        convert_element_type_233: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(div_36, torch.bfloat16);  div_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convert_element_type_234: "bf16[1152, 1, 5, 5][25, 1, 5, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_272, torch.bfloat16);  primals_272 = None
        convolution_61: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_233, convert_element_type_234, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_228: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_273, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_235: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_61, torch.float32)
        var_mean_37 = torch.ops.aten.var_mean.correction(convert_element_type_235, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_235 = None
        getitem_74: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = var_mean_37[0]
        getitem_75: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = var_mean_37[1];  var_mean_37 = None
        add_229: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_74, 0.001)
        rsqrt_37: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
        sub_37: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convolution_61, getitem_75)
        mul_271: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
        squeeze_111: "f32[1152][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3])
        mul_272: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1);  squeeze_111 = None
        mul_273: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_274, 0.9)
        add_230: "f32[1152][1]cuda:0" = torch.ops.aten.add.Tensor(mul_272, mul_273);  mul_272 = mul_273 = None
        squeeze_113: "f32[1152][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
        mul_274: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0001594642002871);  squeeze_113 = None
        mul_275: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_274, 0.1);  mul_274 = None
        mul_276: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_275, 0.9)
        add_231: "f32[1152][1]cuda:0" = torch.ops.aten.add.Tensor(mul_275, mul_276);  mul_275 = mul_276 = None
        unsqueeze_148: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_276, -1)
        unsqueeze_149: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
        mul_277: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(mul_271, unsqueeze_149);  mul_271 = unsqueeze_149 = None
        unsqueeze_150: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_277, -1)
        unsqueeze_151: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
        add_232: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_277, unsqueeze_151);  mul_277 = unsqueeze_151 = None
        convert_element_type_236: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(add_232, torch.bfloat16);  add_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_237: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_236, torch.float32);  convert_element_type_236 = None
        neg_37: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.neg.default(convert_element_type_237)
        exp_37: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.exp.default(neg_37);  neg_37 = None
        add_233: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(exp_37, 1);  exp_37 = None
        div_37: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_237, add_233);  convert_element_type_237 = add_233 = None
        convert_element_type_238: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(div_37, torch.bfloat16);  div_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_12: "bf16[128, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_238, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convert_element_type_239: "bf16[48][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_279, torch.bfloat16);  primals_279 = None
        convert_element_type_240: "bf16[48, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(primals_278, torch.bfloat16);  primals_278 = None
        convolution_62: "bf16[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.convolution.default(mean_12, convert_element_type_240, convert_element_type_239, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_241: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_62, torch.float32)
        neg_38: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.neg.default(convert_element_type_241)
        exp_38: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.exp.default(neg_38);  neg_38 = None
        add_234: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.add.Tensor(exp_38, 1);  exp_38 = None
        div_38: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_241, add_234);  convert_element_type_241 = add_234 = None
        convert_element_type_242: "bf16[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(div_38, torch.bfloat16);  div_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convert_element_type_243: "bf16[1152][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_281, torch.bfloat16);  primals_281 = None
        convert_element_type_244: "bf16[1152, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(primals_280, torch.bfloat16);  primals_280 = None
        convolution_63: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_242, convert_element_type_244, convert_element_type_243, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_12: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.aten.sigmoid.default(convolution_63)
        mul_278: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_238, sigmoid_12);  convert_element_type_238 = sigmoid_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convert_element_type_245: "bf16[192, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(primals_282, torch.bfloat16);  primals_282 = None
        convolution_64: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.convolution.default(mul_278, convert_element_type_245, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_235: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_283, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_246: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_64, torch.float32)
        var_mean_38 = torch.ops.aten.var_mean.correction(convert_element_type_246, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_246 = None
        getitem_76: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_38[0]
        getitem_77: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_38[1];  var_mean_38 = None
        add_236: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_76, 0.001)
        rsqrt_38: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_236);  add_236 = None
        sub_38: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_64, getitem_77)
        mul_279: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
        squeeze_114: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
        squeeze_115: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
        mul_280: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
        mul_281: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_284, 0.9)
        add_237: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_280, mul_281);  mul_280 = mul_281 = None
        squeeze_116: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
        mul_282: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0001594642002871);  squeeze_116 = None
        mul_283: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_282, 0.1);  mul_282 = None
        mul_284: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_285, 0.9)
        add_238: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_283, mul_284);  mul_283 = mul_284 = None
        unsqueeze_152: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_286, -1)
        unsqueeze_153: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
        mul_285: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_279, unsqueeze_153);  mul_279 = unsqueeze_153 = None
        unsqueeze_154: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_287, -1);  primals_287 = None
        unsqueeze_155: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
        add_239: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_285, unsqueeze_155);  mul_285 = unsqueeze_155 = None
        convert_element_type_247: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_239, torch.bfloat16);  add_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:338 in forward, code: x = self.drop_path(x) + shortcut
        add_240: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_247, convert_element_type_228);  convert_element_type_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convert_element_type_248: "bf16[1152, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(primals_288, torch.bfloat16);  primals_288 = None
        convolution_65: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.convolution.default(add_240, convert_element_type_248, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_241: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_289, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_249: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_65, torch.float32)
        var_mean_39 = torch.ops.aten.var_mean.correction(convert_element_type_249, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_249 = None
        getitem_78: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = var_mean_39[0]
        getitem_79: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = var_mean_39[1];  var_mean_39 = None
        add_242: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_78, 0.001)
        rsqrt_39: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_242);  add_242 = None
        sub_39: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convolution_65, getitem_79)
        mul_286: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
        squeeze_117: "f32[1152][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3])
        mul_287: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1);  squeeze_117 = None
        mul_288: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_290, 0.9)
        add_243: "f32[1152][1]cuda:0" = torch.ops.aten.add.Tensor(mul_287, mul_288);  mul_287 = mul_288 = None
        squeeze_119: "f32[1152][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
        mul_289: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_119, 1.0001594642002871);  squeeze_119 = None
        mul_290: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_289, 0.1);  mul_289 = None
        mul_291: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_291, 0.9)
        add_244: "f32[1152][1]cuda:0" = torch.ops.aten.add.Tensor(mul_290, mul_291);  mul_290 = mul_291 = None
        unsqueeze_156: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_292, -1)
        unsqueeze_157: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
        mul_292: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(mul_286, unsqueeze_157);  mul_286 = unsqueeze_157 = None
        unsqueeze_158: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_293, -1)
        unsqueeze_159: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
        add_245: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_292, unsqueeze_159);  mul_292 = unsqueeze_159 = None
        convert_element_type_250: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(add_245, torch.bfloat16);  add_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_251: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_250, torch.float32);  convert_element_type_250 = None
        neg_39: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.neg.default(convert_element_type_251)
        exp_39: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.exp.default(neg_39);  neg_39 = None
        add_246: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(exp_39, 1);  exp_39 = None
        div_39: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_251, add_246);  convert_element_type_251 = add_246 = None
        convert_element_type_252: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(div_39, torch.bfloat16);  div_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convert_element_type_253: "bf16[1152, 1, 5, 5][25, 1, 5, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_294, torch.bfloat16);  primals_294 = None
        convolution_66: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_252, convert_element_type_253, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_247: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_295, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_254: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_66, torch.float32)
        var_mean_40 = torch.ops.aten.var_mean.correction(convert_element_type_254, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_254 = None
        getitem_80: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = var_mean_40[0]
        getitem_81: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = var_mean_40[1];  var_mean_40 = None
        add_248: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_80, 0.001)
        rsqrt_40: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_248);  add_248 = None
        sub_40: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convolution_66, getitem_81)
        mul_293: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
        squeeze_120: "f32[1152][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3])
        mul_294: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1);  squeeze_120 = None
        mul_295: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_296, 0.9)
        add_249: "f32[1152][1]cuda:0" = torch.ops.aten.add.Tensor(mul_294, mul_295);  mul_294 = mul_295 = None
        squeeze_122: "f32[1152][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
        mul_296: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_122, 1.0001594642002871);  squeeze_122 = None
        mul_297: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_296, 0.1);  mul_296 = None
        mul_298: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_297, 0.9)
        add_250: "f32[1152][1]cuda:0" = torch.ops.aten.add.Tensor(mul_297, mul_298);  mul_297 = mul_298 = None
        unsqueeze_160: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_298, -1)
        unsqueeze_161: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
        mul_299: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(mul_293, unsqueeze_161);  mul_293 = unsqueeze_161 = None
        unsqueeze_162: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_299, -1)
        unsqueeze_163: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
        add_251: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_299, unsqueeze_163);  mul_299 = unsqueeze_163 = None
        convert_element_type_255: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(add_251, torch.bfloat16);  add_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_256: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_255, torch.float32);  convert_element_type_255 = None
        neg_40: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.neg.default(convert_element_type_256)
        exp_40: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.exp.default(neg_40);  neg_40 = None
        add_252: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(exp_40, 1);  exp_40 = None
        div_40: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_256, add_252);  convert_element_type_256 = add_252 = None
        convert_element_type_257: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(div_40, torch.bfloat16);  div_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_13: "bf16[128, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_257, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convert_element_type_258: "bf16[48][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_301, torch.bfloat16);  primals_301 = None
        convert_element_type_259: "bf16[48, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(primals_300, torch.bfloat16);  primals_300 = None
        convolution_67: "bf16[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.convolution.default(mean_13, convert_element_type_259, convert_element_type_258, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_260: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_67, torch.float32)
        neg_41: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.neg.default(convert_element_type_260)
        exp_41: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.exp.default(neg_41);  neg_41 = None
        add_253: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.add.Tensor(exp_41, 1);  exp_41 = None
        div_41: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_260, add_253);  convert_element_type_260 = add_253 = None
        convert_element_type_261: "bf16[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(div_41, torch.bfloat16);  div_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convert_element_type_262: "bf16[1152][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_303, torch.bfloat16);  primals_303 = None
        convert_element_type_263: "bf16[1152, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(primals_302, torch.bfloat16);  primals_302 = None
        convolution_68: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_261, convert_element_type_263, convert_element_type_262, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_13: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.aten.sigmoid.default(convolution_68)
        mul_300: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_257, sigmoid_13);  convert_element_type_257 = sigmoid_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convert_element_type_264: "bf16[192, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(primals_304, torch.bfloat16);  primals_304 = None
        convolution_69: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.convolution.default(mul_300, convert_element_type_264, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_254: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_305, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_265: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_69, torch.float32)
        var_mean_41 = torch.ops.aten.var_mean.correction(convert_element_type_265, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_265 = None
        getitem_82: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_41[0]
        getitem_83: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_41[1];  var_mean_41 = None
        add_255: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_82, 0.001)
        rsqrt_41: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_255);  add_255 = None
        sub_41: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_69, getitem_83)
        mul_301: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
        squeeze_123: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
        squeeze_124: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
        mul_302: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
        mul_303: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_306, 0.9)
        add_256: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_302, mul_303);  mul_302 = mul_303 = None
        squeeze_125: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
        mul_304: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_125, 1.0001594642002871);  squeeze_125 = None
        mul_305: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_304, 0.1);  mul_304 = None
        mul_306: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_307, 0.9)
        add_257: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_305, mul_306);  mul_305 = mul_306 = None
        unsqueeze_164: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_308, -1)
        unsqueeze_165: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
        mul_307: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_165);  mul_301 = unsqueeze_165 = None
        unsqueeze_166: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_309, -1);  primals_309 = None
        unsqueeze_167: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
        add_258: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_307, unsqueeze_167);  mul_307 = unsqueeze_167 = None
        convert_element_type_266: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_258, torch.bfloat16);  add_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:338 in forward, code: x = self.drop_path(x) + shortcut
        add_259: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_266, add_240);  convert_element_type_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convert_element_type_267: "bf16[1152, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(primals_310, torch.bfloat16);  primals_310 = None
        convolution_70: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.convolution.default(add_259, convert_element_type_267, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_260: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_311, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_268: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_70, torch.float32)
        var_mean_42 = torch.ops.aten.var_mean.correction(convert_element_type_268, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_268 = None
        getitem_84: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = var_mean_42[0]
        getitem_85: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = var_mean_42[1];  var_mean_42 = None
        add_261: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_84, 0.001)
        rsqrt_42: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_261);  add_261 = None
        sub_42: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convolution_70, getitem_85)
        mul_308: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
        squeeze_126: "f32[1152][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3])
        mul_309: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1);  squeeze_126 = None
        mul_310: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_312, 0.9)
        add_262: "f32[1152][1]cuda:0" = torch.ops.aten.add.Tensor(mul_309, mul_310);  mul_309 = mul_310 = None
        squeeze_128: "f32[1152][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
        mul_311: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_128, 1.0001594642002871);  squeeze_128 = None
        mul_312: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_311, 0.1);  mul_311 = None
        mul_313: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_313, 0.9)
        add_263: "f32[1152][1]cuda:0" = torch.ops.aten.add.Tensor(mul_312, mul_313);  mul_312 = mul_313 = None
        unsqueeze_168: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_314, -1)
        unsqueeze_169: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
        mul_314: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(mul_308, unsqueeze_169);  mul_308 = unsqueeze_169 = None
        unsqueeze_170: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_315, -1)
        unsqueeze_171: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
        add_264: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_314, unsqueeze_171);  mul_314 = unsqueeze_171 = None
        convert_element_type_269: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(add_264, torch.bfloat16);  add_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_270: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_269, torch.float32);  convert_element_type_269 = None
        neg_42: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.neg.default(convert_element_type_270)
        exp_42: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.exp.default(neg_42);  neg_42 = None
        add_265: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(exp_42, 1);  exp_42 = None
        div_42: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_270, add_265);  convert_element_type_270 = add_265 = None
        convert_element_type_271: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(div_42, torch.bfloat16);  div_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convert_element_type_272: "bf16[1152, 1, 5, 5][25, 1, 5, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_316, torch.bfloat16);  primals_316 = None
        convolution_71: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_271, convert_element_type_272, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_266: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_317, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_273: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_71, torch.float32)
        var_mean_43 = torch.ops.aten.var_mean.correction(convert_element_type_273, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_273 = None
        getitem_86: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = var_mean_43[0]
        getitem_87: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = var_mean_43[1];  var_mean_43 = None
        add_267: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_86, 0.001)
        rsqrt_43: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_267);  add_267 = None
        sub_43: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convolution_71, getitem_87)
        mul_315: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
        squeeze_129: "f32[1152][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3])
        mul_316: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1);  squeeze_129 = None
        mul_317: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_318, 0.9)
        add_268: "f32[1152][1]cuda:0" = torch.ops.aten.add.Tensor(mul_316, mul_317);  mul_316 = mul_317 = None
        squeeze_131: "f32[1152][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
        mul_318: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_131, 1.0001594642002871);  squeeze_131 = None
        mul_319: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_318, 0.1);  mul_318 = None
        mul_320: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_319, 0.9)
        add_269: "f32[1152][1]cuda:0" = torch.ops.aten.add.Tensor(mul_319, mul_320);  mul_319 = mul_320 = None
        unsqueeze_172: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_320, -1)
        unsqueeze_173: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
        mul_321: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(mul_315, unsqueeze_173);  mul_315 = unsqueeze_173 = None
        unsqueeze_174: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_321, -1)
        unsqueeze_175: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
        add_270: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_321, unsqueeze_175);  mul_321 = unsqueeze_175 = None
        convert_element_type_274: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(add_270, torch.bfloat16);  add_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_275: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_274, torch.float32);  convert_element_type_274 = None
        neg_43: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.neg.default(convert_element_type_275)
        exp_43: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.exp.default(neg_43);  neg_43 = None
        add_271: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(exp_43, 1);  exp_43 = None
        div_43: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_275, add_271);  convert_element_type_275 = add_271 = None
        convert_element_type_276: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(div_43, torch.bfloat16);  div_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_14: "bf16[128, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_276, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convert_element_type_277: "bf16[48][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_323, torch.bfloat16);  primals_323 = None
        convert_element_type_278: "bf16[48, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(primals_322, torch.bfloat16);  primals_322 = None
        convolution_72: "bf16[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.convolution.default(mean_14, convert_element_type_278, convert_element_type_277, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_279: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_72, torch.float32)
        neg_44: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.neg.default(convert_element_type_279)
        exp_44: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.exp.default(neg_44);  neg_44 = None
        add_272: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.add.Tensor(exp_44, 1);  exp_44 = None
        div_44: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_279, add_272);  convert_element_type_279 = add_272 = None
        convert_element_type_280: "bf16[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(div_44, torch.bfloat16);  div_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convert_element_type_281: "bf16[1152][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_325, torch.bfloat16);  primals_325 = None
        convert_element_type_282: "bf16[1152, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(primals_324, torch.bfloat16);  primals_324 = None
        convolution_73: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_280, convert_element_type_282, convert_element_type_281, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_14: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.aten.sigmoid.default(convolution_73)
        mul_322: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_276, sigmoid_14);  convert_element_type_276 = sigmoid_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convert_element_type_283: "bf16[192, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(primals_326, torch.bfloat16);  primals_326 = None
        convolution_74: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.convolution.default(mul_322, convert_element_type_283, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_273: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_327, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_284: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_74, torch.float32)
        var_mean_44 = torch.ops.aten.var_mean.correction(convert_element_type_284, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_284 = None
        getitem_88: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_44[0]
        getitem_89: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_44[1];  var_mean_44 = None
        add_274: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_88, 0.001)
        rsqrt_44: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_274);  add_274 = None
        sub_44: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_74, getitem_89)
        mul_323: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
        squeeze_132: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
        squeeze_133: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
        mul_324: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_132, 0.1)
        mul_325: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_328, 0.9)
        add_275: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_324, mul_325);  mul_324 = mul_325 = None
        squeeze_134: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
        mul_326: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_134, 1.0001594642002871);  squeeze_134 = None
        mul_327: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_326, 0.1);  mul_326 = None
        mul_328: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_329, 0.9)
        add_276: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_327, mul_328);  mul_327 = mul_328 = None
        unsqueeze_176: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_330, -1)
        unsqueeze_177: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
        mul_329: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_323, unsqueeze_177);  mul_323 = unsqueeze_177 = None
        unsqueeze_178: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_331, -1);  primals_331 = None
        unsqueeze_179: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
        add_277: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_329, unsqueeze_179);  mul_329 = unsqueeze_179 = None
        convert_element_type_285: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_277, torch.bfloat16);  add_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:338 in forward, code: x = self.drop_path(x) + shortcut
        add_278: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_285, add_259);  convert_element_type_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convert_element_type_286: "bf16[1152, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(primals_332, torch.bfloat16);  primals_332 = None
        convolution_75: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.convolution.default(add_278, convert_element_type_286, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_279: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_333, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_287: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_75, torch.float32)
        var_mean_45 = torch.ops.aten.var_mean.correction(convert_element_type_287, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_287 = None
        getitem_90: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = var_mean_45[0]
        getitem_91: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = var_mean_45[1];  var_mean_45 = None
        add_280: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_90, 0.001)
        rsqrt_45: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_280);  add_280 = None
        sub_45: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convolution_75, getitem_91)
        mul_330: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
        squeeze_135: "f32[1152][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3])
        mul_331: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_135, 0.1);  squeeze_135 = None
        mul_332: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_334, 0.9)
        add_281: "f32[1152][1]cuda:0" = torch.ops.aten.add.Tensor(mul_331, mul_332);  mul_331 = mul_332 = None
        squeeze_137: "f32[1152][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
        mul_333: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_137, 1.0001594642002871);  squeeze_137 = None
        mul_334: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_333, 0.1);  mul_333 = None
        mul_335: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_335, 0.9)
        add_282: "f32[1152][1]cuda:0" = torch.ops.aten.add.Tensor(mul_334, mul_335);  mul_334 = mul_335 = None
        unsqueeze_180: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_336, -1)
        unsqueeze_181: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
        mul_336: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(mul_330, unsqueeze_181);  mul_330 = unsqueeze_181 = None
        unsqueeze_182: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_337, -1)
        unsqueeze_183: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
        add_283: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_336, unsqueeze_183);  mul_336 = unsqueeze_183 = None
        convert_element_type_288: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(add_283, torch.bfloat16);  add_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_289: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_288, torch.float32);  convert_element_type_288 = None
        neg_45: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.neg.default(convert_element_type_289)
        exp_45: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.exp.default(neg_45);  neg_45 = None
        add_284: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(exp_45, 1);  exp_45 = None
        div_45: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_289, add_284);  convert_element_type_289 = add_284 = None
        convert_element_type_290: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(div_45, torch.bfloat16);  div_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convert_element_type_291: "bf16[1152, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_338, torch.bfloat16);  primals_338 = None
        convolution_76: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_290, convert_element_type_291, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1152)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_285: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_339, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_292: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_76, torch.float32)
        var_mean_46 = torch.ops.aten.var_mean.correction(convert_element_type_292, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_292 = None
        getitem_92: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = var_mean_46[0]
        getitem_93: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = var_mean_46[1];  var_mean_46 = None
        add_286: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_92, 0.001)
        rsqrt_46: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_286);  add_286 = None
        sub_46: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convolution_76, getitem_93)
        mul_337: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
        squeeze_138: "f32[1152][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3])
        mul_338: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_138, 0.1);  squeeze_138 = None
        mul_339: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_340, 0.9)
        add_287: "f32[1152][1]cuda:0" = torch.ops.aten.add.Tensor(mul_338, mul_339);  mul_338 = mul_339 = None
        squeeze_140: "f32[1152][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_92, [0, 2, 3]);  getitem_92 = None
        mul_340: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_140, 1.0001594642002871);  squeeze_140 = None
        mul_341: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_340, 0.1);  mul_340 = None
        mul_342: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_341, 0.9)
        add_288: "f32[1152][1]cuda:0" = torch.ops.aten.add.Tensor(mul_341, mul_342);  mul_341 = mul_342 = None
        unsqueeze_184: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_342, -1)
        unsqueeze_185: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
        mul_343: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(mul_337, unsqueeze_185);  mul_337 = unsqueeze_185 = None
        unsqueeze_186: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_343, -1)
        unsqueeze_187: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
        add_289: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_343, unsqueeze_187);  mul_343 = unsqueeze_187 = None
        convert_element_type_293: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(add_289, torch.bfloat16);  add_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_294: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_293, torch.float32);  convert_element_type_293 = None
        neg_46: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.neg.default(convert_element_type_294)
        exp_46: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.exp.default(neg_46);  neg_46 = None
        add_290: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(exp_46, 1);  exp_46 = None
        div_46: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_294, add_290);  convert_element_type_294 = add_290 = None
        convert_element_type_295: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(div_46, torch.bfloat16);  div_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_15: "bf16[128, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_295, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        convert_element_type_296: "bf16[48][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_345, torch.bfloat16);  primals_345 = None
        convert_element_type_297: "bf16[48, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(primals_344, torch.bfloat16);  primals_344 = None
        convolution_77: "bf16[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.convolution.default(mean_15, convert_element_type_297, convert_element_type_296, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_298: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_77, torch.float32)
        neg_47: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.neg.default(convert_element_type_298)
        exp_47: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.exp.default(neg_47);  neg_47 = None
        add_291: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.add.Tensor(exp_47, 1);  exp_47 = None
        div_47: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_298, add_291);  convert_element_type_298 = add_291 = None
        convert_element_type_299: "bf16[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(div_47, torch.bfloat16);  div_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        convert_element_type_300: "bf16[1152][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_347, torch.bfloat16);  primals_347 = None
        convert_element_type_301: "bf16[1152, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(primals_346, torch.bfloat16);  primals_346 = None
        convolution_78: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_299, convert_element_type_301, convert_element_type_300, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        sigmoid_15: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.aten.sigmoid.default(convolution_78)
        mul_344: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_295, sigmoid_15);  convert_element_type_295 = sigmoid_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convert_element_type_302: "bf16[320, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(primals_348, torch.bfloat16);  primals_348 = None
        convolution_79: "bf16[128, 320, 7, 7][15680, 1, 2240, 320]cuda:0" = torch.ops.aten.convolution.default(mul_344, convert_element_type_302, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_292: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_349, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_303: "f32[128, 320, 7, 7][15680, 1, 2240, 320]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_79, torch.float32)
        var_mean_47 = torch.ops.aten.var_mean.correction(convert_element_type_303, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_303 = None
        getitem_94: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = var_mean_47[0]
        getitem_95: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = var_mean_47[1];  var_mean_47 = None
        add_293: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_94, 0.001)
        rsqrt_47: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_293);  add_293 = None
        sub_47: "f32[128, 320, 7, 7][15680, 1, 2240, 320]cuda:0" = torch.ops.aten.sub.Tensor(convolution_79, getitem_95)
        mul_345: "f32[128, 320, 7, 7][15680, 1, 2240, 320]cuda:0" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
        squeeze_141: "f32[320][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
        squeeze_142: "f32[320][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
        mul_346: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_141, 0.1)
        mul_347: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_350, 0.9)
        add_294: "f32[320][1]cuda:0" = torch.ops.aten.add.Tensor(mul_346, mul_347);  mul_346 = mul_347 = None
        squeeze_143: "f32[320][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_94, [0, 2, 3]);  getitem_94 = None
        mul_348: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_143, 1.0001594642002871);  squeeze_143 = None
        mul_349: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_348, 0.1);  mul_348 = None
        mul_350: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_351, 0.9)
        add_295: "f32[320][1]cuda:0" = torch.ops.aten.add.Tensor(mul_349, mul_350);  mul_349 = mul_350 = None
        unsqueeze_188: "f32[320, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_352, -1)
        unsqueeze_189: "f32[320, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
        mul_351: "f32[128, 320, 7, 7][15680, 1, 2240, 320]cuda:0" = torch.ops.aten.mul.Tensor(mul_345, unsqueeze_189);  mul_345 = unsqueeze_189 = None
        unsqueeze_190: "f32[320, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_353, -1);  primals_353 = None
        unsqueeze_191: "f32[320, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
        add_296: "f32[128, 320, 7, 7][15680, 1, 2240, 320]cuda:0" = torch.ops.aten.add.Tensor(mul_351, unsqueeze_191);  mul_351 = unsqueeze_191 = None
        convert_element_type_304: "bf16[128, 320, 7, 7][15680, 1, 2240, 320]cuda:0" = torch.ops.prims.convert_element_type.default(add_296, torch.bfloat16);  add_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/efficientnet.py:327 in forward_features, code: x = self.conv_head(x)
        convert_element_type_305: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.prims.convert_element_type.default(primals_354, torch.bfloat16);  primals_354 = None
        convolution_80: "bf16[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_304, convert_element_type_305, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_297: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_355, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_306: "f32[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_80, torch.float32)
        var_mean_48 = torch.ops.aten.var_mean.correction(convert_element_type_306, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_306 = None
        getitem_96: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = var_mean_48[0]
        getitem_97: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = var_mean_48[1];  var_mean_48 = None
        add_298: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_96, 0.001)
        rsqrt_48: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_298);  add_298 = None
        sub_48: "f32[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.sub.Tensor(convolution_80, getitem_97)
        mul_352: "f32[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
        squeeze_144: "f32[1280][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3])
        mul_353: "f32[1280][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_144, 0.1);  squeeze_144 = None
        mul_354: "f32[1280][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_356, 0.9)
        add_299: "f32[1280][1]cuda:0" = torch.ops.aten.add.Tensor(mul_353, mul_354);  mul_353 = mul_354 = None
        squeeze_146: "f32[1280][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
        mul_355: "f32[1280][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_146, 1.0001594642002871);  squeeze_146 = None
        mul_356: "f32[1280][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_355, 0.1);  mul_355 = None
        mul_357: "f32[1280][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_357, 0.9)
        add_300: "f32[1280][1]cuda:0" = torch.ops.aten.add.Tensor(mul_356, mul_357);  mul_356 = mul_357 = None
        unsqueeze_192: "f32[1280, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_358, -1)
        unsqueeze_193: "f32[1280, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
        mul_358: "f32[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_352, unsqueeze_193);  mul_352 = unsqueeze_193 = None
        unsqueeze_194: "f32[1280, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_359, -1)
        unsqueeze_195: "f32[1280, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
        add_301: "f32[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.add.Tensor(mul_358, unsqueeze_195);  mul_358 = unsqueeze_195 = None
        convert_element_type_307: "bf16[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(add_301, torch.bfloat16);  add_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_308: "f32[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_307, torch.float32);  convert_element_type_307 = None
        neg_48: "f32[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.neg.default(convert_element_type_308)
        exp_48: "f32[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.exp.default(neg_48);  neg_48 = None
        add_302: "f32[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.add.Tensor(exp_48, 1);  exp_48 = None
        div_48: "f32[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_308, add_302);  convert_element_type_308 = add_302 = None
        convert_element_type_309: "bf16[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(div_48, torch.bfloat16);  div_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_16: "bf16[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(convert_element_type_309, [-1, -2], True);  convert_element_type_309 = None
        as_strided: "bf16[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.as_strided.default(mean_16, [128, 1280, 1, 1], [1280, 1, 1280, 1280]);  mean_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view: "bf16[128, 1280][1280, 1]cuda:0" = torch.ops.aten.reshape.default(as_strided, [128, 1280]);  as_strided = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/efficientnet.py:344 in forward_head, code: return x if pre_logits else self.classifier(x)
        convert_element_type_310: "bf16[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_361, torch.bfloat16);  primals_361 = None
        convert_element_type_311: "bf16[1000, 1280][1280, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_360, torch.bfloat16);  primals_360 = None
        permute: "bf16[1280, 1000][1, 1280]cuda:0" = torch.ops.aten.permute.default(convert_element_type_311, [1, 0]);  convert_element_type_311 = None
        addmm: "bf16[128, 1000][1000, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_310, view, permute);  convert_element_type_310 = None
        permute_1: "bf16[1000, 1280][1280, 1]cuda:0" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_208: "f32[1, 320][320, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_141, 0);  squeeze_141 = None
        unsqueeze_209: "f32[1, 320, 1][320, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_208, 2);  unsqueeze_208 = None
        unsqueeze_210: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 3);  unsqueeze_209 = None
        unsqueeze_244: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
        unsqueeze_245: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 2);  unsqueeze_244 = None
        unsqueeze_246: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 3);  unsqueeze_245 = None
        unsqueeze_280: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
        unsqueeze_281: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 2);  unsqueeze_280 = None
        unsqueeze_282: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 3);  unsqueeze_281 = None
        unsqueeze_316: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
        unsqueeze_317: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_316, 2);  unsqueeze_316 = None
        unsqueeze_318: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 3);  unsqueeze_317 = None
        unsqueeze_352: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
        unsqueeze_353: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_352, 2);  unsqueeze_352 = None
        unsqueeze_354: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 3);  unsqueeze_353 = None
        unsqueeze_388: "f32[1, 112][112, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
        unsqueeze_389: "f32[1, 112, 1][112, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 2);  unsqueeze_388 = None
        unsqueeze_390: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 3);  unsqueeze_389 = None
        unsqueeze_424: "f32[1, 112][112, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
        unsqueeze_425: "f32[1, 112, 1][112, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 2);  unsqueeze_424 = None
        unsqueeze_426: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 3);  unsqueeze_425 = None
        unsqueeze_460: "f32[1, 112][112, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
        unsqueeze_461: "f32[1, 112, 1][112, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_460, 2);  unsqueeze_460 = None
        unsqueeze_462: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 3);  unsqueeze_461 = None
        unsqueeze_496: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
        unsqueeze_497: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 2);  unsqueeze_496 = None
        unsqueeze_498: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 3);  unsqueeze_497 = None
        unsqueeze_532: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
        unsqueeze_533: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_532, 2);  unsqueeze_532 = None
        unsqueeze_534: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 3);  unsqueeze_533 = None
        unsqueeze_568: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
        unsqueeze_569: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_568, 2);  unsqueeze_568 = None
        unsqueeze_570: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 3);  unsqueeze_569 = None
        unsqueeze_604: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
        unsqueeze_605: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_604, 2);  unsqueeze_604 = None
        unsqueeze_606: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 3);  unsqueeze_605 = None
        unsqueeze_640: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
        unsqueeze_641: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_640, 2);  unsqueeze_640 = None
        unsqueeze_642: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 3);  unsqueeze_641 = None
        unsqueeze_676: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
        unsqueeze_677: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_676, 2);  unsqueeze_676 = None
        unsqueeze_678: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 3);  unsqueeze_677 = None
        unsqueeze_712: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
        unsqueeze_713: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_712, 2);  unsqueeze_712 = None
        unsqueeze_714: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 3);  unsqueeze_713 = None
        unsqueeze_748: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
        unsqueeze_749: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_748, 2);  unsqueeze_748 = None
        unsqueeze_750: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 3);  unsqueeze_749 = None

        # No stacktrace found for following nodes
        copy_: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_3, add);  primals_3 = add = copy_ = None
        copy__1: "f32[32][1]cuda:0" = torch.ops.aten.copy_.default(primals_4, add_2);  primals_4 = add_2 = copy__1 = None
        copy__2: "f32[32][1]cuda:0" = torch.ops.aten.copy_.default(primals_5, add_3);  primals_5 = add_3 = copy__2 = None
        copy__3: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_9, add_6);  primals_9 = add_6 = copy__3 = None
        copy__4: "f32[32][1]cuda:0" = torch.ops.aten.copy_.default(primals_10, add_8);  primals_10 = add_8 = copy__4 = None
        copy__5: "f32[32][1]cuda:0" = torch.ops.aten.copy_.default(primals_11, add_9);  primals_11 = add_9 = copy__5 = None
        copy__6: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_19, add_13);  primals_19 = add_13 = copy__6 = None
        copy__7: "f32[16][1]cuda:0" = torch.ops.aten.copy_.default(primals_20, add_15);  primals_20 = add_15 = copy__7 = None
        copy__8: "f32[16][1]cuda:0" = torch.ops.aten.copy_.default(primals_21, add_16);  primals_21 = add_16 = copy__8 = None
        copy__9: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_25, add_18);  primals_25 = add_18 = copy__9 = None
        copy__10: "f32[96][1]cuda:0" = torch.ops.aten.copy_.default(primals_26, add_20);  primals_26 = add_20 = copy__10 = None
        copy__11: "f32[96][1]cuda:0" = torch.ops.aten.copy_.default(primals_27, add_21);  primals_27 = add_21 = copy__11 = None
        copy__12: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_31, add_24);  primals_31 = add_24 = copy__12 = None
        copy__13: "f32[96][1]cuda:0" = torch.ops.aten.copy_.default(primals_32, add_26);  primals_32 = add_26 = copy__13 = None
        copy__14: "f32[96][1]cuda:0" = torch.ops.aten.copy_.default(primals_33, add_27);  primals_33 = add_27 = copy__14 = None
        copy__15: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_41, add_31);  primals_41 = add_31 = copy__15 = None
        copy__16: "f32[24][1]cuda:0" = torch.ops.aten.copy_.default(primals_42, add_33);  primals_42 = add_33 = copy__16 = None
        copy__17: "f32[24][1]cuda:0" = torch.ops.aten.copy_.default(primals_43, add_34);  primals_43 = add_34 = copy__17 = None
        copy__18: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_47, add_36);  primals_47 = add_36 = copy__18 = None
        copy__19: "f32[144][1]cuda:0" = torch.ops.aten.copy_.default(primals_48, add_38);  primals_48 = add_38 = copy__19 = None
        copy__20: "f32[144][1]cuda:0" = torch.ops.aten.copy_.default(primals_49, add_39);  primals_49 = add_39 = copy__20 = None
        copy__21: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_53, add_42);  primals_53 = add_42 = copy__21 = None
        copy__22: "f32[144][1]cuda:0" = torch.ops.aten.copy_.default(primals_54, add_44);  primals_54 = add_44 = copy__22 = None
        copy__23: "f32[144][1]cuda:0" = torch.ops.aten.copy_.default(primals_55, add_45);  primals_55 = add_45 = copy__23 = None
        copy__24: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_63, add_49);  primals_63 = add_49 = copy__24 = None
        copy__25: "f32[24][1]cuda:0" = torch.ops.aten.copy_.default(primals_64, add_51);  primals_64 = add_51 = copy__25 = None
        copy__26: "f32[24][1]cuda:0" = torch.ops.aten.copy_.default(primals_65, add_52);  primals_65 = add_52 = copy__26 = None
        copy__27: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_69, add_55);  primals_69 = add_55 = copy__27 = None
        copy__28: "f32[144][1]cuda:0" = torch.ops.aten.copy_.default(primals_70, add_57);  primals_70 = add_57 = copy__28 = None
        copy__29: "f32[144][1]cuda:0" = torch.ops.aten.copy_.default(primals_71, add_58);  primals_71 = add_58 = copy__29 = None
        copy__30: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_75, add_61);  primals_75 = add_61 = copy__30 = None
        copy__31: "f32[144][1]cuda:0" = torch.ops.aten.copy_.default(primals_76, add_63);  primals_76 = add_63 = copy__31 = None
        copy__32: "f32[144][1]cuda:0" = torch.ops.aten.copy_.default(primals_77, add_64);  primals_77 = add_64 = copy__32 = None
        copy__33: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_85, add_68);  primals_85 = add_68 = copy__33 = None
        copy__34: "f32[40][1]cuda:0" = torch.ops.aten.copy_.default(primals_86, add_70);  primals_86 = add_70 = copy__34 = None
        copy__35: "f32[40][1]cuda:0" = torch.ops.aten.copy_.default(primals_87, add_71);  primals_87 = add_71 = copy__35 = None
        copy__36: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_91, add_73);  primals_91 = add_73 = copy__36 = None
        copy__37: "f32[240][1]cuda:0" = torch.ops.aten.copy_.default(primals_92, add_75);  primals_92 = add_75 = copy__37 = None
        copy__38: "f32[240][1]cuda:0" = torch.ops.aten.copy_.default(primals_93, add_76);  primals_93 = add_76 = copy__38 = None
        copy__39: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_97, add_79);  primals_97 = add_79 = copy__39 = None
        copy__40: "f32[240][1]cuda:0" = torch.ops.aten.copy_.default(primals_98, add_81);  primals_98 = add_81 = copy__40 = None
        copy__41: "f32[240][1]cuda:0" = torch.ops.aten.copy_.default(primals_99, add_82);  primals_99 = add_82 = copy__41 = None
        copy__42: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_107, add_86);  primals_107 = add_86 = copy__42 = None
        copy__43: "f32[40][1]cuda:0" = torch.ops.aten.copy_.default(primals_108, add_88);  primals_108 = add_88 = copy__43 = None
        copy__44: "f32[40][1]cuda:0" = torch.ops.aten.copy_.default(primals_109, add_89);  primals_109 = add_89 = copy__44 = None
        copy__45: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_113, add_92);  primals_113 = add_92 = copy__45 = None
        copy__46: "f32[240][1]cuda:0" = torch.ops.aten.copy_.default(primals_114, add_94);  primals_114 = add_94 = copy__46 = None
        copy__47: "f32[240][1]cuda:0" = torch.ops.aten.copy_.default(primals_115, add_95);  primals_115 = add_95 = copy__47 = None
        copy__48: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_119, add_98);  primals_119 = add_98 = copy__48 = None
        copy__49: "f32[240][1]cuda:0" = torch.ops.aten.copy_.default(primals_120, add_100);  primals_120 = add_100 = copy__49 = None
        copy__50: "f32[240][1]cuda:0" = torch.ops.aten.copy_.default(primals_121, add_101);  primals_121 = add_101 = copy__50 = None
        copy__51: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_129, add_105);  primals_129 = add_105 = copy__51 = None
        copy__52: "f32[80][1]cuda:0" = torch.ops.aten.copy_.default(primals_130, add_107);  primals_130 = add_107 = copy__52 = None
        copy__53: "f32[80][1]cuda:0" = torch.ops.aten.copy_.default(primals_131, add_108);  primals_131 = add_108 = copy__53 = None
        copy__54: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_135, add_110);  primals_135 = add_110 = copy__54 = None
        copy__55: "f32[480][1]cuda:0" = torch.ops.aten.copy_.default(primals_136, add_112);  primals_136 = add_112 = copy__55 = None
        copy__56: "f32[480][1]cuda:0" = torch.ops.aten.copy_.default(primals_137, add_113);  primals_137 = add_113 = copy__56 = None
        copy__57: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_141, add_116);  primals_141 = add_116 = copy__57 = None
        copy__58: "f32[480][1]cuda:0" = torch.ops.aten.copy_.default(primals_142, add_118);  primals_142 = add_118 = copy__58 = None
        copy__59: "f32[480][1]cuda:0" = torch.ops.aten.copy_.default(primals_143, add_119);  primals_143 = add_119 = copy__59 = None
        copy__60: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_151, add_123);  primals_151 = add_123 = copy__60 = None
        copy__61: "f32[80][1]cuda:0" = torch.ops.aten.copy_.default(primals_152, add_125);  primals_152 = add_125 = copy__61 = None
        copy__62: "f32[80][1]cuda:0" = torch.ops.aten.copy_.default(primals_153, add_126);  primals_153 = add_126 = copy__62 = None
        copy__63: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_157, add_129);  primals_157 = add_129 = copy__63 = None
        copy__64: "f32[480][1]cuda:0" = torch.ops.aten.copy_.default(primals_158, add_131);  primals_158 = add_131 = copy__64 = None
        copy__65: "f32[480][1]cuda:0" = torch.ops.aten.copy_.default(primals_159, add_132);  primals_159 = add_132 = copy__65 = None
        copy__66: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_163, add_135);  primals_163 = add_135 = copy__66 = None
        copy__67: "f32[480][1]cuda:0" = torch.ops.aten.copy_.default(primals_164, add_137);  primals_164 = add_137 = copy__67 = None
        copy__68: "f32[480][1]cuda:0" = torch.ops.aten.copy_.default(primals_165, add_138);  primals_165 = add_138 = copy__68 = None
        copy__69: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_173, add_142);  primals_173 = add_142 = copy__69 = None
        copy__70: "f32[80][1]cuda:0" = torch.ops.aten.copy_.default(primals_174, add_144);  primals_174 = add_144 = copy__70 = None
        copy__71: "f32[80][1]cuda:0" = torch.ops.aten.copy_.default(primals_175, add_145);  primals_175 = add_145 = copy__71 = None
        copy__72: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_179, add_148);  primals_179 = add_148 = copy__72 = None
        copy__73: "f32[480][1]cuda:0" = torch.ops.aten.copy_.default(primals_180, add_150);  primals_180 = add_150 = copy__73 = None
        copy__74: "f32[480][1]cuda:0" = torch.ops.aten.copy_.default(primals_181, add_151);  primals_181 = add_151 = copy__74 = None
        copy__75: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_185, add_154);  primals_185 = add_154 = copy__75 = None
        copy__76: "f32[480][1]cuda:0" = torch.ops.aten.copy_.default(primals_186, add_156);  primals_186 = add_156 = copy__76 = None
        copy__77: "f32[480][1]cuda:0" = torch.ops.aten.copy_.default(primals_187, add_157);  primals_187 = add_157 = copy__77 = None
        copy__78: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_195, add_161);  primals_195 = add_161 = copy__78 = None
        copy__79: "f32[112][1]cuda:0" = torch.ops.aten.copy_.default(primals_196, add_163);  primals_196 = add_163 = copy__79 = None
        copy__80: "f32[112][1]cuda:0" = torch.ops.aten.copy_.default(primals_197, add_164);  primals_197 = add_164 = copy__80 = None
        copy__81: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_201, add_166);  primals_201 = add_166 = copy__81 = None
        copy__82: "f32[672][1]cuda:0" = torch.ops.aten.copy_.default(primals_202, add_168);  primals_202 = add_168 = copy__82 = None
        copy__83: "f32[672][1]cuda:0" = torch.ops.aten.copy_.default(primals_203, add_169);  primals_203 = add_169 = copy__83 = None
        copy__84: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_207, add_172);  primals_207 = add_172 = copy__84 = None
        copy__85: "f32[672][1]cuda:0" = torch.ops.aten.copy_.default(primals_208, add_174);  primals_208 = add_174 = copy__85 = None
        copy__86: "f32[672][1]cuda:0" = torch.ops.aten.copy_.default(primals_209, add_175);  primals_209 = add_175 = copy__86 = None
        copy__87: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_217, add_179);  primals_217 = add_179 = copy__87 = None
        copy__88: "f32[112][1]cuda:0" = torch.ops.aten.copy_.default(primals_218, add_181);  primals_218 = add_181 = copy__88 = None
        copy__89: "f32[112][1]cuda:0" = torch.ops.aten.copy_.default(primals_219, add_182);  primals_219 = add_182 = copy__89 = None
        copy__90: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_223, add_185);  primals_223 = add_185 = copy__90 = None
        copy__91: "f32[672][1]cuda:0" = torch.ops.aten.copy_.default(primals_224, add_187);  primals_224 = add_187 = copy__91 = None
        copy__92: "f32[672][1]cuda:0" = torch.ops.aten.copy_.default(primals_225, add_188);  primals_225 = add_188 = copy__92 = None
        copy__93: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_229, add_191);  primals_229 = add_191 = copy__93 = None
        copy__94: "f32[672][1]cuda:0" = torch.ops.aten.copy_.default(primals_230, add_193);  primals_230 = add_193 = copy__94 = None
        copy__95: "f32[672][1]cuda:0" = torch.ops.aten.copy_.default(primals_231, add_194);  primals_231 = add_194 = copy__95 = None
        copy__96: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_239, add_198);  primals_239 = add_198 = copy__96 = None
        copy__97: "f32[112][1]cuda:0" = torch.ops.aten.copy_.default(primals_240, add_200);  primals_240 = add_200 = copy__97 = None
        copy__98: "f32[112][1]cuda:0" = torch.ops.aten.copy_.default(primals_241, add_201);  primals_241 = add_201 = copy__98 = None
        copy__99: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_245, add_204);  primals_245 = add_204 = copy__99 = None
        copy__100: "f32[672][1]cuda:0" = torch.ops.aten.copy_.default(primals_246, add_206);  primals_246 = add_206 = copy__100 = None
        copy__101: "f32[672][1]cuda:0" = torch.ops.aten.copy_.default(primals_247, add_207);  primals_247 = add_207 = copy__101 = None
        copy__102: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_251, add_210);  primals_251 = add_210 = copy__102 = None
        copy__103: "f32[672][1]cuda:0" = torch.ops.aten.copy_.default(primals_252, add_212);  primals_252 = add_212 = copy__103 = None
        copy__104: "f32[672][1]cuda:0" = torch.ops.aten.copy_.default(primals_253, add_213);  primals_253 = add_213 = copy__104 = None
        copy__105: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_261, add_217);  primals_261 = add_217 = copy__105 = None
        copy__106: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_262, add_219);  primals_262 = add_219 = copy__106 = None
        copy__107: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_263, add_220);  primals_263 = add_220 = copy__107 = None
        copy__108: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_267, add_222);  primals_267 = add_222 = copy__108 = None
        copy__109: "f32[1152][1]cuda:0" = torch.ops.aten.copy_.default(primals_268, add_224);  primals_268 = add_224 = copy__109 = None
        copy__110: "f32[1152][1]cuda:0" = torch.ops.aten.copy_.default(primals_269, add_225);  primals_269 = add_225 = copy__110 = None
        copy__111: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_273, add_228);  primals_273 = add_228 = copy__111 = None
        copy__112: "f32[1152][1]cuda:0" = torch.ops.aten.copy_.default(primals_274, add_230);  primals_274 = add_230 = copy__112 = None
        copy__113: "f32[1152][1]cuda:0" = torch.ops.aten.copy_.default(primals_275, add_231);  primals_275 = add_231 = copy__113 = None
        copy__114: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_283, add_235);  primals_283 = add_235 = copy__114 = None
        copy__115: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_284, add_237);  primals_284 = add_237 = copy__115 = None
        copy__116: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_285, add_238);  primals_285 = add_238 = copy__116 = None
        copy__117: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_289, add_241);  primals_289 = add_241 = copy__117 = None
        copy__118: "f32[1152][1]cuda:0" = torch.ops.aten.copy_.default(primals_290, add_243);  primals_290 = add_243 = copy__118 = None
        copy__119: "f32[1152][1]cuda:0" = torch.ops.aten.copy_.default(primals_291, add_244);  primals_291 = add_244 = copy__119 = None
        copy__120: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_295, add_247);  primals_295 = add_247 = copy__120 = None
        copy__121: "f32[1152][1]cuda:0" = torch.ops.aten.copy_.default(primals_296, add_249);  primals_296 = add_249 = copy__121 = None
        copy__122: "f32[1152][1]cuda:0" = torch.ops.aten.copy_.default(primals_297, add_250);  primals_297 = add_250 = copy__122 = None
        copy__123: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_305, add_254);  primals_305 = add_254 = copy__123 = None
        copy__124: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_306, add_256);  primals_306 = add_256 = copy__124 = None
        copy__125: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_307, add_257);  primals_307 = add_257 = copy__125 = None
        copy__126: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_311, add_260);  primals_311 = add_260 = copy__126 = None
        copy__127: "f32[1152][1]cuda:0" = torch.ops.aten.copy_.default(primals_312, add_262);  primals_312 = add_262 = copy__127 = None
        copy__128: "f32[1152][1]cuda:0" = torch.ops.aten.copy_.default(primals_313, add_263);  primals_313 = add_263 = copy__128 = None
        copy__129: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_317, add_266);  primals_317 = add_266 = copy__129 = None
        copy__130: "f32[1152][1]cuda:0" = torch.ops.aten.copy_.default(primals_318, add_268);  primals_318 = add_268 = copy__130 = None
        copy__131: "f32[1152][1]cuda:0" = torch.ops.aten.copy_.default(primals_319, add_269);  primals_319 = add_269 = copy__131 = None
        copy__132: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_327, add_273);  primals_327 = add_273 = copy__132 = None
        copy__133: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_328, add_275);  primals_328 = add_275 = copy__133 = None
        copy__134: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_329, add_276);  primals_329 = add_276 = copy__134 = None
        copy__135: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_333, add_279);  primals_333 = add_279 = copy__135 = None
        copy__136: "f32[1152][1]cuda:0" = torch.ops.aten.copy_.default(primals_334, add_281);  primals_334 = add_281 = copy__136 = None
        copy__137: "f32[1152][1]cuda:0" = torch.ops.aten.copy_.default(primals_335, add_282);  primals_335 = add_282 = copy__137 = None
        copy__138: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_339, add_285);  primals_339 = add_285 = copy__138 = None
        copy__139: "f32[1152][1]cuda:0" = torch.ops.aten.copy_.default(primals_340, add_287);  primals_340 = add_287 = copy__139 = None
        copy__140: "f32[1152][1]cuda:0" = torch.ops.aten.copy_.default(primals_341, add_288);  primals_341 = add_288 = copy__140 = None
        copy__141: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_349, add_292);  primals_349 = add_292 = copy__141 = None
        copy__142: "f32[320][1]cuda:0" = torch.ops.aten.copy_.default(primals_350, add_294);  primals_350 = add_294 = copy__142 = None
        copy__143: "f32[320][1]cuda:0" = torch.ops.aten.copy_.default(primals_351, add_295);  primals_351 = add_295 = copy__143 = None
        copy__144: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_355, add_297);  primals_355 = add_297 = copy__144 = None
        copy__145: "f32[1280][1]cuda:0" = torch.ops.aten.copy_.default(primals_356, add_299);  primals_356 = add_299 = copy__145 = None
        copy__146: "f32[1280][1]cuda:0" = torch.ops.aten.copy_.default(primals_357, add_300);  primals_357 = add_300 = copy__146 = None
        return (addmm, primals_6, primals_7, primals_12, primals_13, primals_22, primals_28, primals_29, primals_34, primals_35, primals_44, primals_50, primals_51, primals_56, primals_57, primals_66, primals_72, primals_73, primals_78, primals_79, primals_88, primals_94, primals_95, primals_100, primals_101, primals_110, primals_116, primals_117, primals_122, primals_123, primals_132, primals_138, primals_139, primals_144, primals_145, primals_154, primals_160, primals_161, primals_166, primals_167, primals_176, primals_182, primals_183, primals_188, primals_189, primals_198, primals_204, primals_205, primals_210, primals_211, primals_220, primals_226, primals_227, primals_232, primals_233, primals_242, primals_248, primals_249, primals_254, primals_255, primals_264, primals_270, primals_271, primals_276, primals_277, primals_286, primals_292, primals_293, primals_298, primals_299, primals_308, primals_314, primals_315, primals_320, primals_321, primals_330, primals_336, primals_337, primals_342, primals_343, primals_352, primals_358, primals_359, convert_element_type, convert_element_type_1, convolution, getitem_1, rsqrt, convert_element_type_5, convert_element_type_6, convolution_1, getitem_3, rsqrt_1, mean, convert_element_type_12, convolution_2, convert_element_type_14, convert_element_type_16, convolution_3, mul_14, convert_element_type_17, convolution_4, squeeze_7, convert_element_type_19, convert_element_type_20, convolution_5, getitem_7, rsqrt_3, constant_pad_nd_1, convert_element_type_25, convolution_6, getitem_9, rsqrt_4, mean_1, convert_element_type_31, convolution_7, convert_element_type_33, convert_element_type_35, convolution_8, mul_36, convert_element_type_36, convolution_9, squeeze_16, convert_element_type_38, convert_element_type_39, convolution_10, getitem_13, rsqrt_6, convert_element_type_43, convert_element_type_44, convolution_11, getitem_15, rsqrt_7, mean_2, convert_element_type_50, convolution_12, convert_element_type_52, convert_element_type_54, convolution_13, mul_58, convert_element_type_55, convolution_14, squeeze_25, add_54, convert_element_type_58, convolution_15, getitem_19, rsqrt_9, constant_pad_nd_2, convert_element_type_63, convolution_16, getitem_21, rsqrt_10, mean_3, convert_element_type_69, convolution_17, convert_element_type_71, convert_element_type_73, convolution_18, mul_80, convert_element_type_74, convolution_19, squeeze_34, convert_element_type_76, convert_element_type_77, convolution_20, getitem_25, rsqrt_12, convert_element_type_81, convert_element_type_82, convolution_21, getitem_27, rsqrt_13, mean_4, convert_element_type_88, convolution_22, convert_element_type_90, convert_element_type_92, convolution_23, mul_102, convert_element_type_93, convolution_24, squeeze_43, add_91, convert_element_type_96, convolution_25, getitem_31, rsqrt_15, constant_pad_nd_3, convert_element_type_101, convolution_26, getitem_33, rsqrt_16, mean_5, convert_element_type_107, convolution_27, convert_element_type_109, convert_element_type_111, convolution_28, mul_124, convert_element_type_112, convolution_29, squeeze_52, convert_element_type_114, convert_element_type_115, convolution_30, getitem_37, rsqrt_18, convert_element_type_119, convert_element_type_120, convolution_31, getitem_39, rsqrt_19, mean_6, convert_element_type_126, convolution_32, convert_element_type_128, convert_element_type_130, convolution_33, mul_146, convert_element_type_131, convolution_34, squeeze_61, add_128, convert_element_type_134, convolution_35, getitem_43, rsqrt_21, convert_element_type_138, convert_element_type_139, convolution_36, getitem_45, rsqrt_22, mean_7, convert_element_type_145, convolution_37, convert_element_type_147, convert_element_type_149, convolution_38, mul_168, convert_element_type_150, convolution_39, squeeze_70, add_147, convert_element_type_153, convolution_40, getitem_49, rsqrt_24, convert_element_type_157, convert_element_type_158, convolution_41, getitem_51, rsqrt_25, mean_8, convert_element_type_164, convolution_42, convert_element_type_166, convert_element_type_168, convolution_43, mul_190, convert_element_type_169, convolution_44, squeeze_79, convert_element_type_171, convert_element_type_172, convolution_45, getitem_55, rsqrt_27, convert_element_type_176, convert_element_type_177, convolution_46, getitem_57, rsqrt_28, mean_9, convert_element_type_183, convolution_47, convert_element_type_185, convert_element_type_187, convolution_48, mul_212, convert_element_type_188, convolution_49, squeeze_88, add_184, convert_element_type_191, convolution_50, getitem_61, rsqrt_30, convert_element_type_195, convert_element_type_196, convolution_51, getitem_63, rsqrt_31, mean_10, convert_element_type_202, convolution_52, convert_element_type_204, convert_element_type_206, convolution_53, mul_234, convert_element_type_207, convolution_54, squeeze_97, add_203, convert_element_type_210, convolution_55, getitem_67, rsqrt_33, constant_pad_nd_4, convert_element_type_215, convolution_56, getitem_69, rsqrt_34, mean_11, convert_element_type_221, convolution_57, convert_element_type_223, convert_element_type_225, convolution_58, mul_256, convert_element_type_226, convolution_59, squeeze_106, convert_element_type_228, convert_element_type_229, convolution_60, getitem_73, rsqrt_36, convert_element_type_233, convert_element_type_234, convolution_61, getitem_75, rsqrt_37, mean_12, convert_element_type_240, convolution_62, convert_element_type_242, convert_element_type_244, convolution_63, mul_278, convert_element_type_245, convolution_64, squeeze_115, add_240, convert_element_type_248, convolution_65, getitem_79, rsqrt_39, convert_element_type_252, convert_element_type_253, convolution_66, getitem_81, rsqrt_40, mean_13, convert_element_type_259, convolution_67, convert_element_type_261, convert_element_type_263, convolution_68, mul_300, convert_element_type_264, convolution_69, squeeze_124, add_259, convert_element_type_267, convolution_70, getitem_85, rsqrt_42, convert_element_type_271, convert_element_type_272, convolution_71, getitem_87, rsqrt_43, mean_14, convert_element_type_278, convolution_72, convert_element_type_280, convert_element_type_282, convolution_73, mul_322, convert_element_type_283, convolution_74, squeeze_133, add_278, convert_element_type_286, convolution_75, getitem_91, rsqrt_45, convert_element_type_290, convert_element_type_291, convolution_76, getitem_93, rsqrt_46, mean_15, convert_element_type_297, convolution_77, convert_element_type_299, convert_element_type_301, convolution_78, mul_344, convert_element_type_302, convolution_79, squeeze_142, convert_element_type_304, convert_element_type_305, convolution_80, getitem_97, rsqrt_48, view, permute_1, unsqueeze_210, unsqueeze_246, unsqueeze_282, unsqueeze_318, unsqueeze_354, unsqueeze_390, unsqueeze_426, unsqueeze_462, unsqueeze_498, unsqueeze_534, unsqueeze_570, unsqueeze_606, unsqueeze_642, unsqueeze_678, unsqueeze_714, unsqueeze_750)
