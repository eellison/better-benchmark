class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[128, 3, 224, 224][150528, 1, 672, 3]cuda:0", primals_2: "f32[128, 3, 4, 4][48, 1, 12, 3]cuda:0", primals_3: "f32[128][1]cuda:0", primals_4: "f32[128][1]cuda:0", primals_5: "f32[128][1]cuda:0", primals_6: "f32[128][1]cuda:0", primals_7: "f32[128][1]cuda:0", primals_8: "f32[384, 128][128, 1]cuda:0", primals_9: "f32[384][1]cuda:0", primals_10: "f32[169, 4][4, 1]cuda:0", primals_11: "i64[49, 49][49, 1]cuda:0", primals_12: "f32[128, 128][128, 1]cuda:0", primals_13: "f32[128][1]cuda:0", primals_14: "f32[128][1]cuda:0", primals_15: "f32[128][1]cuda:0", primals_16: "f32[512, 128][128, 1]cuda:0", primals_17: "f32[512][1]cuda:0", primals_18: "f32[128, 512][512, 1]cuda:0", primals_19: "f32[128][1]cuda:0", primals_20: "f32[128][1]cuda:0", primals_21: "f32[128][1]cuda:0", primals_22: "f32[64, 49, 49][2401, 49, 1]cuda:0", primals_23: "f32[384, 128][128, 1]cuda:0", primals_24: "f32[384][1]cuda:0", primals_25: "f32[169, 4][4, 1]cuda:0", primals_26: "i64[49, 49][49, 1]cuda:0", primals_27: "f32[128, 128][128, 1]cuda:0", primals_28: "f32[128][1]cuda:0", primals_29: "f32[128][1]cuda:0", primals_30: "f32[128][1]cuda:0", primals_31: "f32[512, 128][128, 1]cuda:0", primals_32: "f32[512][1]cuda:0", primals_33: "f32[128, 512][512, 1]cuda:0", primals_34: "f32[128][1]cuda:0", primals_35: "f32[512][1]cuda:0", primals_36: "f32[512][1]cuda:0", primals_37: "f32[256, 512][512, 1]cuda:0", primals_38: "f32[256][1]cuda:0", primals_39: "f32[256][1]cuda:0", primals_40: "f32[768, 256][256, 1]cuda:0", primals_41: "f32[768][1]cuda:0", primals_42: "f32[169, 8][8, 1]cuda:0", primals_43: "i64[49, 49][49, 1]cuda:0", primals_44: "f32[256, 256][256, 1]cuda:0", primals_45: "f32[256][1]cuda:0", primals_46: "f32[256][1]cuda:0", primals_47: "f32[256][1]cuda:0", primals_48: "f32[1024, 256][256, 1]cuda:0", primals_49: "f32[1024][1]cuda:0", primals_50: "f32[256, 1024][1024, 1]cuda:0", primals_51: "f32[256][1]cuda:0", primals_52: "f32[256][1]cuda:0", primals_53: "f32[256][1]cuda:0", primals_54: "f32[16, 49, 49][2401, 49, 1]cuda:0", primals_55: "f32[768, 256][256, 1]cuda:0", primals_56: "f32[768][1]cuda:0", primals_57: "f32[169, 8][8, 1]cuda:0", primals_58: "i64[49, 49][49, 1]cuda:0", primals_59: "f32[256, 256][256, 1]cuda:0", primals_60: "f32[256][1]cuda:0", primals_61: "f32[256][1]cuda:0", primals_62: "f32[256][1]cuda:0", primals_63: "f32[1024, 256][256, 1]cuda:0", primals_64: "f32[1024][1]cuda:0", primals_65: "f32[256, 1024][1024, 1]cuda:0", primals_66: "f32[256][1]cuda:0", primals_67: "f32[1024][1]cuda:0", primals_68: "f32[1024][1]cuda:0", primals_69: "f32[512, 1024][1024, 1]cuda:0", primals_70: "f32[512][1]cuda:0", primals_71: "f32[512][1]cuda:0", primals_72: "f32[1536, 512][512, 1]cuda:0", primals_73: "f32[1536][1]cuda:0", primals_74: "f32[169, 16][16, 1]cuda:0", primals_75: "i64[49, 49][49, 1]cuda:0", primals_76: "f32[512, 512][512, 1]cuda:0", primals_77: "f32[512][1]cuda:0", primals_78: "f32[512][1]cuda:0", primals_79: "f32[512][1]cuda:0", primals_80: "f32[2048, 512][512, 1]cuda:0", primals_81: "f32[2048][1]cuda:0", primals_82: "f32[512, 2048][2048, 1]cuda:0", primals_83: "f32[512][1]cuda:0", primals_84: "f32[512][1]cuda:0", primals_85: "f32[512][1]cuda:0", primals_86: "f32[4, 49, 49][2401, 49, 1]cuda:0", primals_87: "f32[1536, 512][512, 1]cuda:0", primals_88: "f32[1536][1]cuda:0", primals_89: "f32[169, 16][16, 1]cuda:0", primals_90: "i64[49, 49][49, 1]cuda:0", primals_91: "f32[512, 512][512, 1]cuda:0", primals_92: "f32[512][1]cuda:0", primals_93: "f32[512][1]cuda:0", primals_94: "f32[512][1]cuda:0", primals_95: "f32[2048, 512][512, 1]cuda:0", primals_96: "f32[2048][1]cuda:0", primals_97: "f32[512, 2048][2048, 1]cuda:0", primals_98: "f32[512][1]cuda:0", primals_99: "f32[512][1]cuda:0", primals_100: "f32[512][1]cuda:0", primals_101: "f32[1536, 512][512, 1]cuda:0", primals_102: "f32[1536][1]cuda:0", primals_103: "f32[169, 16][16, 1]cuda:0", primals_104: "i64[49, 49][49, 1]cuda:0", primals_105: "f32[512, 512][512, 1]cuda:0", primals_106: "f32[512][1]cuda:0", primals_107: "f32[512][1]cuda:0", primals_108: "f32[512][1]cuda:0", primals_109: "f32[2048, 512][512, 1]cuda:0", primals_110: "f32[2048][1]cuda:0", primals_111: "f32[512, 2048][2048, 1]cuda:0", primals_112: "f32[512][1]cuda:0", primals_113: "f32[512][1]cuda:0", primals_114: "f32[512][1]cuda:0", primals_115: "f32[4, 49, 49][2401, 49, 1]cuda:0", primals_116: "f32[1536, 512][512, 1]cuda:0", primals_117: "f32[1536][1]cuda:0", primals_118: "f32[169, 16][16, 1]cuda:0", primals_119: "i64[49, 49][49, 1]cuda:0", primals_120: "f32[512, 512][512, 1]cuda:0", primals_121: "f32[512][1]cuda:0", primals_122: "f32[512][1]cuda:0", primals_123: "f32[512][1]cuda:0", primals_124: "f32[2048, 512][512, 1]cuda:0", primals_125: "f32[2048][1]cuda:0", primals_126: "f32[512, 2048][2048, 1]cuda:0", primals_127: "f32[512][1]cuda:0", primals_128: "f32[512][1]cuda:0", primals_129: "f32[512][1]cuda:0", primals_130: "f32[1536, 512][512, 1]cuda:0", primals_131: "f32[1536][1]cuda:0", primals_132: "f32[169, 16][16, 1]cuda:0", primals_133: "i64[49, 49][49, 1]cuda:0", primals_134: "f32[512, 512][512, 1]cuda:0", primals_135: "f32[512][1]cuda:0", primals_136: "f32[512][1]cuda:0", primals_137: "f32[512][1]cuda:0", primals_138: "f32[2048, 512][512, 1]cuda:0", primals_139: "f32[2048][1]cuda:0", primals_140: "f32[512, 2048][2048, 1]cuda:0", primals_141: "f32[512][1]cuda:0", primals_142: "f32[512][1]cuda:0", primals_143: "f32[512][1]cuda:0", primals_144: "f32[4, 49, 49][2401, 49, 1]cuda:0", primals_145: "f32[1536, 512][512, 1]cuda:0", primals_146: "f32[1536][1]cuda:0", primals_147: "f32[169, 16][16, 1]cuda:0", primals_148: "i64[49, 49][49, 1]cuda:0", primals_149: "f32[512, 512][512, 1]cuda:0", primals_150: "f32[512][1]cuda:0", primals_151: "f32[512][1]cuda:0", primals_152: "f32[512][1]cuda:0", primals_153: "f32[2048, 512][512, 1]cuda:0", primals_154: "f32[2048][1]cuda:0", primals_155: "f32[512, 2048][2048, 1]cuda:0", primals_156: "f32[512][1]cuda:0", primals_157: "f32[512][1]cuda:0", primals_158: "f32[512][1]cuda:0", primals_159: "f32[1536, 512][512, 1]cuda:0", primals_160: "f32[1536][1]cuda:0", primals_161: "f32[169, 16][16, 1]cuda:0", primals_162: "i64[49, 49][49, 1]cuda:0", primals_163: "f32[512, 512][512, 1]cuda:0", primals_164: "f32[512][1]cuda:0", primals_165: "f32[512][1]cuda:0", primals_166: "f32[512][1]cuda:0", primals_167: "f32[2048, 512][512, 1]cuda:0", primals_168: "f32[2048][1]cuda:0", primals_169: "f32[512, 2048][2048, 1]cuda:0", primals_170: "f32[512][1]cuda:0", primals_171: "f32[512][1]cuda:0", primals_172: "f32[512][1]cuda:0", primals_173: "f32[4, 49, 49][2401, 49, 1]cuda:0", primals_174: "f32[1536, 512][512, 1]cuda:0", primals_175: "f32[1536][1]cuda:0", primals_176: "f32[169, 16][16, 1]cuda:0", primals_177: "i64[49, 49][49, 1]cuda:0", primals_178: "f32[512, 512][512, 1]cuda:0", primals_179: "f32[512][1]cuda:0", primals_180: "f32[512][1]cuda:0", primals_181: "f32[512][1]cuda:0", primals_182: "f32[2048, 512][512, 1]cuda:0", primals_183: "f32[2048][1]cuda:0", primals_184: "f32[512, 2048][2048, 1]cuda:0", primals_185: "f32[512][1]cuda:0", primals_186: "f32[512][1]cuda:0", primals_187: "f32[512][1]cuda:0", primals_188: "f32[1536, 512][512, 1]cuda:0", primals_189: "f32[1536][1]cuda:0", primals_190: "f32[169, 16][16, 1]cuda:0", primals_191: "i64[49, 49][49, 1]cuda:0", primals_192: "f32[512, 512][512, 1]cuda:0", primals_193: "f32[512][1]cuda:0", primals_194: "f32[512][1]cuda:0", primals_195: "f32[512][1]cuda:0", primals_196: "f32[2048, 512][512, 1]cuda:0", primals_197: "f32[2048][1]cuda:0", primals_198: "f32[512, 2048][2048, 1]cuda:0", primals_199: "f32[512][1]cuda:0", primals_200: "f32[512][1]cuda:0", primals_201: "f32[512][1]cuda:0", primals_202: "f32[4, 49, 49][2401, 49, 1]cuda:0", primals_203: "f32[1536, 512][512, 1]cuda:0", primals_204: "f32[1536][1]cuda:0", primals_205: "f32[169, 16][16, 1]cuda:0", primals_206: "i64[49, 49][49, 1]cuda:0", primals_207: "f32[512, 512][512, 1]cuda:0", primals_208: "f32[512][1]cuda:0", primals_209: "f32[512][1]cuda:0", primals_210: "f32[512][1]cuda:0", primals_211: "f32[2048, 512][512, 1]cuda:0", primals_212: "f32[2048][1]cuda:0", primals_213: "f32[512, 2048][2048, 1]cuda:0", primals_214: "f32[512][1]cuda:0", primals_215: "f32[512][1]cuda:0", primals_216: "f32[512][1]cuda:0", primals_217: "f32[1536, 512][512, 1]cuda:0", primals_218: "f32[1536][1]cuda:0", primals_219: "f32[169, 16][16, 1]cuda:0", primals_220: "i64[49, 49][49, 1]cuda:0", primals_221: "f32[512, 512][512, 1]cuda:0", primals_222: "f32[512][1]cuda:0", primals_223: "f32[512][1]cuda:0", primals_224: "f32[512][1]cuda:0", primals_225: "f32[2048, 512][512, 1]cuda:0", primals_226: "f32[2048][1]cuda:0", primals_227: "f32[512, 2048][2048, 1]cuda:0", primals_228: "f32[512][1]cuda:0", primals_229: "f32[512][1]cuda:0", primals_230: "f32[512][1]cuda:0", primals_231: "f32[4, 49, 49][2401, 49, 1]cuda:0", primals_232: "f32[1536, 512][512, 1]cuda:0", primals_233: "f32[1536][1]cuda:0", primals_234: "f32[169, 16][16, 1]cuda:0", primals_235: "i64[49, 49][49, 1]cuda:0", primals_236: "f32[512, 512][512, 1]cuda:0", primals_237: "f32[512][1]cuda:0", primals_238: "f32[512][1]cuda:0", primals_239: "f32[512][1]cuda:0", primals_240: "f32[2048, 512][512, 1]cuda:0", primals_241: "f32[2048][1]cuda:0", primals_242: "f32[512, 2048][2048, 1]cuda:0", primals_243: "f32[512][1]cuda:0", primals_244: "f32[512][1]cuda:0", primals_245: "f32[512][1]cuda:0", primals_246: "f32[1536, 512][512, 1]cuda:0", primals_247: "f32[1536][1]cuda:0", primals_248: "f32[169, 16][16, 1]cuda:0", primals_249: "i64[49, 49][49, 1]cuda:0", primals_250: "f32[512, 512][512, 1]cuda:0", primals_251: "f32[512][1]cuda:0", primals_252: "f32[512][1]cuda:0", primals_253: "f32[512][1]cuda:0", primals_254: "f32[2048, 512][512, 1]cuda:0", primals_255: "f32[2048][1]cuda:0", primals_256: "f32[512, 2048][2048, 1]cuda:0", primals_257: "f32[512][1]cuda:0", primals_258: "f32[512][1]cuda:0", primals_259: "f32[512][1]cuda:0", primals_260: "f32[4, 49, 49][2401, 49, 1]cuda:0", primals_261: "f32[1536, 512][512, 1]cuda:0", primals_262: "f32[1536][1]cuda:0", primals_263: "f32[169, 16][16, 1]cuda:0", primals_264: "i64[49, 49][49, 1]cuda:0", primals_265: "f32[512, 512][512, 1]cuda:0", primals_266: "f32[512][1]cuda:0", primals_267: "f32[512][1]cuda:0", primals_268: "f32[512][1]cuda:0", primals_269: "f32[2048, 512][512, 1]cuda:0", primals_270: "f32[2048][1]cuda:0", primals_271: "f32[512, 2048][2048, 1]cuda:0", primals_272: "f32[512][1]cuda:0", primals_273: "f32[512][1]cuda:0", primals_274: "f32[512][1]cuda:0", primals_275: "f32[1536, 512][512, 1]cuda:0", primals_276: "f32[1536][1]cuda:0", primals_277: "f32[169, 16][16, 1]cuda:0", primals_278: "i64[49, 49][49, 1]cuda:0", primals_279: "f32[512, 512][512, 1]cuda:0", primals_280: "f32[512][1]cuda:0", primals_281: "f32[512][1]cuda:0", primals_282: "f32[512][1]cuda:0", primals_283: "f32[2048, 512][512, 1]cuda:0", primals_284: "f32[2048][1]cuda:0", primals_285: "f32[512, 2048][2048, 1]cuda:0", primals_286: "f32[512][1]cuda:0", primals_287: "f32[512][1]cuda:0", primals_288: "f32[512][1]cuda:0", primals_289: "f32[4, 49, 49][2401, 49, 1]cuda:0", primals_290: "f32[1536, 512][512, 1]cuda:0", primals_291: "f32[1536][1]cuda:0", primals_292: "f32[169, 16][16, 1]cuda:0", primals_293: "i64[49, 49][49, 1]cuda:0", primals_294: "f32[512, 512][512, 1]cuda:0", primals_295: "f32[512][1]cuda:0", primals_296: "f32[512][1]cuda:0", primals_297: "f32[512][1]cuda:0", primals_298: "f32[2048, 512][512, 1]cuda:0", primals_299: "f32[2048][1]cuda:0", primals_300: "f32[512, 2048][2048, 1]cuda:0", primals_301: "f32[512][1]cuda:0", primals_302: "f32[512][1]cuda:0", primals_303: "f32[512][1]cuda:0", primals_304: "f32[1536, 512][512, 1]cuda:0", primals_305: "f32[1536][1]cuda:0", primals_306: "f32[169, 16][16, 1]cuda:0", primals_307: "i64[49, 49][49, 1]cuda:0", primals_308: "f32[512, 512][512, 1]cuda:0", primals_309: "f32[512][1]cuda:0", primals_310: "f32[512][1]cuda:0", primals_311: "f32[512][1]cuda:0", primals_312: "f32[2048, 512][512, 1]cuda:0", primals_313: "f32[2048][1]cuda:0", primals_314: "f32[512, 2048][2048, 1]cuda:0", primals_315: "f32[512][1]cuda:0", primals_316: "f32[512][1]cuda:0", primals_317: "f32[512][1]cuda:0", primals_318: "f32[4, 49, 49][2401, 49, 1]cuda:0", primals_319: "f32[1536, 512][512, 1]cuda:0", primals_320: "f32[1536][1]cuda:0", primals_321: "f32[169, 16][16, 1]cuda:0", primals_322: "i64[49, 49][49, 1]cuda:0", primals_323: "f32[512, 512][512, 1]cuda:0", primals_324: "f32[512][1]cuda:0", primals_325: "f32[512][1]cuda:0", primals_326: "f32[512][1]cuda:0", primals_327: "f32[2048, 512][512, 1]cuda:0", primals_328: "f32[2048][1]cuda:0", primals_329: "f32[512, 2048][2048, 1]cuda:0", primals_330: "f32[512][1]cuda:0", primals_331: "f32[2048][1]cuda:0", primals_332: "f32[2048][1]cuda:0", primals_333: "f32[1024, 2048][2048, 1]cuda:0", primals_334: "f32[1024][1]cuda:0", primals_335: "f32[1024][1]cuda:0", primals_336: "f32[3072, 1024][1024, 1]cuda:0", primals_337: "f32[3072][1]cuda:0", primals_338: "f32[169, 32][32, 1]cuda:0", primals_339: "i64[49, 49][49, 1]cuda:0", primals_340: "f32[1024, 1024][1024, 1]cuda:0", primals_341: "f32[1024][1]cuda:0", primals_342: "f32[1024][1]cuda:0", primals_343: "f32[1024][1]cuda:0", primals_344: "f32[4096, 1024][1024, 1]cuda:0", primals_345: "f32[4096][1]cuda:0", primals_346: "f32[1024, 4096][4096, 1]cuda:0", primals_347: "f32[1024][1]cuda:0", primals_348: "f32[1024][1]cuda:0", primals_349: "f32[1024][1]cuda:0", primals_350: "f32[3072, 1024][1024, 1]cuda:0", primals_351: "f32[3072][1]cuda:0", primals_352: "f32[169, 32][32, 1]cuda:0", primals_353: "i64[49, 49][49, 1]cuda:0", primals_354: "f32[1024, 1024][1024, 1]cuda:0", primals_355: "f32[1024][1]cuda:0", primals_356: "f32[1024][1]cuda:0", primals_357: "f32[1024][1]cuda:0", primals_358: "f32[4096, 1024][1024, 1]cuda:0", primals_359: "f32[4096][1]cuda:0", primals_360: "f32[1024, 4096][4096, 1]cuda:0", primals_361: "f32[1024][1]cuda:0", primals_362: "f32[1024][1]cuda:0", primals_363: "f32[1024][1]cuda:0", primals_364: "f32[1000, 1024][1024, 1]cuda:0", primals_365: "f32[1000][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.proj(x)
        convert_element_type: "bf16[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_3, torch.bfloat16);  primals_3 = None
        convert_element_type_1: "bf16[128, 3, 4, 4][48, 1, 12, 3]cuda:0" = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16);  primals_2 = None
        convert_element_type_2: "bf16[128, 3, 224, 224][150528, 1, 672, 3]cuda:0" = torch.ops.prims.convert_element_type.default(primals_1, torch.bfloat16);  primals_1 = None
        convolution: "bf16[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_2, convert_element_type_1, convert_element_type, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/format.py:68 in nchw_to, code: x = x.permute(0, 2, 3, 1)
        permute: "bf16[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.permute.default(convolution, [0, 2, 3, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:141 in forward, code: x = self.norm(x)
        convert_element_type_3: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute, torch.float32);  permute = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_3, [3], correction = 0, keepdim = True)
        getitem: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        add: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_3, getitem_1);  convert_element_type_3 = None
        mul: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, primals_4);  mul = None
        add_1: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, primals_5);  mul_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_1 = torch.ops.aten.var_mean.correction(add_1, [3], correction = 0, keepdim = True)
        getitem_2: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = var_mean_1[0]
        getitem_3: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        add_2: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub_1: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_1, getitem_3)
        mul_2: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_3: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2, primals_6);  mul_2 = None
        add_3: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_3, primals_7);  mul_3 = primals_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view: "f32[128, 8, 7, 8, 7, 128][401408, 50176, 7168, 896, 128, 1]cuda:0" = torch.ops.aten.reshape.default(add_3, [128, 8, 7, 8, 7, 128]);  add_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_1: "f32[128, 8, 8, 7, 7, 128][401408, 50176, 896, 7168, 128, 1]cuda:0" = torch.ops.aten.permute.default(view, [0, 1, 3, 2, 4, 5]);  view = None
        clone: "f32[128, 8, 8, 7, 7, 128][401408, 50176, 6272, 896, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_1: "f32[8192, 7, 7, 128][6272, 896, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone, [-1, 7, 7, 128]);  clone = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_2: "f32[8192, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_1, [-1, 49, 128]);  view_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        convert_element_type_4: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_9, torch.bfloat16);  primals_9 = None
        convert_element_type_5: "bf16[384, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_8, torch.bfloat16);  primals_8 = None
        convert_element_type_6: "bf16[8192, 49, 128][6272, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16);  view_2 = None
        view_3: "bf16[401408, 128][128, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_6, [401408, 128]);  convert_element_type_6 = None
        permute_2: "bf16[128, 384][1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_5, [1, 0]);  convert_element_type_5 = None
        addmm: "bf16[401408, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_4, view_3, permute_2);  convert_element_type_4 = None
        view_4: "bf16[8192, 49, 384][18816, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [8192, 49, 384]);  addmm = None
        view_5: "bf16[8192, 49, 3, 4, 32][18816, 384, 128, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_4, [8192, 49, 3, 4, -1]);  view_4 = None
        permute_3: "bf16[3, 8192, 4, 49, 32][128, 18816, 32, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_5, [2, 0, 3, 1, 4]);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind = torch.ops.aten.unbind.int(permute_3);  permute_3 = None
        getitem_4: "bf16[8192, 4, 49, 32][18816, 32, 384, 1]cuda:0" = unbind[0]
        getitem_5: "bf16[8192, 4, 49, 32][18816, 32, 384, 1]cuda:0" = unbind[1]
        getitem_6: "bf16[8192, 4, 49, 32][18816, 32, 384, 1]cuda:0" = unbind[2];  unbind = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_4: "bf16[8192, 4, 49, 32][6272, 32, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_4, 0.1767766952966369);  getitem_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_4: "bf16[8192, 4, 32, 49][18816, 32, 1, 384]cuda:0" = torch.ops.aten.permute.default(getitem_5, [0, 1, 3, 2]);  getitem_5 = None
        expand: "bf16[8192, 4, 49, 32][6272, 32, 128, 1]cuda:0" = torch.ops.aten.expand.default(mul_4, [8192, 4, 49, 32]);  mul_4 = None
        clone_1: "bf16[8192, 4, 49, 32][6272, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_6: "bf16[32768, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_1, [32768, 49, 32]);  clone_1 = None
        expand_1: "bf16[8192, 4, 32, 49][18816, 32, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_4, [8192, 4, 32, 49]);  permute_4 = None
        clone_2: "bf16[8192, 4, 32, 49][6272, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_7: "bf16[32768, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [32768, 32, 49]);  clone_2 = None
        constant_pad_nd_default_90: "bf16[32768, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_6, [0, 0, 0, 7, 0, 0])
        constant_pad_nd_default_91: "bf16[32768, 32, 56][1792, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_7, [0, 7, 0, 0, 0, 0])
        bmm_default_45: "bf16[32768, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_90, constant_pad_nd_default_91);  constant_pad_nd_default_90 = constant_pad_nd_default_91 = None
        slice_tensor_66: "bf16[32768, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_45, 1, 0, -7)
        slice_tensor_67: "bf16[32768, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_66, 2, 0, -7);  slice_tensor_66 = None
        view_8: "bf16[8192, 4, 49, 49][12544, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_67, [8192, 4, 49, 49]);  slice_tensor_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_9: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_11, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index: "f32[2401, 4][4, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_10, [view_9]);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_10: "f32[49, 49, 4][196, 4, 1]cuda:0" = torch.ops.aten.reshape.default(index, [49, 49, -1]);  index = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_5: "f32[4, 49, 49][1, 196, 4]cuda:0" = torch.ops.aten.permute.default(view_10, [2, 0, 1]);  view_10 = None
        clone_3: "f32[4, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze: "f32[1, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_3, 0);  clone_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_4: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_8, unsqueeze);  view_8 = unsqueeze = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax: "f32[8192, 4, 49, 1][196, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_4, [-1], True)
        sub_2: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_4, amax);  add_4 = None
        exp: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_1: "f32[8192, 4, 49, 1][196, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        convert_element_type_12: "bf16[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        expand_2: "bf16[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_12, [8192, 4, 49, 49]);  convert_element_type_12 = None
        view_11: "bf16[32768, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_2, [32768, 49, 49]);  expand_2 = None
        expand_3: "bf16[8192, 4, 49, 32][18816, 32, 384, 1]cuda:0" = torch.ops.aten.expand.default(getitem_6, [8192, 4, 49, 32]);  getitem_6 = None
        clone_5: "bf16[8192, 4, 49, 32][6272, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_12: "bf16[32768, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_5, [32768, 49, 32]);  clone_5 = None
        constant_pad_nd_default_88: "bf16[32768, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_11, [0, 7, 0, 7, 0, 0])
        constant_pad_nd_default_89: "bf16[32768, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_12, [0, 0, 0, 7, 0, 0])
        bmm_default_44: "bf16[32768, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_88, constant_pad_nd_default_89);  constant_pad_nd_default_88 = constant_pad_nd_default_89 = None
        slice_tensor_65: "bf16[32768, 49, 32][1792, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_44, 1, 0, -7);  bmm_default_44 = None
        view_13: "bf16[8192, 4, 49, 32][7168, 1792, 32, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_65, [8192, 4, 49, 32]);  slice_tensor_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_6: "bf16[8192, 49, 4, 32][7168, 32, 1792, 1]cuda:0" = torch.ops.aten.permute.default(view_13, [0, 2, 1, 3]);  view_13 = None
        clone_6: "bf16[8192, 49, 4, 32][6272, 128, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
        view_14: "bf16[8192, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_6, [8192, 49, 128]);  clone_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        convert_element_type_15: "bf16[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_13, torch.bfloat16);  primals_13 = None
        convert_element_type_16: "bf16[128, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_12, torch.bfloat16);  primals_12 = None
        view_15: "bf16[401408, 128][128, 1]cuda:0" = torch.ops.aten.reshape.default(view_14, [401408, 128]);  view_14 = None
        permute_7: "bf16[128, 128][1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_16, [1, 0]);  convert_element_type_16 = None
        addmm_1: "bf16[401408, 128][128, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_15, view_15, permute_7);  convert_element_type_15 = None
        view_16: "bf16[8192, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [8192, 49, 128]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_17: "bf16[8192, 7, 7, 128][6272, 896, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_16, [-1, 7, 7, 128]);  view_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_18: "bf16[128, 8, 8, 7, 7, 128][401408, 50176, 6272, 896, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_17, [-1, 8, 8, 7, 7, 128]);  view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_8: "bf16[128, 8, 7, 8, 7, 128][401408, 50176, 896, 6272, 128, 1]cuda:0" = torch.ops.aten.permute.default(view_18, [0, 1, 3, 2, 4, 5]);  view_18 = None
        clone_8: "bf16[128, 8, 7, 8, 7, 128][401408, 50176, 7168, 896, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
        view_19: "bf16[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_8, [-1, 56, 56, 128]);  clone_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_5: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1, view_19);  add_1 = view_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_20: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.reshape.default(add_5, [128, -1, 128]);  add_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_2 = torch.ops.aten.var_mean.correction(view_20, [2], correction = 0, keepdim = True)
        getitem_7: "f32[128, 3136, 1][3136, 1, 1]cuda:0" = var_mean_2[0]
        getitem_8: "f32[128, 3136, 1][3136, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        add_6: "f32[128, 3136, 1][3136, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_7, 1e-05);  getitem_7 = None
        rsqrt_2: "f32[128, 3136, 1][3136, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_3: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_20, getitem_8);  getitem_8 = None
        mul_5: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_6: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_5, primals_14)
        add_7: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_6, primals_15);  mul_6 = primals_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_20: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_17, torch.bfloat16);  primals_17 = None
        convert_element_type_21: "bf16[512, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_16, torch.bfloat16);  primals_16 = None
        convert_element_type_22: "bf16[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_7, torch.bfloat16);  add_7 = None
        view_21: "bf16[401408, 128][128, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_22, [401408, 128]);  convert_element_type_22 = None
        permute_9: "bf16[128, 512][1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_21, [1, 0]);  convert_element_type_21 = None
        addmm_2: "bf16[401408, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_20, view_21, permute_9);  convert_element_type_20 = None
        view_22: "bf16[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [128, 3136, 512])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_26: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_22, torch.float32);  view_22 = None
        mul_7: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_26, 0.5)
        mul_8: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_26, 0.7071067811865476);  convert_element_type_26 = None
        erf: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
        add_8: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_9: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_7, add_8);  mul_7 = add_8 = None
        convert_element_type_27: "bf16[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_28: "bf16[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_19, torch.bfloat16);  primals_19 = None
        convert_element_type_29: "bf16[128, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_18, torch.bfloat16);  primals_18 = None
        view_23: "bf16[401408, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_27, [401408, 512]);  convert_element_type_27 = None
        permute_10: "bf16[512, 128][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_29, [1, 0]);  convert_element_type_29 = None
        addmm_3: "bf16[401408, 128][128, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_28, view_23, permute_10);  convert_element_type_28 = None
        view_24: "bf16[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [128, 3136, 128]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_9: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_20, view_24);  view_20 = view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_25: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.reshape.default(add_9, [128, 56, 56, 128]);  add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_3 = torch.ops.aten.var_mean.correction(view_25, [3], correction = 0, keepdim = True)
        getitem_9: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = var_mean_3[0]
        getitem_10: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        add_10: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_9, 1e-05);  getitem_9 = None
        rsqrt_3: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_4: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_25, getitem_10);  getitem_10 = None
        mul_10: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = None
        mul_11: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, primals_20)
        add_11: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_11, primals_21);  mul_11 = primals_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota: "i64[56][1]cuda:0" = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_12: "i64[56][1]cuda:0" = torch.ops.aten.add.Tensor(iota, 3)
        fmod: "i64[56][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_12, 56);  add_12 = None
        index_1: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.index.Tensor(add_11, [None, fmod]);  add_11 = None
        index_2: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.index.Tensor(index_1, [None, None, fmod]);  index_1 = fmod = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_26: "f32[128, 8, 7, 8, 7, 128][401408, 50176, 7168, 896, 128, 1]cuda:0" = torch.ops.aten.reshape.default(index_2, [128, 8, 7, 8, 7, 128]);  index_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_11: "f32[128, 8, 8, 7, 7, 128][401408, 50176, 896, 7168, 128, 1]cuda:0" = torch.ops.aten.permute.default(view_26, [0, 1, 3, 2, 4, 5]);  view_26 = None
        clone_11: "f32[128, 8, 8, 7, 7, 128][401408, 50176, 6272, 896, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
        view_27: "f32[8192, 7, 7, 128][6272, 896, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_11, [-1, 7, 7, 128]);  clone_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_28: "f32[8192, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_27, [-1, 49, 128]);  view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        convert_element_type_33: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_24, torch.bfloat16);  primals_24 = None
        convert_element_type_34: "bf16[384, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_23, torch.bfloat16);  primals_23 = None
        convert_element_type_35: "bf16[8192, 49, 128][6272, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_28, torch.bfloat16);  view_28 = None
        view_29: "bf16[401408, 128][128, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_35, [401408, 128]);  convert_element_type_35 = None
        permute_12: "bf16[128, 384][1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_34, [1, 0]);  convert_element_type_34 = None
        addmm_4: "bf16[401408, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_33, view_29, permute_12);  convert_element_type_33 = None
        view_30: "bf16[8192, 49, 384][18816, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [8192, 49, 384]);  addmm_4 = None
        view_31: "bf16[8192, 49, 3, 4, 32][18816, 384, 128, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_30, [8192, 49, 3, 4, -1]);  view_30 = None
        permute_13: "bf16[3, 8192, 4, 49, 32][128, 18816, 32, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_31, [2, 0, 3, 1, 4]);  view_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_1 = torch.ops.aten.unbind.int(permute_13);  permute_13 = None
        getitem_11: "bf16[8192, 4, 49, 32][18816, 32, 384, 1]cuda:0" = unbind_1[0]
        getitem_12: "bf16[8192, 4, 49, 32][18816, 32, 384, 1]cuda:0" = unbind_1[1]
        getitem_13: "bf16[8192, 4, 49, 32][18816, 32, 384, 1]cuda:0" = unbind_1[2];  unbind_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_12: "bf16[8192, 4, 49, 32][6272, 32, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_11, 0.1767766952966369);  getitem_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_14: "bf16[8192, 4, 32, 49][18816, 32, 1, 384]cuda:0" = torch.ops.aten.permute.default(getitem_12, [0, 1, 3, 2]);  getitem_12 = None
        expand_4: "bf16[8192, 4, 49, 32][6272, 32, 128, 1]cuda:0" = torch.ops.aten.expand.default(mul_12, [8192, 4, 49, 32]);  mul_12 = None
        clone_12: "bf16[8192, 4, 49, 32][6272, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_32: "bf16[32768, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_12, [32768, 49, 32]);  clone_12 = None
        expand_5: "bf16[8192, 4, 32, 49][18816, 32, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_14, [8192, 4, 32, 49]);  permute_14 = None
        clone_13: "bf16[8192, 4, 32, 49][6272, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_33: "bf16[32768, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_13, [32768, 32, 49]);  clone_13 = None
        constant_pad_nd_default_86: "bf16[32768, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_32, [0, 0, 0, 7, 0, 0])
        constant_pad_nd_default_87: "bf16[32768, 32, 56][1792, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_33, [0, 7, 0, 0, 0, 0])
        bmm_default_43: "bf16[32768, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_86, constant_pad_nd_default_87);  constant_pad_nd_default_86 = constant_pad_nd_default_87 = None
        slice_tensor_63: "bf16[32768, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_43, 1, 0, -7)
        slice_tensor_64: "bf16[32768, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_63, 2, 0, -7);  slice_tensor_63 = None
        view_34: "bf16[8192, 4, 49, 49][12544, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_64, [8192, 4, 49, 49]);  slice_tensor_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_35: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_26, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_3: "f32[2401, 4][4, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_25, [view_35]);  view_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_36: "f32[49, 49, 4][196, 4, 1]cuda:0" = torch.ops.aten.reshape.default(index_3, [49, 49, -1]);  index_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_15: "f32[4, 49, 49][1, 196, 4]cuda:0" = torch.ops.aten.permute.default(view_36, [2, 0, 1]);  view_36 = None
        clone_14: "f32[4, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_1: "f32[1, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_14, 0);  clone_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_14: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_34, unsqueeze_1);  view_34 = unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_37: "f32[128, 64, 4, 49, 49][614656, 9604, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_14, [-1, 64, 4, 49, 49]);  add_14 = None
        unsqueeze_2: "f32[64, 1, 49, 49][2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_22, 1)
        unsqueeze_3: "f32[1, 64, 1, 49, 49][153664, 2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 0);  unsqueeze_2 = None
        add_15: "f32[128, 64, 4, 49, 49][614656, 9604, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_37, unsqueeze_3);  view_37 = unsqueeze_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_38: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_15, [-1, 4, 49, 49]);  add_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_1: "f32[8192, 4, 49, 1][196, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(view_38, [-1], True)
        sub_5: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_38, amax_1);  view_38 = None
        exp_1: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        sum_2: "f32[8192, 4, 49, 1][196, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        convert_element_type_41: "bf16[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        expand_6: "bf16[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_41, [8192, 4, 49, 49]);  convert_element_type_41 = None
        view_39: "bf16[32768, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_6, [32768, 49, 49]);  expand_6 = None
        expand_7: "bf16[8192, 4, 49, 32][18816, 32, 384, 1]cuda:0" = torch.ops.aten.expand.default(getitem_13, [8192, 4, 49, 32]);  getitem_13 = None
        clone_16: "bf16[8192, 4, 49, 32][6272, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_40: "bf16[32768, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_16, [32768, 49, 32]);  clone_16 = None
        constant_pad_nd_default_84: "bf16[32768, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_39, [0, 7, 0, 7, 0, 0])
        constant_pad_nd_default_85: "bf16[32768, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_40, [0, 0, 0, 7, 0, 0])
        bmm_default_42: "bf16[32768, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_84, constant_pad_nd_default_85);  constant_pad_nd_default_84 = constant_pad_nd_default_85 = None
        slice_tensor_62: "bf16[32768, 49, 32][1792, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_42, 1, 0, -7);  bmm_default_42 = None
        view_41: "bf16[8192, 4, 49, 32][7168, 1792, 32, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_62, [8192, 4, 49, 32]);  slice_tensor_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_16: "bf16[8192, 49, 4, 32][7168, 32, 1792, 1]cuda:0" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
        clone_17: "bf16[8192, 49, 4, 32][6272, 128, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        view_42: "bf16[8192, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_17, [8192, 49, 128]);  clone_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        convert_element_type_44: "bf16[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_28, torch.bfloat16);  primals_28 = None
        convert_element_type_45: "bf16[128, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_27, torch.bfloat16);  primals_27 = None
        view_43: "bf16[401408, 128][128, 1]cuda:0" = torch.ops.aten.reshape.default(view_42, [401408, 128]);  view_42 = None
        permute_17: "bf16[128, 128][1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_45, [1, 0]);  convert_element_type_45 = None
        addmm_5: "bf16[401408, 128][128, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_44, view_43, permute_17);  convert_element_type_44 = None
        view_44: "bf16[8192, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [8192, 49, 128]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_45: "bf16[8192, 7, 7, 128][6272, 896, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_44, [-1, 7, 7, 128]);  view_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_46: "bf16[128, 8, 8, 7, 7, 128][401408, 50176, 6272, 896, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_45, [-1, 8, 8, 7, 7, 128]);  view_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_18: "bf16[128, 8, 7, 8, 7, 128][401408, 50176, 896, 6272, 128, 1]cuda:0" = torch.ops.aten.permute.default(view_46, [0, 1, 3, 2, 4, 5]);  view_46 = None
        clone_19: "bf16[128, 8, 7, 8, 7, 128][401408, 50176, 7168, 896, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_47: "bf16[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_19, [-1, 56, 56, 128]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        add_16: "i64[56][1]cuda:0" = torch.ops.aten.add.Tensor(iota, 53);  iota = None
        fmod_2: "i64[56][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_16, 56);  add_16 = None
        index_4: "bf16[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.index.Tensor(view_47, [None, fmod_2]);  view_47 = None
        index_5: "bf16[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.index.Tensor(index_4, [None, None, fmod_2]);  index_4 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[46][1]cuda:0" = torch.ops.prims.inductor_seeds.default(46, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_45: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        lt: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_45, 0.9956521736457944);  inductor_random_default_45 = None
        convert_element_type_49: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_2: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_49, 0.9956521736457944);  convert_element_type_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_13: "bf16[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(index_5, div_2);  index_5 = div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_18: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_25, mul_13);  view_25 = mul_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_48: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.reshape.default(add_18, [128, -1, 128]);  add_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_4 = torch.ops.aten.var_mean.correction(view_48, [2], correction = 0, keepdim = True)
        getitem_14: "f32[128, 3136, 1][3136, 1, 1]cuda:0" = var_mean_4[0]
        getitem_15: "f32[128, 3136, 1][3136, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        add_19: "f32[128, 3136, 1][3136, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_4: "f32[128, 3136, 1][3136, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        sub_6: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_48, getitem_15);  getitem_15 = None
        mul_14: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
        mul_15: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, primals_29)
        add_20: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_15, primals_30);  mul_15 = primals_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_50: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_32, torch.bfloat16);  primals_32 = None
        convert_element_type_51: "bf16[512, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_31, torch.bfloat16);  primals_31 = None
        convert_element_type_52: "bf16[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_20, torch.bfloat16);  add_20 = None
        view_49: "bf16[401408, 128][128, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_52, [401408, 128]);  convert_element_type_52 = None
        permute_19: "bf16[128, 512][1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_51, [1, 0]);  convert_element_type_51 = None
        addmm_6: "bf16[401408, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_50, view_49, permute_19);  convert_element_type_50 = None
        view_50: "bf16[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [128, 3136, 512])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_56: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_50, torch.float32);  view_50 = None
        mul_16: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_56, 0.5)
        mul_17: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_56, 0.7071067811865476);  convert_element_type_56 = None
        erf_1: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.erf.default(mul_17);  mul_17 = None
        add_21: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_18: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_16, add_21);  mul_16 = add_21 = None
        convert_element_type_57: "bf16[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_18, torch.bfloat16);  mul_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_58: "bf16[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_34, torch.bfloat16);  primals_34 = None
        convert_element_type_59: "bf16[128, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_33, torch.bfloat16);  primals_33 = None
        view_51: "bf16[401408, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_57, [401408, 512]);  convert_element_type_57 = None
        permute_20: "bf16[512, 128][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_59, [1, 0]);  convert_element_type_59 = None
        addmm_7: "bf16[401408, 128][128, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_58, view_51, permute_20);  convert_element_type_58 = None
        view_52: "bf16[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [128, 3136, 128]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_1: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_44: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        lt_1: "b8[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_44, 0.9956521736457944);  inductor_random_default_44 = None
        convert_element_type_63: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_1, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_3: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_63, 0.9956521736457944);  convert_element_type_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_19: "bf16[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_52, div_3);  view_52 = div_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_22: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_48, mul_19);  view_48 = mul_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_53: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.reshape.default(add_22, [128, 56, 56, 128]);  add_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:539 in forward, code: x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        view_54: "f32[128, 28, 2, 28, 2, 128][401408, 14336, 7168, 256, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_53, [128, 28, 2, 28, 2, 128]);  view_53 = None
        permute_21: "f32[128, 28, 28, 2, 2, 128][401408, 14336, 256, 128, 7168, 1]cuda:0" = torch.ops.aten.permute.default(view_54, [0, 1, 3, 4, 2, 5]);  view_54 = None
        clone_22: "f32[128, 28, 28, 2, 2, 128][401408, 14336, 512, 256, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_21, memory_format = torch.contiguous_format);  permute_21 = None
        view_55: "f32[128, 28, 28, 512][401408, 14336, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_22, [128, 28, 28, 512]);  clone_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:540 in forward, code: x = self.norm(x)
        var_mean_5 = torch.ops.aten.var_mean.correction(view_55, [3], correction = 0, keepdim = True)
        getitem_16: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = var_mean_5[0]
        getitem_17: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        add_23: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_5: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        sub_7: "f32[128, 28, 28, 512][401408, 14336, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_55, getitem_17);  view_55 = getitem_17 = None
        mul_20: "f32[128, 28, 28, 512][401408, 14336, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_5);  sub_7 = None
        mul_21: "f32[128, 28, 28, 512][401408, 14336, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_20, primals_35)
        add_24: "f32[128, 28, 28, 512][401408, 14336, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_21, primals_36);  mul_21 = primals_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:541 in forward, code: x = self.reduction(x)
        convert_element_type_64: "bf16[256, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_37, torch.bfloat16);  primals_37 = None
        convert_element_type_65: "bf16[128, 28, 28, 512][401408, 14336, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_24, torch.bfloat16);  add_24 = None
        permute_22: "bf16[512, 256][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_64, [1, 0]);  convert_element_type_64 = None
        view_56: "bf16[100352, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_65, [100352, 512]);  convert_element_type_65 = None
        mm: "bf16[100352, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_56, permute_22)
        view_57: "bf16[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [128, 28, 28, 256])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        convert_element_type_68: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_57, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_68, [3], correction = 0, keepdim = True)
        getitem_18: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = var_mean_6[0]
        getitem_19: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        add_25: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_6: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        sub_8: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_68, getitem_19);  convert_element_type_68 = None
        mul_22: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_6);  sub_8 = None
        mul_23: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_22, primals_38);  mul_22 = None
        add_26: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_23, primals_39);  mul_23 = primals_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_58: "f32[128, 4, 7, 4, 7, 256][200704, 50176, 7168, 1792, 256, 1]cuda:0" = torch.ops.aten.reshape.default(add_26, [128, 4, 7, 4, 7, 256]);  add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_23: "f32[128, 4, 4, 7, 7, 256][200704, 50176, 1792, 7168, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_58, [0, 1, 3, 2, 4, 5]);  view_58 = None
        clone_23: "f32[128, 4, 4, 7, 7, 256][200704, 50176, 12544, 1792, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
        view_59: "f32[2048, 7, 7, 256][12544, 1792, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_23, [-1, 7, 7, 256]);  clone_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_60: "f32[2048, 49, 256][12544, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_59, [-1, 49, 256]);  view_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        convert_element_type_69: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_41, torch.bfloat16);  primals_41 = None
        convert_element_type_70: "bf16[768, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_40, torch.bfloat16);  primals_40 = None
        convert_element_type_71: "bf16[2048, 49, 256][12544, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_60, torch.bfloat16);  view_60 = None
        view_61: "bf16[100352, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_71, [100352, 256]);  convert_element_type_71 = None
        permute_24: "bf16[256, 768][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_70, [1, 0]);  convert_element_type_70 = None
        addmm_8: "bf16[100352, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_69, view_61, permute_24);  convert_element_type_69 = None
        view_62: "bf16[2048, 49, 768][37632, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [2048, 49, 768]);  addmm_8 = None
        view_63: "bf16[2048, 49, 3, 8, 32][37632, 768, 256, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_62, [2048, 49, 3, 8, -1]);  view_62 = None
        permute_25: "bf16[3, 2048, 8, 49, 32][256, 37632, 32, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_63, [2, 0, 3, 1, 4]);  view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_2 = torch.ops.aten.unbind.int(permute_25);  permute_25 = None
        getitem_20: "bf16[2048, 8, 49, 32][37632, 32, 768, 1]cuda:0" = unbind_2[0]
        getitem_21: "bf16[2048, 8, 49, 32][37632, 32, 768, 1]cuda:0" = unbind_2[1]
        getitem_22: "bf16[2048, 8, 49, 32][37632, 32, 768, 1]cuda:0" = unbind_2[2];  unbind_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_24: "bf16[2048, 8, 49, 32][12544, 32, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_20, 0.1767766952966369);  getitem_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_26: "bf16[2048, 8, 32, 49][37632, 32, 1, 768]cuda:0" = torch.ops.aten.permute.default(getitem_21, [0, 1, 3, 2]);  getitem_21 = None
        expand_8: "bf16[2048, 8, 49, 32][12544, 32, 256, 1]cuda:0" = torch.ops.aten.expand.default(mul_24, [2048, 8, 49, 32]);  mul_24 = None
        clone_24: "bf16[2048, 8, 49, 32][12544, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_64: "bf16[16384, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_24, [16384, 49, 32]);  clone_24 = None
        expand_9: "bf16[2048, 8, 32, 49][37632, 32, 1, 768]cuda:0" = torch.ops.aten.expand.default(permute_26, [2048, 8, 32, 49]);  permute_26 = None
        clone_25: "bf16[2048, 8, 32, 49][12544, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_65: "bf16[16384, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_25, [16384, 32, 49]);  clone_25 = None
        constant_pad_nd_default_82: "bf16[16384, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_64, [0, 0, 0, 7, 0, 0])
        constant_pad_nd_default_83: "bf16[16384, 32, 56][1792, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_65, [0, 7, 0, 0, 0, 0])
        bmm_default_41: "bf16[16384, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_82, constant_pad_nd_default_83);  constant_pad_nd_default_82 = constant_pad_nd_default_83 = None
        slice_tensor_60: "bf16[16384, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_41, 1, 0, -7)
        slice_tensor_61: "bf16[16384, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_60, 2, 0, -7);  slice_tensor_60 = None
        view_66: "bf16[2048, 8, 49, 49][25088, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_61, [2048, 8, 49, 49]);  slice_tensor_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_67: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_43, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_6: "f32[2401, 8][8, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_42, [view_67]);  view_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_68: "f32[49, 49, 8][392, 8, 1]cuda:0" = torch.ops.aten.reshape.default(index_6, [49, 49, -1]);  index_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_27: "f32[8, 49, 49][1, 392, 8]cuda:0" = torch.ops.aten.permute.default(view_68, [2, 0, 1]);  view_68 = None
        clone_26: "f32[8, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_4: "f32[1, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_26, 0);  clone_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_27: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_66, unsqueeze_4);  view_66 = unsqueeze_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_2: "f32[2048, 8, 49, 1][392, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_27, [-1], True)
        sub_9: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_27, amax_2);  add_27 = None
        exp_2: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        sum_3: "f32[2048, 8, 49, 1][392, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_4: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        convert_element_type_77: "bf16[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None
        expand_10: "bf16[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_77, [2048, 8, 49, 49]);  convert_element_type_77 = None
        view_69: "bf16[16384, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_10, [16384, 49, 49]);  expand_10 = None
        expand_11: "bf16[2048, 8, 49, 32][37632, 32, 768, 1]cuda:0" = torch.ops.aten.expand.default(getitem_22, [2048, 8, 49, 32]);  getitem_22 = None
        clone_28: "bf16[2048, 8, 49, 32][12544, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_70: "bf16[16384, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_28, [16384, 49, 32]);  clone_28 = None
        constant_pad_nd_default_80: "bf16[16384, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_69, [0, 7, 0, 7, 0, 0])
        constant_pad_nd_default_81: "bf16[16384, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_70, [0, 0, 0, 7, 0, 0])
        bmm_default_40: "bf16[16384, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_80, constant_pad_nd_default_81);  constant_pad_nd_default_80 = constant_pad_nd_default_81 = None
        slice_tensor_59: "bf16[16384, 49, 32][1792, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_40, 1, 0, -7);  bmm_default_40 = None
        view_71: "bf16[2048, 8, 49, 32][14336, 1792, 32, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_59, [2048, 8, 49, 32]);  slice_tensor_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_28: "bf16[2048, 49, 8, 32][14336, 32, 1792, 1]cuda:0" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
        clone_29: "bf16[2048, 49, 8, 32][12544, 256, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_28, memory_format = torch.contiguous_format);  permute_28 = None
        view_72: "bf16[2048, 49, 256][12544, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_29, [2048, 49, 256]);  clone_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        convert_element_type_80: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_45, torch.bfloat16);  primals_45 = None
        convert_element_type_81: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16);  primals_44 = None
        view_73: "bf16[100352, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_72, [100352, 256]);  view_72 = None
        permute_29: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_81, [1, 0]);  convert_element_type_81 = None
        addmm_9: "bf16[100352, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_80, view_73, permute_29);  convert_element_type_80 = None
        view_74: "bf16[2048, 49, 256][12544, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [2048, 49, 256]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_75: "bf16[2048, 7, 7, 256][12544, 1792, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_74, [-1, 7, 7, 256]);  view_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_76: "bf16[128, 4, 4, 7, 7, 256][200704, 50176, 12544, 1792, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_75, [-1, 4, 4, 7, 7, 256]);  view_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_30: "bf16[128, 4, 7, 4, 7, 256][200704, 50176, 1792, 12544, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_76, [0, 1, 3, 2, 4, 5]);  view_76 = None
        clone_31: "bf16[128, 4, 7, 4, 7, 256][200704, 50176, 7168, 1792, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
        view_77: "bf16[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_31, [-1, 28, 28, 256]);  clone_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_2: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_43: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        lt_2: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_43, 0.9913043472915888);  inductor_random_default_43 = None
        convert_element_type_85: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_2, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_5: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_85, 0.9913043472915888);  convert_element_type_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_25: "bf16[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_77, div_5);  view_77 = div_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_28: "bf16[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(view_57, mul_25);  view_57 = mul_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_78: "bf16[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.reshape.default(add_28, [128, -1, 256]);  add_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        convert_element_type_86: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_78, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_86, [2], correction = 0, keepdim = True)
        getitem_23: "f32[128, 784, 1][784, 1, 1]cuda:0" = var_mean_7[0]
        getitem_24: "f32[128, 784, 1][784, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        add_29: "f32[128, 784, 1][784, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_23, 1e-05);  getitem_23 = None
        rsqrt_7: "f32[128, 784, 1][784, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        sub_10: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_86, getitem_24);  convert_element_type_86 = None
        mul_26: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_7);  sub_10 = None
        mul_27: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_26, primals_46);  mul_26 = None
        add_30: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_27, primals_47);  mul_27 = primals_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_87: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_49, torch.bfloat16);  primals_49 = None
        convert_element_type_88: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_48, torch.bfloat16);  primals_48 = None
        convert_element_type_89: "bf16[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_30, torch.bfloat16);  add_30 = None
        view_79: "bf16[100352, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_89, [100352, 256]);  convert_element_type_89 = None
        permute_31: "bf16[256, 1024][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_88, [1, 0]);  convert_element_type_88 = None
        addmm_10: "bf16[100352, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_87, view_79, permute_31);  convert_element_type_87 = None
        view_80: "bf16[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [128, 784, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_93: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_80, torch.float32);  view_80 = None
        mul_28: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_93, 0.5)
        mul_29: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_93, 0.7071067811865476);  convert_element_type_93 = None
        erf_2: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.erf.default(mul_29);  mul_29 = None
        add_31: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_30: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, add_31);  mul_28 = add_31 = None
        convert_element_type_94: "bf16[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_30, torch.bfloat16);  mul_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_95: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_51, torch.bfloat16);  primals_51 = None
        convert_element_type_96: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_50, torch.bfloat16);  primals_50 = None
        view_81: "bf16[100352, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_94, [100352, 1024]);  convert_element_type_94 = None
        permute_32: "bf16[1024, 256][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_96, [1, 0]);  convert_element_type_96 = None
        addmm_11: "bf16[100352, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_95, view_81, permute_32);  convert_element_type_95 = None
        view_82: "bf16[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [128, 784, 256]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_3: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_42: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        lt_3: "b8[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_42, 0.9913043472915888);  inductor_random_default_42 = None
        convert_element_type_100: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_3, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_6: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_100, 0.9913043472915888);  convert_element_type_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_31: "bf16[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_82, div_6);  view_82 = div_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_32: "bf16[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(view_78, mul_31);  mul_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_83: "bf16[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.reshape.default(add_32, [128, 28, 28, 256]);  add_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        convert_element_type_101: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_83, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_101, [3], correction = 0, keepdim = True)
        getitem_25: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = var_mean_8[0]
        getitem_26: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        add_33: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_25, 1e-05);  getitem_25 = None
        rsqrt_8: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        sub_11: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_101, getitem_26);  convert_element_type_101 = None
        mul_32: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_8);  sub_11 = None
        mul_33: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_32, primals_52);  mul_32 = None
        add_34: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_33, primals_53);  mul_33 = primals_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota_4: "i64[28][1]cuda:0" = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_35: "i64[28][1]cuda:0" = torch.ops.aten.add.Tensor(iota_4, 3)
        fmod_4: "i64[28][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_35, 28);  add_35 = None
        index_7: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.index.Tensor(add_34, [None, fmod_4]);  add_34 = None
        index_8: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.index.Tensor(index_7, [None, None, fmod_4]);  index_7 = fmod_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_84: "f32[128, 4, 7, 4, 7, 256][200704, 50176, 7168, 1792, 256, 1]cuda:0" = torch.ops.aten.reshape.default(index_8, [128, 4, 7, 4, 7, 256]);  index_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_33: "f32[128, 4, 4, 7, 7, 256][200704, 50176, 1792, 7168, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_84, [0, 1, 3, 2, 4, 5]);  view_84 = None
        clone_34: "f32[128, 4, 4, 7, 7, 256][200704, 50176, 12544, 1792, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_33, memory_format = torch.contiguous_format);  permute_33 = None
        view_85: "f32[2048, 7, 7, 256][12544, 1792, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_34, [-1, 7, 7, 256]);  clone_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_86: "f32[2048, 49, 256][12544, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_85, [-1, 49, 256]);  view_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        convert_element_type_102: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_56, torch.bfloat16);  primals_56 = None
        convert_element_type_103: "bf16[768, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_55, torch.bfloat16);  primals_55 = None
        convert_element_type_104: "bf16[2048, 49, 256][12544, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_86, torch.bfloat16);  view_86 = None
        view_87: "bf16[100352, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_104, [100352, 256]);  convert_element_type_104 = None
        permute_34: "bf16[256, 768][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_103, [1, 0]);  convert_element_type_103 = None
        addmm_12: "bf16[100352, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_102, view_87, permute_34);  convert_element_type_102 = None
        view_88: "bf16[2048, 49, 768][37632, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [2048, 49, 768]);  addmm_12 = None
        view_89: "bf16[2048, 49, 3, 8, 32][37632, 768, 256, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_88, [2048, 49, 3, 8, -1]);  view_88 = None
        permute_35: "bf16[3, 2048, 8, 49, 32][256, 37632, 32, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_89, [2, 0, 3, 1, 4]);  view_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_3 = torch.ops.aten.unbind.int(permute_35);  permute_35 = None
        getitem_27: "bf16[2048, 8, 49, 32][37632, 32, 768, 1]cuda:0" = unbind_3[0]
        getitem_28: "bf16[2048, 8, 49, 32][37632, 32, 768, 1]cuda:0" = unbind_3[1]
        getitem_29: "bf16[2048, 8, 49, 32][37632, 32, 768, 1]cuda:0" = unbind_3[2];  unbind_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_34: "bf16[2048, 8, 49, 32][12544, 32, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_27, 0.1767766952966369);  getitem_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_36: "bf16[2048, 8, 32, 49][37632, 32, 1, 768]cuda:0" = torch.ops.aten.permute.default(getitem_28, [0, 1, 3, 2]);  getitem_28 = None
        expand_12: "bf16[2048, 8, 49, 32][12544, 32, 256, 1]cuda:0" = torch.ops.aten.expand.default(mul_34, [2048, 8, 49, 32]);  mul_34 = None
        clone_35: "bf16[2048, 8, 49, 32][12544, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_90: "bf16[16384, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_35, [16384, 49, 32]);  clone_35 = None
        expand_13: "bf16[2048, 8, 32, 49][37632, 32, 1, 768]cuda:0" = torch.ops.aten.expand.default(permute_36, [2048, 8, 32, 49]);  permute_36 = None
        clone_36: "bf16[2048, 8, 32, 49][12544, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_91: "bf16[16384, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_36, [16384, 32, 49]);  clone_36 = None
        constant_pad_nd_default_78: "bf16[16384, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_90, [0, 0, 0, 7, 0, 0])
        constant_pad_nd_default_79: "bf16[16384, 32, 56][1792, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_91, [0, 7, 0, 0, 0, 0])
        bmm_default_39: "bf16[16384, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_78, constant_pad_nd_default_79);  constant_pad_nd_default_78 = constant_pad_nd_default_79 = None
        slice_tensor_57: "bf16[16384, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_39, 1, 0, -7)
        slice_tensor_58: "bf16[16384, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_57, 2, 0, -7);  slice_tensor_57 = None
        view_92: "bf16[2048, 8, 49, 49][25088, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_58, [2048, 8, 49, 49]);  slice_tensor_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_93: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_58, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_9: "f32[2401, 8][8, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_57, [view_93]);  view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_94: "f32[49, 49, 8][392, 8, 1]cuda:0" = torch.ops.aten.reshape.default(index_9, [49, 49, -1]);  index_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_37: "f32[8, 49, 49][1, 392, 8]cuda:0" = torch.ops.aten.permute.default(view_94, [2, 0, 1]);  view_94 = None
        clone_37: "f32[8, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_5: "f32[1, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_37, 0);  clone_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_37: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_92, unsqueeze_5);  view_92 = unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_95: "f32[128, 16, 8, 49, 49][307328, 19208, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_37, [-1, 16, 8, 49, 49]);  add_37 = None
        unsqueeze_6: "f32[16, 1, 49, 49][2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_54, 1)
        unsqueeze_7: "f32[1, 16, 1, 49, 49][38416, 2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 0);  unsqueeze_6 = None
        add_38: "f32[128, 16, 8, 49, 49][307328, 19208, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_95, unsqueeze_7);  view_95 = unsqueeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_96: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_38, [-1, 8, 49, 49]);  add_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_3: "f32[2048, 8, 49, 1][392, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(view_96, [-1], True)
        sub_12: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_96, amax_3);  view_96 = None
        exp_3: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_4: "f32[2048, 8, 49, 1][392, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_7: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        convert_element_type_110: "bf16[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None
        expand_14: "bf16[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_110, [2048, 8, 49, 49]);  convert_element_type_110 = None
        view_97: "bf16[16384, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_14, [16384, 49, 49]);  expand_14 = None
        expand_15: "bf16[2048, 8, 49, 32][37632, 32, 768, 1]cuda:0" = torch.ops.aten.expand.default(getitem_29, [2048, 8, 49, 32]);  getitem_29 = None
        clone_39: "bf16[2048, 8, 49, 32][12544, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_98: "bf16[16384, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_39, [16384, 49, 32]);  clone_39 = None
        constant_pad_nd_default_76: "bf16[16384, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_97, [0, 7, 0, 7, 0, 0])
        constant_pad_nd_default_77: "bf16[16384, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_98, [0, 0, 0, 7, 0, 0])
        bmm_default_38: "bf16[16384, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_76, constant_pad_nd_default_77);  constant_pad_nd_default_76 = constant_pad_nd_default_77 = None
        slice_tensor_56: "bf16[16384, 49, 32][1792, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_38, 1, 0, -7);  bmm_default_38 = None
        view_99: "bf16[2048, 8, 49, 32][14336, 1792, 32, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_56, [2048, 8, 49, 32]);  slice_tensor_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_38: "bf16[2048, 49, 8, 32][14336, 32, 1792, 1]cuda:0" = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None
        clone_40: "bf16[2048, 49, 8, 32][12544, 256, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
        view_100: "bf16[2048, 49, 256][12544, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_40, [2048, 49, 256]);  clone_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        convert_element_type_113: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_60, torch.bfloat16);  primals_60 = None
        convert_element_type_114: "bf16[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_59, torch.bfloat16);  primals_59 = None
        view_101: "bf16[100352, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_100, [100352, 256]);  view_100 = None
        permute_39: "bf16[256, 256][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_114, [1, 0]);  convert_element_type_114 = None
        addmm_13: "bf16[100352, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_113, view_101, permute_39);  convert_element_type_113 = None
        view_102: "bf16[2048, 49, 256][12544, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [2048, 49, 256]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_103: "bf16[2048, 7, 7, 256][12544, 1792, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_102, [-1, 7, 7, 256]);  view_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_104: "bf16[128, 4, 4, 7, 7, 256][200704, 50176, 12544, 1792, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_103, [-1, 4, 4, 7, 7, 256]);  view_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_40: "bf16[128, 4, 7, 4, 7, 256][200704, 50176, 1792, 12544, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_104, [0, 1, 3, 2, 4, 5]);  view_104 = None
        clone_42: "bf16[128, 4, 7, 4, 7, 256][200704, 50176, 7168, 1792, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_105: "bf16[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_42, [-1, 28, 28, 256]);  clone_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        add_39: "i64[28][1]cuda:0" = torch.ops.aten.add.Tensor(iota_4, 25);  iota_4 = None
        fmod_6: "i64[28][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_39, 28);  add_39 = None
        index_10: "bf16[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.index.Tensor(view_105, [None, fmod_6]);  view_105 = None
        index_11: "bf16[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.index.Tensor(index_10, [None, None, fmod_6]);  index_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_4: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_random_default_41: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        lt_4: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_41, 0.9869565209373832);  inductor_random_default_41 = None
        convert_element_type_118: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_4, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_8: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_118, 0.9869565209373832);  convert_element_type_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_35: "bf16[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(index_11, div_8);  index_11 = div_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_41: "bf16[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(view_83, mul_35);  mul_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_106: "bf16[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.reshape.default(add_41, [128, -1, 256]);  add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        convert_element_type_119: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_106, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_119, [2], correction = 0, keepdim = True)
        getitem_30: "f32[128, 784, 1][784, 1, 1]cuda:0" = var_mean_9[0]
        getitem_31: "f32[128, 784, 1][784, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        add_42: "f32[128, 784, 1][784, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_9: "f32[128, 784, 1][784, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_13: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_119, getitem_31);  convert_element_type_119 = None
        mul_36: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_9);  sub_13 = None
        mul_37: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_36, primals_61);  mul_36 = None
        add_43: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_37, primals_62);  mul_37 = primals_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_120: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_64, torch.bfloat16);  primals_64 = None
        convert_element_type_121: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_63, torch.bfloat16);  primals_63 = None
        convert_element_type_122: "bf16[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_43, torch.bfloat16);  add_43 = None
        view_107: "bf16[100352, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_122, [100352, 256]);  convert_element_type_122 = None
        permute_41: "bf16[256, 1024][1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_121, [1, 0]);  convert_element_type_121 = None
        addmm_14: "bf16[100352, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_120, view_107, permute_41);  convert_element_type_120 = None
        view_108: "bf16[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [128, 784, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_126: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_108, torch.float32);  view_108 = None
        mul_38: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_126, 0.5)
        mul_39: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_126, 0.7071067811865476);  convert_element_type_126 = None
        erf_3: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
        add_44: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_40: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, add_44);  mul_38 = add_44 = None
        convert_element_type_127: "bf16[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_40, torch.bfloat16);  mul_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_128: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_66, torch.bfloat16);  primals_66 = None
        convert_element_type_129: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_65, torch.bfloat16);  primals_65 = None
        view_109: "bf16[100352, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_127, [100352, 1024]);  convert_element_type_127 = None
        permute_42: "bf16[1024, 256][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_129, [1, 0]);  convert_element_type_129 = None
        addmm_15: "bf16[100352, 256][256, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_128, view_109, permute_42);  convert_element_type_128 = None
        view_110: "bf16[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [128, 784, 256]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_5: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_random_default_40: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_5, 'rand');  inductor_lookup_seed_default_5 = None
        lt_5: "b8[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_40, 0.9869565209373832);  inductor_random_default_40 = None
        convert_element_type_133: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_5, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_9: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_133, 0.9869565209373832);  convert_element_type_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_41: "bf16[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_110, div_9);  view_110 = div_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_45: "bf16[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(view_106, mul_41);  mul_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_111: "bf16[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.reshape.default(add_45, [128, 28, 28, 256]);  add_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:539 in forward, code: x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        view_112: "bf16[128, 14, 2, 14, 2, 256][200704, 14336, 7168, 512, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_111, [128, 14, 2, 14, 2, 256]);  view_111 = None
        permute_43: "bf16[128, 14, 14, 2, 2, 256][200704, 14336, 512, 256, 7168, 1]cuda:0" = torch.ops.aten.permute.default(view_112, [0, 1, 3, 4, 2, 5]);  view_112 = None
        clone_45: "bf16[128, 14, 14, 2, 2, 256][200704, 14336, 1024, 512, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_43, memory_format = torch.contiguous_format);  permute_43 = None
        view_113: "bf16[128, 14, 14, 1024][200704, 14336, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_45, [128, 14, 14, 1024]);  clone_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:540 in forward, code: x = self.norm(x)
        convert_element_type_134: "f32[128, 14, 14, 1024][200704, 14336, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_113, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_134, [3], correction = 0, keepdim = True)
        getitem_32: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_10[0]
        getitem_33: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        add_46: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_10: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_14: "f32[128, 14, 14, 1024][200704, 14336, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_134, getitem_33);  convert_element_type_134 = None
        mul_42: "f32[128, 14, 14, 1024][200704, 14336, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_10);  sub_14 = None
        mul_43: "f32[128, 14, 14, 1024][200704, 14336, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, primals_67);  mul_42 = None
        add_47: "f32[128, 14, 14, 1024][200704, 14336, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_43, primals_68);  mul_43 = primals_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:541 in forward, code: x = self.reduction(x)
        convert_element_type_135: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_69, torch.bfloat16);  primals_69 = None
        convert_element_type_136: "bf16[128, 14, 14, 1024][200704, 14336, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_47, torch.bfloat16);  add_47 = None
        permute_44: "bf16[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_135, [1, 0]);  convert_element_type_135 = None
        view_114: "bf16[25088, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_136, [25088, 1024]);  convert_element_type_136 = None
        mm_1: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_114, permute_44)
        view_115: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_1, [128, 14, 14, 512])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        convert_element_type_139: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_115, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_139, [3], correction = 0, keepdim = True)
        getitem_34: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_11[0]
        getitem_35: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        add_48: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_11: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        sub_15: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_139, getitem_35);  convert_element_type_139 = None
        mul_44: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_11);  sub_15 = None
        mul_45: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_44, primals_70);  mul_44 = None
        add_49: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_45, primals_71);  mul_45 = primals_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_116: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_49, [128, 2, 7, 2, 7, 512]);  add_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_45: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_116, [0, 1, 3, 2, 4, 5]);  view_116 = None
        clone_46: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
        view_117: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_46, [-1, 7, 7, 512]);  clone_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_118: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_117, [-1, 49, 512]);  view_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        convert_element_type_140: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_73, torch.bfloat16);  primals_73 = None
        convert_element_type_141: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_72, torch.bfloat16);  primals_72 = None
        convert_element_type_142: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_118, torch.bfloat16);  view_118 = None
        view_119: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_142, [25088, 512]);  convert_element_type_142 = None
        permute_46: "bf16[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_141, [1, 0]);  convert_element_type_141 = None
        addmm_16: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_140, view_119, permute_46);  convert_element_type_140 = None
        view_120: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [512, 49, 1536]);  addmm_16 = None
        view_121: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_120, [512, 49, 3, 16, -1]);  view_120 = None
        permute_47: "bf16[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_121, [2, 0, 3, 1, 4]);  view_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_4 = torch.ops.aten.unbind.int(permute_47);  permute_47 = None
        getitem_36: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_4[0]
        getitem_37: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_4[1]
        getitem_38: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_4[2];  unbind_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_46: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_36, 0.1767766952966369);  getitem_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_48: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_37, [0, 1, 3, 2]);  getitem_37 = None
        expand_16: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_46, [512, 16, 49, 32]);  mul_46 = None
        clone_47: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_122: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_47, [8192, 49, 32]);  clone_47 = None
        expand_17: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_48, [512, 16, 32, 49]);  permute_48 = None
        clone_48: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_123: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_48, [8192, 32, 49]);  clone_48 = None
        constant_pad_nd_default_74: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_122, [0, 0, 0, 7, 0, 0])
        constant_pad_nd_default_75: "bf16[8192, 32, 56][1792, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_123, [0, 7, 0, 0, 0, 0])
        bmm_default_37: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_74, constant_pad_nd_default_75);  constant_pad_nd_default_74 = constant_pad_nd_default_75 = None
        slice_tensor_54: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_37, 1, 0, -7)
        slice_tensor_55: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_54, 2, 0, -7);  slice_tensor_54 = None
        view_124: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_55, [512, 16, 49, 49]);  slice_tensor_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_125: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_75, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_12: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_74, [view_125]);  view_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_126: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_12, [49, 49, -1]);  index_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_49: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_126, [2, 0, 1]);  view_126 = None
        clone_49: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_8: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_49, 0);  clone_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_50: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_124, unsqueeze_8);  view_124 = unsqueeze_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_4: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_50, [-1], True)
        sub_16: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_50, amax_4);  add_50 = None
        exp_4: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_5: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_10: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        convert_element_type_148: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None
        expand_18: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_148, [512, 16, 49, 49]);  convert_element_type_148 = None
        view_127: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_18, [8192, 49, 49]);  expand_18 = None
        expand_19: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_38, [512, 16, 49, 32]);  getitem_38 = None
        clone_51: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_128: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_51, [8192, 49, 32]);  clone_51 = None
        constant_pad_nd_default_72: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_127, [0, 7, 0, 7, 0, 0])
        constant_pad_nd_default_73: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_128, [0, 0, 0, 7, 0, 0])
        bmm_default_36: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_72, constant_pad_nd_default_73);  constant_pad_nd_default_72 = constant_pad_nd_default_73 = None
        slice_tensor_53: "bf16[8192, 49, 32][1792, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_36, 1, 0, -7);  bmm_default_36 = None
        view_129: "bf16[512, 16, 49, 32][28672, 1792, 32, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_53, [512, 16, 49, 32]);  slice_tensor_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_50: "bf16[512, 49, 16, 32][28672, 32, 1792, 1]cuda:0" = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None
        clone_52: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_50, memory_format = torch.contiguous_format);  permute_50 = None
        view_130: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_52, [512, 49, 512]);  clone_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        convert_element_type_151: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_77, torch.bfloat16);  primals_77 = None
        convert_element_type_152: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_76, torch.bfloat16);  primals_76 = None
        view_131: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_130, [25088, 512]);  view_130 = None
        permute_51: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_152, [1, 0]);  convert_element_type_152 = None
        addmm_17: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_151, view_131, permute_51);  convert_element_type_151 = None
        view_132: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [512, 49, 512]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_133: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_132, [-1, 7, 7, 512]);  view_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_134: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_133, [-1, 2, 2, 7, 7, 512]);  view_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_52: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_134, [0, 1, 3, 2, 4, 5]);  view_134 = None
        clone_54: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
        view_135: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_54, [-1, 14, 14, 512]);  clone_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_6: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6)
        inductor_random_default_39: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_6, 'rand');  inductor_lookup_seed_default_6 = None
        lt_6: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_39, 0.9826086945831776);  inductor_random_default_39 = None
        convert_element_type_156: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_6, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_11: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_156, 0.9826086945831776);  convert_element_type_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_47: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_135, div_11);  view_135 = div_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_51: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_115, mul_47);  view_115 = mul_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_136: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_51, [128, -1, 512]);  add_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        convert_element_type_157: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_136, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_157, [2], correction = 0, keepdim = True)
        getitem_39: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_12[0]
        getitem_40: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        add_52: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_39, 1e-05);  getitem_39 = None
        rsqrt_12: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        sub_17: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_157, getitem_40);  convert_element_type_157 = None
        mul_48: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_12);  sub_17 = None
        mul_49: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_48, primals_78);  mul_48 = None
        add_53: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_49, primals_79);  mul_49 = primals_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_158: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_81, torch.bfloat16);  primals_81 = None
        convert_element_type_159: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_80, torch.bfloat16);  primals_80 = None
        convert_element_type_160: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_53, torch.bfloat16);  add_53 = None
        view_137: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_160, [25088, 512]);  convert_element_type_160 = None
        permute_53: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_159, [1, 0]);  convert_element_type_159 = None
        addmm_18: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_158, view_137, permute_53);  convert_element_type_158 = None
        view_138: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_164: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_138, torch.float32);  view_138 = None
        mul_50: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_164, 0.5)
        mul_51: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_164, 0.7071067811865476);  convert_element_type_164 = None
        erf_4: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
        add_54: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_52: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_50, add_54);  mul_50 = add_54 = None
        convert_element_type_165: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_52, torch.bfloat16);  mul_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_166: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_83, torch.bfloat16);  primals_83 = None
        convert_element_type_167: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_82, torch.bfloat16);  primals_82 = None
        view_139: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_165, [25088, 2048]);  convert_element_type_165 = None
        permute_54: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_167, [1, 0]);  convert_element_type_167 = None
        addmm_19: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_166, view_139, permute_54);  convert_element_type_166 = None
        view_140: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [128, 196, 512]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_7: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 7)
        inductor_random_default_38: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_7, 'rand');  inductor_lookup_seed_default_7 = None
        lt_7: "b8[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_38, 0.9826086945831776);  inductor_random_default_38 = None
        convert_element_type_171: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_7, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_12: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_171, 0.9826086945831776);  convert_element_type_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_53: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_140, div_12);  view_140 = div_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_55: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_136, mul_53);  mul_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_141: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_55, [128, 14, 14, 512]);  add_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        convert_element_type_172: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_141, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_172, [3], correction = 0, keepdim = True)
        getitem_41: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_13[0]
        getitem_42: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        add_56: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_41, 1e-05);  getitem_41 = None
        rsqrt_13: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        sub_18: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_172, getitem_42);  convert_element_type_172 = None
        mul_54: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_13);  sub_18 = None
        mul_55: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_54, primals_84);  mul_54 = None
        add_57: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_55, primals_85);  mul_55 = primals_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota_8: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_58: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_8, 3)
        fmod_8: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_58, 14);  add_58 = None
        index_13: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(add_57, [None, fmod_8]);  add_57 = None
        index_14: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_13, [None, None, fmod_8]);  index_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_142: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(index_14, [128, 2, 7, 2, 7, 512]);  index_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_55: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_142, [0, 1, 3, 2, 4, 5]);  view_142 = None
        clone_57: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_55, memory_format = torch.contiguous_format);  permute_55 = None
        view_143: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_57, [-1, 7, 7, 512]);  clone_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_144: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_143, [-1, 49, 512]);  view_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        convert_element_type_173: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_88, torch.bfloat16);  primals_88 = None
        convert_element_type_174: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_87, torch.bfloat16);  primals_87 = None
        convert_element_type_175: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_144, torch.bfloat16);  view_144 = None
        view_145: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_175, [25088, 512]);  convert_element_type_175 = None
        permute_56: "bf16[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_174, [1, 0]);  convert_element_type_174 = None
        addmm_20: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_173, view_145, permute_56);  convert_element_type_173 = None
        view_146: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [512, 49, 1536]);  addmm_20 = None
        view_147: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_146, [512, 49, 3, 16, -1]);  view_146 = None
        permute_57: "bf16[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_147, [2, 0, 3, 1, 4]);  view_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_5 = torch.ops.aten.unbind.int(permute_57);  permute_57 = None
        getitem_43: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_5[0]
        getitem_44: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_5[1]
        getitem_45: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_5[2];  unbind_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_56: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_43, 0.1767766952966369);  getitem_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_58: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_44, [0, 1, 3, 2]);  getitem_44 = None
        expand_20: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_56, [512, 16, 49, 32]);  mul_56 = None
        clone_58: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_148: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_58, [8192, 49, 32]);  clone_58 = None
        expand_21: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_58, [512, 16, 32, 49]);  permute_58 = None
        clone_59: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_149: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_59, [8192, 32, 49]);  clone_59 = None
        constant_pad_nd_default_70: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_148, [0, 0, 0, 7, 0, 0])
        constant_pad_nd_default_71: "bf16[8192, 32, 56][1792, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_149, [0, 7, 0, 0, 0, 0])
        bmm_default_35: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_70, constant_pad_nd_default_71);  constant_pad_nd_default_70 = constant_pad_nd_default_71 = None
        slice_tensor_51: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_35, 1, 0, -7)
        slice_tensor_52: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_51, 2, 0, -7);  slice_tensor_51 = None
        view_150: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_52, [512, 16, 49, 49]);  slice_tensor_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_151: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_90, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_15: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_89, [view_151]);  view_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_152: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_15, [49, 49, -1]);  index_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_59: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_152, [2, 0, 1]);  view_152 = None
        clone_60: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_9: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_60, 0);  clone_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_60: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_150, unsqueeze_9);  view_150 = unsqueeze_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_153: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_60, [-1, 4, 16, 49, 49]);  add_60 = None
        unsqueeze_10: "f32[4, 1, 49, 49][2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_86, 1)
        unsqueeze_11: "f32[1, 4, 1, 49, 49][9604, 2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 0);  unsqueeze_10 = None
        add_61: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_153, unsqueeze_11);  view_153 = unsqueeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_154: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_61, [-1, 16, 49, 49]);  add_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_5: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(view_154, [-1], True)
        sub_19: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_154, amax_5);  view_154 = None
        exp_5: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_6: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_13: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        convert_element_type_181: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None
        expand_22: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_181, [512, 16, 49, 49]);  convert_element_type_181 = None
        view_155: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_22, [8192, 49, 49]);  expand_22 = None
        expand_23: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_45, [512, 16, 49, 32]);  getitem_45 = None
        clone_62: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_156: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_62, [8192, 49, 32]);  clone_62 = None
        constant_pad_nd_default_68: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_155, [0, 7, 0, 7, 0, 0])
        constant_pad_nd_default_69: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_156, [0, 0, 0, 7, 0, 0])
        bmm_default_34: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_68, constant_pad_nd_default_69);  constant_pad_nd_default_68 = constant_pad_nd_default_69 = None
        slice_tensor_50: "bf16[8192, 49, 32][1792, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_34, 1, 0, -7);  bmm_default_34 = None
        view_157: "bf16[512, 16, 49, 32][28672, 1792, 32, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_50, [512, 16, 49, 32]);  slice_tensor_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_60: "bf16[512, 49, 16, 32][28672, 32, 1792, 1]cuda:0" = torch.ops.aten.permute.default(view_157, [0, 2, 1, 3]);  view_157 = None
        clone_63: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
        view_158: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_63, [512, 49, 512]);  clone_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        convert_element_type_184: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_92, torch.bfloat16);  primals_92 = None
        convert_element_type_185: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_91, torch.bfloat16);  primals_91 = None
        view_159: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_158, [25088, 512]);  view_158 = None
        permute_61: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_185, [1, 0]);  convert_element_type_185 = None
        addmm_21: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_184, view_159, permute_61);  convert_element_type_184 = None
        view_160: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [512, 49, 512]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_161: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_160, [-1, 7, 7, 512]);  view_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_162: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_161, [-1, 2, 2, 7, 7, 512]);  view_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_62: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_162, [0, 1, 3, 2, 4, 5]);  view_162 = None
        clone_65: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_163: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_65, [-1, 14, 14, 512]);  clone_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        add_62: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_8, 11);  iota_8 = None
        fmod_10: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_62, 14);  add_62 = None
        index_16: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_163, [None, fmod_10]);  view_163 = None
        index_17: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_16, [None, None, fmod_10]);  index_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_8: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 8)
        inductor_random_default_37: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_8, 'rand');  inductor_lookup_seed_default_8 = None
        lt_8: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_37, 0.9782608672976494);  inductor_random_default_37 = None
        convert_element_type_189: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_8, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_14: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_189, 0.9782608672976494);  convert_element_type_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_57: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(index_17, div_14);  index_17 = div_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_64: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_141, mul_57);  mul_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_164: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_64, [128, -1, 512]);  add_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        convert_element_type_190: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_164, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_190, [2], correction = 0, keepdim = True)
        getitem_46: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_14[0]
        getitem_47: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        add_65: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_14: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        sub_20: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_190, getitem_47);  convert_element_type_190 = None
        mul_58: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_14);  sub_20 = None
        mul_59: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_58, primals_93);  mul_58 = None
        add_66: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_59, primals_94);  mul_59 = primals_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_191: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_96, torch.bfloat16);  primals_96 = None
        convert_element_type_192: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_95, torch.bfloat16);  primals_95 = None
        convert_element_type_193: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_66, torch.bfloat16);  add_66 = None
        view_165: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_193, [25088, 512]);  convert_element_type_193 = None
        permute_63: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_192, [1, 0]);  convert_element_type_192 = None
        addmm_22: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_191, view_165, permute_63);  convert_element_type_191 = None
        view_166: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_197: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_166, torch.float32);  view_166 = None
        mul_60: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_197, 0.5)
        mul_61: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_197, 0.7071067811865476);  convert_element_type_197 = None
        erf_5: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
        add_67: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_62: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_60, add_67);  mul_60 = add_67 = None
        convert_element_type_198: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_62, torch.bfloat16);  mul_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_199: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_98, torch.bfloat16);  primals_98 = None
        convert_element_type_200: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_97, torch.bfloat16);  primals_97 = None
        view_167: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_198, [25088, 2048]);  convert_element_type_198 = None
        permute_64: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_200, [1, 0]);  convert_element_type_200 = None
        addmm_23: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_199, view_167, permute_64);  convert_element_type_199 = None
        view_168: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [128, 196, 512]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_9: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 9)
        inductor_random_default_36: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_9, 'rand');  inductor_lookup_seed_default_9 = None
        lt_9: "b8[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_36, 0.9782608672976494);  inductor_random_default_36 = None
        convert_element_type_204: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_9, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_15: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_204, 0.9782608672976494);  convert_element_type_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_63: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_168, div_15);  view_168 = div_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_68: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_164, mul_63);  mul_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_169: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_68, [128, 14, 14, 512]);  add_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        convert_element_type_205: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_169, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_205, [3], correction = 0, keepdim = True)
        getitem_48: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_15[0]
        getitem_49: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        add_69: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_15: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_21: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_205, getitem_49);  convert_element_type_205 = None
        mul_64: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_15);  sub_21 = None
        mul_65: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_64, primals_99);  mul_64 = None
        add_70: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_65, primals_100);  mul_65 = primals_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_170: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_70, [128, 2, 7, 2, 7, 512]);  add_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_65: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_170, [0, 1, 3, 2, 4, 5]);  view_170 = None
        clone_68: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_65, memory_format = torch.contiguous_format);  permute_65 = None
        view_171: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_68, [-1, 7, 7, 512]);  clone_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_172: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_171, [-1, 49, 512]);  view_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        convert_element_type_206: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_102, torch.bfloat16);  primals_102 = None
        convert_element_type_207: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_101, torch.bfloat16);  primals_101 = None
        convert_element_type_208: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_172, torch.bfloat16);  view_172 = None
        view_173: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_208, [25088, 512]);  convert_element_type_208 = None
        permute_66: "bf16[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_207, [1, 0]);  convert_element_type_207 = None
        addmm_24: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_206, view_173, permute_66);  convert_element_type_206 = None
        view_174: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_24, [512, 49, 1536]);  addmm_24 = None
        view_175: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_174, [512, 49, 3, 16, -1]);  view_174 = None
        permute_67: "bf16[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_175, [2, 0, 3, 1, 4]);  view_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_6 = torch.ops.aten.unbind.int(permute_67);  permute_67 = None
        getitem_50: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_6[0]
        getitem_51: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_6[1]
        getitem_52: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_6[2];  unbind_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_66: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_50, 0.1767766952966369);  getitem_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_68: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_51, [0, 1, 3, 2]);  getitem_51 = None
        expand_24: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_66, [512, 16, 49, 32]);  mul_66 = None
        clone_69: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_176: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_69, [8192, 49, 32]);  clone_69 = None
        expand_25: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_68, [512, 16, 32, 49]);  permute_68 = None
        clone_70: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_177: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_70, [8192, 32, 49]);  clone_70 = None
        constant_pad_nd_default_66: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_176, [0, 0, 0, 7, 0, 0])
        constant_pad_nd_default_67: "bf16[8192, 32, 56][1792, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_177, [0, 7, 0, 0, 0, 0])
        bmm_default_33: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_66, constant_pad_nd_default_67);  constant_pad_nd_default_66 = constant_pad_nd_default_67 = None
        slice_tensor_48: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_33, 1, 0, -7)
        slice_tensor_49: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_48, 2, 0, -7);  slice_tensor_48 = None
        view_178: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_49, [512, 16, 49, 49]);  slice_tensor_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_179: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_104, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_18: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_103, [view_179]);  view_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_180: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_18, [49, 49, -1]);  index_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_69: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_180, [2, 0, 1]);  view_180 = None
        clone_71: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_69, memory_format = torch.contiguous_format);  permute_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_12: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_71, 0);  clone_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_71: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_178, unsqueeze_12);  view_178 = unsqueeze_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_6: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_71, [-1], True)
        sub_22: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_71, amax_6);  add_71 = None
        exp_6: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_7: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_16: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        convert_element_type_214: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_16, torch.bfloat16);  div_16 = None
        expand_26: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_214, [512, 16, 49, 49]);  convert_element_type_214 = None
        view_181: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_26, [8192, 49, 49]);  expand_26 = None
        expand_27: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_52, [512, 16, 49, 32]);  getitem_52 = None
        clone_73: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_182: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_73, [8192, 49, 32]);  clone_73 = None
        constant_pad_nd_default_64: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_181, [0, 7, 0, 7, 0, 0])
        constant_pad_nd_default_65: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_182, [0, 0, 0, 7, 0, 0])
        bmm_default_32: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_64, constant_pad_nd_default_65);  constant_pad_nd_default_64 = constant_pad_nd_default_65 = None
        slice_tensor_47: "bf16[8192, 49, 32][1792, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_32, 1, 0, -7);  bmm_default_32 = None
        view_183: "bf16[512, 16, 49, 32][28672, 1792, 32, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_47, [512, 16, 49, 32]);  slice_tensor_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_70: "bf16[512, 49, 16, 32][28672, 32, 1792, 1]cuda:0" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None
        clone_74: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
        view_184: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_74, [512, 49, 512]);  clone_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        convert_element_type_217: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_106, torch.bfloat16);  primals_106 = None
        convert_element_type_218: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_105, torch.bfloat16);  primals_105 = None
        view_185: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_184, [25088, 512]);  view_184 = None
        permute_71: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_218, [1, 0]);  convert_element_type_218 = None
        addmm_25: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_217, view_185, permute_71);  convert_element_type_217 = None
        view_186: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [512, 49, 512]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_187: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_186, [-1, 7, 7, 512]);  view_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_188: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_187, [-1, 2, 2, 7, 7, 512]);  view_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_72: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_188, [0, 1, 3, 2, 4, 5]);  view_188 = None
        clone_76: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format);  permute_72 = None
        view_189: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_76, [-1, 14, 14, 512]);  clone_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_10: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 10)
        inductor_random_default_35: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_10, 'rand');  inductor_lookup_seed_default_10 = None
        lt_10: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_35, 0.9739130418747663);  inductor_random_default_35 = None
        convert_element_type_222: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_10, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_17: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_222, 0.9739130418747663);  convert_element_type_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_67: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_189, div_17);  view_189 = div_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_72: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_169, mul_67);  mul_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_190: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_72, [128, -1, 512]);  add_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        convert_element_type_223: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_190, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_223, [2], correction = 0, keepdim = True)
        getitem_53: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_16[0]
        getitem_54: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        add_73: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_53, 1e-05);  getitem_53 = None
        rsqrt_16: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        sub_23: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_223, getitem_54);  convert_element_type_223 = None
        mul_68: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_16);  sub_23 = None
        mul_69: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, primals_107);  mul_68 = None
        add_74: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_69, primals_108);  mul_69 = primals_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_224: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_110, torch.bfloat16);  primals_110 = None
        convert_element_type_225: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_109, torch.bfloat16);  primals_109 = None
        convert_element_type_226: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_74, torch.bfloat16);  add_74 = None
        view_191: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_226, [25088, 512]);  convert_element_type_226 = None
        permute_73: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_225, [1, 0]);  convert_element_type_225 = None
        addmm_26: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_224, view_191, permute_73);  convert_element_type_224 = None
        view_192: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_230: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_192, torch.float32);  view_192 = None
        mul_70: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_230, 0.5)
        mul_71: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_230, 0.7071067811865476);  convert_element_type_230 = None
        erf_6: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_71);  mul_71 = None
        add_75: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_72: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, add_75);  mul_70 = add_75 = None
        convert_element_type_231: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_72, torch.bfloat16);  mul_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_232: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_112, torch.bfloat16);  primals_112 = None
        convert_element_type_233: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_111, torch.bfloat16);  primals_111 = None
        view_193: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_231, [25088, 2048]);  convert_element_type_231 = None
        permute_74: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_233, [1, 0]);  convert_element_type_233 = None
        addmm_27: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_232, view_193, permute_74);  convert_element_type_232 = None
        view_194: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_27, [128, 196, 512]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_11: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 11)
        inductor_random_default_34: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_11, 'rand');  inductor_lookup_seed_default_11 = None
        lt_11: "b8[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_34, 0.9739130418747663);  inductor_random_default_34 = None
        convert_element_type_237: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_11, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_18: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_237, 0.9739130418747663);  convert_element_type_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_73: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_194, div_18);  view_194 = div_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_76: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_190, mul_73);  mul_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_195: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_76, [128, 14, 14, 512]);  add_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        convert_element_type_238: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_195, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_238, [3], correction = 0, keepdim = True)
        getitem_55: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_17[0]
        getitem_56: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        add_77: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_55, 1e-05);  getitem_55 = None
        rsqrt_17: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        sub_24: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_238, getitem_56);  convert_element_type_238 = None
        mul_74: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_17);  sub_24 = None
        mul_75: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_74, primals_113);  mul_74 = None
        add_78: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_75, primals_114);  mul_75 = primals_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_19: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(add_78, [None, fmod_8]);  add_78 = None
        index_20: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_19, [None, None, fmod_8]);  index_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_196: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(index_20, [128, 2, 7, 2, 7, 512]);  index_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_75: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_196, [0, 1, 3, 2, 4, 5]);  view_196 = None
        clone_79: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_75, memory_format = torch.contiguous_format);  permute_75 = None
        view_197: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_79, [-1, 7, 7, 512]);  clone_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_198: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_197, [-1, 49, 512]);  view_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        convert_element_type_239: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_117, torch.bfloat16);  primals_117 = None
        convert_element_type_240: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_116, torch.bfloat16);  primals_116 = None
        convert_element_type_241: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_198, torch.bfloat16);  view_198 = None
        view_199: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_241, [25088, 512]);  convert_element_type_241 = None
        permute_76: "bf16[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_240, [1, 0]);  convert_element_type_240 = None
        addmm_28: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_239, view_199, permute_76);  convert_element_type_239 = None
        view_200: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [512, 49, 1536]);  addmm_28 = None
        view_201: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_200, [512, 49, 3, 16, -1]);  view_200 = None
        permute_77: "bf16[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_201, [2, 0, 3, 1, 4]);  view_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_7 = torch.ops.aten.unbind.int(permute_77);  permute_77 = None
        getitem_57: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_7[0]
        getitem_58: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_7[1]
        getitem_59: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_7[2];  unbind_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_76: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_57, 0.1767766952966369);  getitem_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_78: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_58, [0, 1, 3, 2]);  getitem_58 = None
        expand_28: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_76, [512, 16, 49, 32]);  mul_76 = None
        clone_80: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_202: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_80, [8192, 49, 32]);  clone_80 = None
        expand_29: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_78, [512, 16, 32, 49]);  permute_78 = None
        clone_81: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_203: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_81, [8192, 32, 49]);  clone_81 = None
        constant_pad_nd_default_62: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_202, [0, 0, 0, 7, 0, 0])
        constant_pad_nd_default_63: "bf16[8192, 32, 56][1792, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_203, [0, 7, 0, 0, 0, 0])
        bmm_default_31: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_62, constant_pad_nd_default_63);  constant_pad_nd_default_62 = constant_pad_nd_default_63 = None
        slice_tensor_45: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_31, 1, 0, -7)
        slice_tensor_46: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_45, 2, 0, -7);  slice_tensor_45 = None
        view_204: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_46, [512, 16, 49, 49]);  slice_tensor_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_205: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_119, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_21: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_118, [view_205]);  view_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_206: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_21, [49, 49, -1]);  index_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_79: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_206, [2, 0, 1]);  view_206 = None
        clone_82: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_13: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_82, 0);  clone_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_81: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_204, unsqueeze_13);  view_204 = unsqueeze_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_207: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_81, [-1, 4, 16, 49, 49]);  add_81 = None
        unsqueeze_14: "f32[4, 1, 49, 49][2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_115, 1)
        unsqueeze_15: "f32[1, 4, 1, 49, 49][9604, 2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_14, 0);  unsqueeze_14 = None
        add_82: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_207, unsqueeze_15);  view_207 = unsqueeze_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_208: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_82, [-1, 16, 49, 49]);  add_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_7: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(view_208, [-1], True)
        sub_25: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_208, amax_7);  view_208 = None
        exp_7: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_8: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_19: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        convert_element_type_247: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16);  div_19 = None
        expand_30: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_247, [512, 16, 49, 49]);  convert_element_type_247 = None
        view_209: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_30, [8192, 49, 49]);  expand_30 = None
        expand_31: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_59, [512, 16, 49, 32]);  getitem_59 = None
        clone_84: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_210: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_84, [8192, 49, 32]);  clone_84 = None
        constant_pad_nd_default_60: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_209, [0, 7, 0, 7, 0, 0])
        constant_pad_nd_default_61: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_210, [0, 0, 0, 7, 0, 0])
        bmm_default_30: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_60, constant_pad_nd_default_61);  constant_pad_nd_default_60 = constant_pad_nd_default_61 = None
        slice_tensor_44: "bf16[8192, 49, 32][1792, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_30, 1, 0, -7);  bmm_default_30 = None
        view_211: "bf16[512, 16, 49, 32][28672, 1792, 32, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_44, [512, 16, 49, 32]);  slice_tensor_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_80: "bf16[512, 49, 16, 32][28672, 32, 1792, 1]cuda:0" = torch.ops.aten.permute.default(view_211, [0, 2, 1, 3]);  view_211 = None
        clone_85: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
        view_212: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_85, [512, 49, 512]);  clone_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        convert_element_type_250: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_121, torch.bfloat16);  primals_121 = None
        convert_element_type_251: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_120, torch.bfloat16);  primals_120 = None
        view_213: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_212, [25088, 512]);  view_212 = None
        permute_81: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_251, [1, 0]);  convert_element_type_251 = None
        addmm_29: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_250, view_213, permute_81);  convert_element_type_250 = None
        view_214: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_29, [512, 49, 512]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_215: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_214, [-1, 7, 7, 512]);  view_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_216: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_215, [-1, 2, 2, 7, 7, 512]);  view_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_82: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_216, [0, 1, 3, 2, 4, 5]);  view_216 = None
        clone_87: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        view_217: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_87, [-1, 14, 14, 512]);  clone_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_22: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_217, [None, fmod_10]);  view_217 = None
        index_23: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_22, [None, None, fmod_10]);  index_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_12: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 12)
        inductor_random_default_33: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_12, 'rand');  inductor_lookup_seed_default_12 = None
        lt_12: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_33, 0.9695652164518833);  inductor_random_default_33 = None
        convert_element_type_255: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_12, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_20: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_255, 0.9695652164518833);  convert_element_type_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_77: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(index_23, div_20);  index_23 = div_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_85: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_195, mul_77);  mul_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_218: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_85, [128, -1, 512]);  add_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        convert_element_type_256: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_218, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_256, [2], correction = 0, keepdim = True)
        getitem_60: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_18[0]
        getitem_61: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        add_86: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
        rsqrt_18: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        sub_26: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_256, getitem_61);  convert_element_type_256 = None
        mul_78: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_18);  sub_26 = None
        mul_79: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_78, primals_122);  mul_78 = None
        add_87: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_79, primals_123);  mul_79 = primals_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_257: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_125, torch.bfloat16);  primals_125 = None
        convert_element_type_258: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_124, torch.bfloat16);  primals_124 = None
        convert_element_type_259: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_87, torch.bfloat16);  add_87 = None
        view_219: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_259, [25088, 512]);  convert_element_type_259 = None
        permute_83: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_258, [1, 0]);  convert_element_type_258 = None
        addmm_30: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_257, view_219, permute_83);  convert_element_type_257 = None
        view_220: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_263: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_220, torch.float32);  view_220 = None
        mul_80: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_263, 0.5)
        mul_81: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_263, 0.7071067811865476);  convert_element_type_263 = None
        erf_7: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_81);  mul_81 = None
        add_88: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_82: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_80, add_88);  mul_80 = add_88 = None
        convert_element_type_264: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_82, torch.bfloat16);  mul_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_265: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_127, torch.bfloat16);  primals_127 = None
        convert_element_type_266: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_126, torch.bfloat16);  primals_126 = None
        view_221: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_264, [25088, 2048]);  convert_element_type_264 = None
        permute_84: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_266, [1, 0]);  convert_element_type_266 = None
        addmm_31: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_265, view_221, permute_84);  convert_element_type_265 = None
        view_222: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [128, 196, 512]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_13: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 13)
        inductor_random_default_32: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_13, 'rand');  inductor_lookup_seed_default_13 = None
        lt_13: "b8[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_32, 0.9695652164518833);  inductor_random_default_32 = None
        convert_element_type_270: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_13, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_21: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_270, 0.9695652164518833);  convert_element_type_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_83: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_222, div_21);  view_222 = div_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_89: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_218, mul_83);  mul_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_223: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_89, [128, 14, 14, 512]);  add_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        convert_element_type_271: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_223, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_271, [3], correction = 0, keepdim = True)
        getitem_62: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_19[0]
        getitem_63: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        add_90: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
        rsqrt_19: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        sub_27: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_271, getitem_63);  convert_element_type_271 = None
        mul_84: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_19);  sub_27 = None
        mul_85: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_84, primals_128);  mul_84 = None
        add_91: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_85, primals_129);  mul_85 = primals_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_224: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_91, [128, 2, 7, 2, 7, 512]);  add_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_85: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_224, [0, 1, 3, 2, 4, 5]);  view_224 = None
        clone_90: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None
        view_225: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_90, [-1, 7, 7, 512]);  clone_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_226: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_225, [-1, 49, 512]);  view_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        convert_element_type_272: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_131, torch.bfloat16);  primals_131 = None
        convert_element_type_273: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_130, torch.bfloat16);  primals_130 = None
        convert_element_type_274: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_226, torch.bfloat16);  view_226 = None
        view_227: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_274, [25088, 512]);  convert_element_type_274 = None
        permute_86: "bf16[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_273, [1, 0]);  convert_element_type_273 = None
        addmm_32: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_272, view_227, permute_86);  convert_element_type_272 = None
        view_228: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_32, [512, 49, 1536]);  addmm_32 = None
        view_229: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_228, [512, 49, 3, 16, -1]);  view_228 = None
        permute_87: "bf16[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_229, [2, 0, 3, 1, 4]);  view_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_8 = torch.ops.aten.unbind.int(permute_87);  permute_87 = None
        getitem_64: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_8[0]
        getitem_65: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_8[1]
        getitem_66: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_8[2];  unbind_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_86: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_64, 0.1767766952966369);  getitem_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_88: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_65, [0, 1, 3, 2]);  getitem_65 = None
        expand_32: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_86, [512, 16, 49, 32]);  mul_86 = None
        clone_91: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_230: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_91, [8192, 49, 32]);  clone_91 = None
        expand_33: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_88, [512, 16, 32, 49]);  permute_88 = None
        clone_92: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_231: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_92, [8192, 32, 49]);  clone_92 = None
        constant_pad_nd_default_58: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_230, [0, 0, 0, 7, 0, 0])
        constant_pad_nd_default_59: "bf16[8192, 32, 56][1792, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_231, [0, 7, 0, 0, 0, 0])
        bmm_default_29: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_58, constant_pad_nd_default_59);  constant_pad_nd_default_58 = constant_pad_nd_default_59 = None
        slice_tensor_42: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_29, 1, 0, -7)
        slice_tensor_43: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_42, 2, 0, -7);  slice_tensor_42 = None
        view_232: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_43, [512, 16, 49, 49]);  slice_tensor_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_233: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_133, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_24: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_132, [view_233]);  view_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_234: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_24, [49, 49, -1]);  index_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_89: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_234, [2, 0, 1]);  view_234 = None
        clone_93: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_89, memory_format = torch.contiguous_format);  permute_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_16: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_93, 0);  clone_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_92: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_232, unsqueeze_16);  view_232 = unsqueeze_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_8: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_92, [-1], True)
        sub_28: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_92, amax_8);  add_92 = None
        exp_8: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_9: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_22: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        convert_element_type_280: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_22, torch.bfloat16);  div_22 = None
        expand_34: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_280, [512, 16, 49, 49]);  convert_element_type_280 = None
        view_235: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_34, [8192, 49, 49]);  expand_34 = None
        expand_35: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_66, [512, 16, 49, 32]);  getitem_66 = None
        clone_95: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_236: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_95, [8192, 49, 32]);  clone_95 = None
        constant_pad_nd_default_56: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_235, [0, 7, 0, 7, 0, 0])
        constant_pad_nd_default_57: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_236, [0, 0, 0, 7, 0, 0])
        bmm_default_28: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_56, constant_pad_nd_default_57);  constant_pad_nd_default_56 = constant_pad_nd_default_57 = None
        slice_tensor_41: "bf16[8192, 49, 32][1792, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_28, 1, 0, -7);  bmm_default_28 = None
        view_237: "bf16[512, 16, 49, 32][28672, 1792, 32, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_41, [512, 16, 49, 32]);  slice_tensor_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_90: "bf16[512, 49, 16, 32][28672, 32, 1792, 1]cuda:0" = torch.ops.aten.permute.default(view_237, [0, 2, 1, 3]);  view_237 = None
        clone_96: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
        view_238: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_96, [512, 49, 512]);  clone_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        convert_element_type_283: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_135, torch.bfloat16);  primals_135 = None
        convert_element_type_284: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_134, torch.bfloat16);  primals_134 = None
        view_239: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_238, [25088, 512]);  view_238 = None
        permute_91: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_284, [1, 0]);  convert_element_type_284 = None
        addmm_33: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_283, view_239, permute_91);  convert_element_type_283 = None
        view_240: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [512, 49, 512]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_241: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_240, [-1, 7, 7, 512]);  view_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_242: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_241, [-1, 2, 2, 7, 7, 512]);  view_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_92: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_242, [0, 1, 3, 2, 4, 5]);  view_242 = None
        clone_98: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
        view_243: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_98, [-1, 14, 14, 512]);  clone_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_14: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 14)
        inductor_random_default_31: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_14, 'rand');  inductor_lookup_seed_default_14 = None
        lt_14: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_31, 0.9652173891663551);  inductor_random_default_31 = None
        convert_element_type_288: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_14, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_23: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_288, 0.9652173891663551);  convert_element_type_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_87: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_243, div_23);  view_243 = div_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_93: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_223, mul_87);  mul_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_244: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_93, [128, -1, 512]);  add_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        convert_element_type_289: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_244, torch.float32)
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_289, [2], correction = 0, keepdim = True)
        getitem_67: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_20[0]
        getitem_68: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None
        add_94: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_67, 1e-05);  getitem_67 = None
        rsqrt_20: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        sub_29: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_289, getitem_68);  convert_element_type_289 = None
        mul_88: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_20);  sub_29 = None
        mul_89: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_88, primals_136);  mul_88 = None
        add_95: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_89, primals_137);  mul_89 = primals_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_290: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_139, torch.bfloat16);  primals_139 = None
        convert_element_type_291: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_138, torch.bfloat16);  primals_138 = None
        convert_element_type_292: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_95, torch.bfloat16);  add_95 = None
        view_245: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_292, [25088, 512]);  convert_element_type_292 = None
        permute_93: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_291, [1, 0]);  convert_element_type_291 = None
        addmm_34: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_290, view_245, permute_93);  convert_element_type_290 = None
        view_246: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_296: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_246, torch.float32);  view_246 = None
        mul_90: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_296, 0.5)
        mul_91: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_296, 0.7071067811865476);  convert_element_type_296 = None
        erf_8: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_91);  mul_91 = None
        add_96: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_92: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_90, add_96);  mul_90 = add_96 = None
        convert_element_type_297: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_92, torch.bfloat16);  mul_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_298: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_141, torch.bfloat16);  primals_141 = None
        convert_element_type_299: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_140, torch.bfloat16);  primals_140 = None
        view_247: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_297, [25088, 2048]);  convert_element_type_297 = None
        permute_94: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_299, [1, 0]);  convert_element_type_299 = None
        addmm_35: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_298, view_247, permute_94);  convert_element_type_298 = None
        view_248: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_35, [128, 196, 512]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_15: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 15)
        inductor_random_default_30: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_15, 'rand');  inductor_lookup_seed_default_15 = None
        lt_15: "b8[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_30, 0.9652173891663551);  inductor_random_default_30 = None
        convert_element_type_303: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_15, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_24: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_303, 0.9652173891663551);  convert_element_type_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_93: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_248, div_24);  view_248 = div_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_97: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_244, mul_93);  mul_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_249: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_97, [128, 14, 14, 512]);  add_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        convert_element_type_304: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_249, torch.float32)
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_304, [3], correction = 0, keepdim = True)
        getitem_69: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_21[0]
        getitem_70: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        add_98: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_69, 1e-05);  getitem_69 = None
        rsqrt_21: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        sub_30: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_304, getitem_70);  convert_element_type_304 = None
        mul_94: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_21);  sub_30 = None
        mul_95: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_94, primals_142);  mul_94 = None
        add_99: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_95, primals_143);  mul_95 = primals_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_25: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(add_99, [None, fmod_8]);  add_99 = None
        index_26: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_25, [None, None, fmod_8]);  index_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_250: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(index_26, [128, 2, 7, 2, 7, 512]);  index_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_95: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_250, [0, 1, 3, 2, 4, 5]);  view_250 = None
        clone_101: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_251: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_101, [-1, 7, 7, 512]);  clone_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_252: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_251, [-1, 49, 512]);  view_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        convert_element_type_305: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_146, torch.bfloat16);  primals_146 = None
        convert_element_type_306: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_145, torch.bfloat16);  primals_145 = None
        convert_element_type_307: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_252, torch.bfloat16);  view_252 = None
        view_253: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_307, [25088, 512]);  convert_element_type_307 = None
        permute_96: "bf16[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_306, [1, 0]);  convert_element_type_306 = None
        addmm_36: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_305, view_253, permute_96);  convert_element_type_305 = None
        view_254: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_36, [512, 49, 1536]);  addmm_36 = None
        view_255: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_254, [512, 49, 3, 16, -1]);  view_254 = None
        permute_97: "bf16[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_255, [2, 0, 3, 1, 4]);  view_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_9 = torch.ops.aten.unbind.int(permute_97);  permute_97 = None
        getitem_71: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_9[0]
        getitem_72: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_9[1]
        getitem_73: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_9[2];  unbind_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_96: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_71, 0.1767766952966369);  getitem_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_98: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_72, [0, 1, 3, 2]);  getitem_72 = None
        expand_36: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_96, [512, 16, 49, 32]);  mul_96 = None
        clone_102: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_256: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_102, [8192, 49, 32]);  clone_102 = None
        expand_37: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_98, [512, 16, 32, 49]);  permute_98 = None
        clone_103: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_257: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_103, [8192, 32, 49]);  clone_103 = None
        constant_pad_nd_default_54: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_256, [0, 0, 0, 7, 0, 0])
        constant_pad_nd_default_55: "bf16[8192, 32, 56][1792, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_257, [0, 7, 0, 0, 0, 0])
        bmm_default_27: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_54, constant_pad_nd_default_55);  constant_pad_nd_default_54 = constant_pad_nd_default_55 = None
        slice_tensor_39: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_27, 1, 0, -7)
        slice_tensor_40: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_39, 2, 0, -7);  slice_tensor_39 = None
        view_258: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_40, [512, 16, 49, 49]);  slice_tensor_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_259: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_148, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_27: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_147, [view_259]);  view_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_260: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_27, [49, 49, -1]);  index_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_99: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_260, [2, 0, 1]);  view_260 = None
        clone_104: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_99, memory_format = torch.contiguous_format);  permute_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_17: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_104, 0);  clone_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_102: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_258, unsqueeze_17);  view_258 = unsqueeze_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_261: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_102, [-1, 4, 16, 49, 49]);  add_102 = None
        unsqueeze_18: "f32[4, 1, 49, 49][2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_144, 1)
        unsqueeze_19: "f32[1, 4, 1, 49, 49][9604, 2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_18, 0);  unsqueeze_18 = None
        add_103: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_261, unsqueeze_19);  view_261 = unsqueeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_262: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_103, [-1, 16, 49, 49]);  add_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_9: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(view_262, [-1], True)
        sub_31: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_262, amax_9);  view_262 = None
        exp_9: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_10: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_25: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        convert_element_type_313: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_25, torch.bfloat16);  div_25 = None
        expand_38: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_313, [512, 16, 49, 49]);  convert_element_type_313 = None
        view_263: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_38, [8192, 49, 49]);  expand_38 = None
        expand_39: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_73, [512, 16, 49, 32]);  getitem_73 = None
        clone_106: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_264: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_106, [8192, 49, 32]);  clone_106 = None
        constant_pad_nd_default_52: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_263, [0, 7, 0, 7, 0, 0])
        constant_pad_nd_default_53: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_264, [0, 0, 0, 7, 0, 0])
        bmm_default_26: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_52, constant_pad_nd_default_53);  constant_pad_nd_default_52 = constant_pad_nd_default_53 = None
        slice_tensor_38: "bf16[8192, 49, 32][1792, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_26, 1, 0, -7);  bmm_default_26 = None
        view_265: "bf16[512, 16, 49, 32][28672, 1792, 32, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_38, [512, 16, 49, 32]);  slice_tensor_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_100: "bf16[512, 49, 16, 32][28672, 32, 1792, 1]cuda:0" = torch.ops.aten.permute.default(view_265, [0, 2, 1, 3]);  view_265 = None
        clone_107: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_100, memory_format = torch.contiguous_format);  permute_100 = None
        view_266: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_107, [512, 49, 512]);  clone_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        convert_element_type_316: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_150, torch.bfloat16);  primals_150 = None
        convert_element_type_317: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_149, torch.bfloat16);  primals_149 = None
        view_267: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_266, [25088, 512]);  view_266 = None
        permute_101: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_317, [1, 0]);  convert_element_type_317 = None
        addmm_37: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_316, view_267, permute_101);  convert_element_type_316 = None
        view_268: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_37, [512, 49, 512]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_269: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_268, [-1, 7, 7, 512]);  view_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_270: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_269, [-1, 2, 2, 7, 7, 512]);  view_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_102: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_270, [0, 1, 3, 2, 4, 5]);  view_270 = None
        clone_109: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
        view_271: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_109, [-1, 14, 14, 512]);  clone_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_28: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_271, [None, fmod_10]);  view_271 = None
        index_29: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_28, [None, None, fmod_10]);  index_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_16: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 16)
        inductor_random_default_29: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_16, 'rand');  inductor_lookup_seed_default_16 = None
        lt_16: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_29, 0.960869561880827);  inductor_random_default_29 = None
        convert_element_type_321: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_16, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_26: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_321, 0.960869561880827);  convert_element_type_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_97: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(index_29, div_26);  index_29 = div_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_106: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_249, mul_97);  mul_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_272: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_106, [128, -1, 512]);  add_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        convert_element_type_322: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_272, torch.float32)
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_322, [2], correction = 0, keepdim = True)
        getitem_74: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_22[0]
        getitem_75: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None
        add_107: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
        rsqrt_22: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_107);  add_107 = None
        sub_32: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_322, getitem_75);  convert_element_type_322 = None
        mul_98: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_22);  sub_32 = None
        mul_99: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_98, primals_151);  mul_98 = None
        add_108: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_99, primals_152);  mul_99 = primals_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_323: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_154, torch.bfloat16);  primals_154 = None
        convert_element_type_324: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_153, torch.bfloat16);  primals_153 = None
        convert_element_type_325: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_108, torch.bfloat16);  add_108 = None
        view_273: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_325, [25088, 512]);  convert_element_type_325 = None
        permute_103: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_324, [1, 0]);  convert_element_type_324 = None
        addmm_38: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_323, view_273, permute_103);  convert_element_type_323 = None
        view_274: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_329: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_274, torch.float32);  view_274 = None
        mul_100: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_329, 0.5)
        mul_101: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_329, 0.7071067811865476);  convert_element_type_329 = None
        erf_9: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_101);  mul_101 = None
        add_109: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_102: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_100, add_109);  mul_100 = add_109 = None
        convert_element_type_330: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_102, torch.bfloat16);  mul_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_331: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_156, torch.bfloat16);  primals_156 = None
        convert_element_type_332: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_155, torch.bfloat16);  primals_155 = None
        view_275: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_330, [25088, 2048]);  convert_element_type_330 = None
        permute_104: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_332, [1, 0]);  convert_element_type_332 = None
        addmm_39: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_331, view_275, permute_104);  convert_element_type_331 = None
        view_276: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_39, [128, 196, 512]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_17: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 17)
        inductor_random_default_28: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_17, 'rand');  inductor_lookup_seed_default_17 = None
        lt_17: "b8[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_28, 0.960869561880827);  inductor_random_default_28 = None
        convert_element_type_336: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_17, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_27: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_336, 0.960869561880827);  convert_element_type_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_103: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_276, div_27);  view_276 = div_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_110: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_272, mul_103);  mul_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_277: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_110, [128, 14, 14, 512]);  add_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        convert_element_type_337: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_277, torch.float32)
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_337, [3], correction = 0, keepdim = True)
        getitem_76: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_23[0]
        getitem_77: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        add_111: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
        rsqrt_23: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
        sub_33: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_337, getitem_77);  convert_element_type_337 = None
        mul_104: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_23);  sub_33 = None
        mul_105: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_104, primals_157);  mul_104 = None
        add_112: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_105, primals_158);  mul_105 = primals_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_278: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_112, [128, 2, 7, 2, 7, 512]);  add_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_105: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_278, [0, 1, 3, 2, 4, 5]);  view_278 = None
        clone_112: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_105, memory_format = torch.contiguous_format);  permute_105 = None
        view_279: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_112, [-1, 7, 7, 512]);  clone_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_280: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_279, [-1, 49, 512]);  view_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        convert_element_type_338: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_160, torch.bfloat16);  primals_160 = None
        convert_element_type_339: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_159, torch.bfloat16);  primals_159 = None
        convert_element_type_340: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_280, torch.bfloat16);  view_280 = None
        view_281: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_340, [25088, 512]);  convert_element_type_340 = None
        permute_106: "bf16[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_339, [1, 0]);  convert_element_type_339 = None
        addmm_40: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_338, view_281, permute_106);  convert_element_type_338 = None
        view_282: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [512, 49, 1536]);  addmm_40 = None
        view_283: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_282, [512, 49, 3, 16, -1]);  view_282 = None
        permute_107: "bf16[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_283, [2, 0, 3, 1, 4]);  view_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_10 = torch.ops.aten.unbind.int(permute_107);  permute_107 = None
        getitem_78: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_10[0]
        getitem_79: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_10[1]
        getitem_80: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_10[2];  unbind_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_106: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_78, 0.1767766952966369);  getitem_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_108: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_79, [0, 1, 3, 2]);  getitem_79 = None
        expand_40: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_106, [512, 16, 49, 32]);  mul_106 = None
        clone_113: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_284: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_113, [8192, 49, 32]);  clone_113 = None
        expand_41: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_108, [512, 16, 32, 49]);  permute_108 = None
        clone_114: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_285: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_114, [8192, 32, 49]);  clone_114 = None
        constant_pad_nd_default_50: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_284, [0, 0, 0, 7, 0, 0])
        constant_pad_nd_default_51: "bf16[8192, 32, 56][1792, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_285, [0, 7, 0, 0, 0, 0])
        bmm_default_25: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_50, constant_pad_nd_default_51);  constant_pad_nd_default_50 = constant_pad_nd_default_51 = None
        slice_tensor_36: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_25, 1, 0, -7)
        slice_tensor_37: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_36, 2, 0, -7);  slice_tensor_36 = None
        view_286: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_37, [512, 16, 49, 49]);  slice_tensor_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_287: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_162, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_30: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_161, [view_287]);  view_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_288: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_30, [49, 49, -1]);  index_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_109: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_288, [2, 0, 1]);  view_288 = None
        clone_115: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_109, memory_format = torch.contiguous_format);  permute_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_20: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_115, 0);  clone_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_113: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_286, unsqueeze_20);  view_286 = unsqueeze_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_10: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_113, [-1], True)
        sub_34: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_113, amax_10);  add_113 = None
        exp_10: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_11: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_28: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        convert_element_type_346: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_28, torch.bfloat16);  div_28 = None
        expand_42: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_346, [512, 16, 49, 49]);  convert_element_type_346 = None
        view_289: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_42, [8192, 49, 49]);  expand_42 = None
        expand_43: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_80, [512, 16, 49, 32]);  getitem_80 = None
        clone_117: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_290: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_117, [8192, 49, 32]);  clone_117 = None
        constant_pad_nd_default_48: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_289, [0, 7, 0, 7, 0, 0])
        constant_pad_nd_default_49: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_290, [0, 0, 0, 7, 0, 0])
        bmm_default_24: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_48, constant_pad_nd_default_49);  constant_pad_nd_default_48 = constant_pad_nd_default_49 = None
        slice_tensor_35: "bf16[8192, 49, 32][1792, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_24, 1, 0, -7);  bmm_default_24 = None
        view_291: "bf16[512, 16, 49, 32][28672, 1792, 32, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_35, [512, 16, 49, 32]);  slice_tensor_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_110: "bf16[512, 49, 16, 32][28672, 32, 1792, 1]cuda:0" = torch.ops.aten.permute.default(view_291, [0, 2, 1, 3]);  view_291 = None
        clone_118: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None
        view_292: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_118, [512, 49, 512]);  clone_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        convert_element_type_349: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_164, torch.bfloat16);  primals_164 = None
        convert_element_type_350: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_163, torch.bfloat16);  primals_163 = None
        view_293: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_292, [25088, 512]);  view_292 = None
        permute_111: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_350, [1, 0]);  convert_element_type_350 = None
        addmm_41: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_349, view_293, permute_111);  convert_element_type_349 = None
        view_294: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_41, [512, 49, 512]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_295: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_294, [-1, 7, 7, 512]);  view_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_296: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_295, [-1, 2, 2, 7, 7, 512]);  view_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_112: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_296, [0, 1, 3, 2, 4, 5]);  view_296 = None
        clone_120: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
        view_297: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_120, [-1, 14, 14, 512]);  clone_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_18: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 18)
        inductor_random_default_27: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_18, 'rand');  inductor_lookup_seed_default_18 = None
        lt_18: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_27, 0.9565217345952988);  inductor_random_default_27 = None
        convert_element_type_354: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_18, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_29: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_354, 0.9565217345952988);  convert_element_type_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_107: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_297, div_29);  view_297 = div_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_114: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_277, mul_107);  mul_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_298: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_114, [128, -1, 512]);  add_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        convert_element_type_355: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_298, torch.float32)
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_355, [2], correction = 0, keepdim = True)
        getitem_81: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_24[0]
        getitem_82: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None
        add_115: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_81, 1e-05);  getitem_81 = None
        rsqrt_24: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
        sub_35: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_355, getitem_82);  convert_element_type_355 = None
        mul_108: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_24);  sub_35 = None
        mul_109: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_108, primals_165);  mul_108 = None
        add_116: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_109, primals_166);  mul_109 = primals_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_356: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_168, torch.bfloat16);  primals_168 = None
        convert_element_type_357: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_167, torch.bfloat16);  primals_167 = None
        convert_element_type_358: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_116, torch.bfloat16);  add_116 = None
        view_299: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_358, [25088, 512]);  convert_element_type_358 = None
        permute_113: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_357, [1, 0]);  convert_element_type_357 = None
        addmm_42: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_356, view_299, permute_113);  convert_element_type_356 = None
        view_300: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_362: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_300, torch.float32);  view_300 = None
        mul_110: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_362, 0.5)
        mul_111: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_362, 0.7071067811865476);  convert_element_type_362 = None
        erf_10: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_111);  mul_111 = None
        add_117: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_112: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_110, add_117);  mul_110 = add_117 = None
        convert_element_type_363: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_112, torch.bfloat16);  mul_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_364: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_170, torch.bfloat16);  primals_170 = None
        convert_element_type_365: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_169, torch.bfloat16);  primals_169 = None
        view_301: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_363, [25088, 2048]);  convert_element_type_363 = None
        permute_114: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_365, [1, 0]);  convert_element_type_365 = None
        addmm_43: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_364, view_301, permute_114);  convert_element_type_364 = None
        view_302: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_43, [128, 196, 512]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_19: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 19)
        inductor_random_default_26: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_19, 'rand');  inductor_lookup_seed_default_19 = None
        lt_19: "b8[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_26, 0.9565217345952988);  inductor_random_default_26 = None
        convert_element_type_369: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_19, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_30: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_369, 0.9565217345952988);  convert_element_type_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_113: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_302, div_30);  view_302 = div_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_118: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_298, mul_113);  mul_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_303: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_118, [128, 14, 14, 512]);  add_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        convert_element_type_370: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_303, torch.float32)
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_370, [3], correction = 0, keepdim = True)
        getitem_83: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_25[0]
        getitem_84: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_25[1];  var_mean_25 = None
        add_119: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_83, 1e-05);  getitem_83 = None
        rsqrt_25: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
        sub_36: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_370, getitem_84);  convert_element_type_370 = None
        mul_114: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_25);  sub_36 = None
        mul_115: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_114, primals_171);  mul_114 = None
        add_120: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_115, primals_172);  mul_115 = primals_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_31: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(add_120, [None, fmod_8]);  add_120 = None
        index_32: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_31, [None, None, fmod_8]);  index_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_304: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(index_32, [128, 2, 7, 2, 7, 512]);  index_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_115: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_304, [0, 1, 3, 2, 4, 5]);  view_304 = None
        clone_123: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
        view_305: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_123, [-1, 7, 7, 512]);  clone_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_306: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_305, [-1, 49, 512]);  view_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        convert_element_type_371: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_175, torch.bfloat16);  primals_175 = None
        convert_element_type_372: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_174, torch.bfloat16);  primals_174 = None
        convert_element_type_373: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_306, torch.bfloat16);  view_306 = None
        view_307: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_373, [25088, 512]);  convert_element_type_373 = None
        permute_116: "bf16[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_372, [1, 0]);  convert_element_type_372 = None
        addmm_44: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_371, view_307, permute_116);  convert_element_type_371 = None
        view_308: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_44, [512, 49, 1536]);  addmm_44 = None
        view_309: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_308, [512, 49, 3, 16, -1]);  view_308 = None
        permute_117: "bf16[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_309, [2, 0, 3, 1, 4]);  view_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_11 = torch.ops.aten.unbind.int(permute_117);  permute_117 = None
        getitem_85: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_11[0]
        getitem_86: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_11[1]
        getitem_87: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_11[2];  unbind_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_116: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_85, 0.1767766952966369);  getitem_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_118: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_86, [0, 1, 3, 2]);  getitem_86 = None
        expand_44: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_116, [512, 16, 49, 32]);  mul_116 = None
        clone_124: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_310: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_124, [8192, 49, 32]);  clone_124 = None
        expand_45: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_118, [512, 16, 32, 49]);  permute_118 = None
        clone_125: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_311: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_125, [8192, 32, 49]);  clone_125 = None
        constant_pad_nd_default_46: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_310, [0, 0, 0, 7, 0, 0])
        constant_pad_nd_default_47: "bf16[8192, 32, 56][1792, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_311, [0, 7, 0, 0, 0, 0])
        bmm_default_23: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_46, constant_pad_nd_default_47);  constant_pad_nd_default_46 = constant_pad_nd_default_47 = None
        slice_tensor_33: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_23, 1, 0, -7)
        slice_tensor_34: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_33, 2, 0, -7);  slice_tensor_33 = None
        view_312: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_34, [512, 16, 49, 49]);  slice_tensor_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_313: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_177, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_33: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_176, [view_313]);  view_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_314: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_33, [49, 49, -1]);  index_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_119: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_314, [2, 0, 1]);  view_314 = None
        clone_126: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_119, memory_format = torch.contiguous_format);  permute_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_21: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_126, 0);  clone_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_123: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_312, unsqueeze_21);  view_312 = unsqueeze_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_315: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_123, [-1, 4, 16, 49, 49]);  add_123 = None
        unsqueeze_22: "f32[4, 1, 49, 49][2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_173, 1)
        unsqueeze_23: "f32[1, 4, 1, 49, 49][9604, 2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_22, 0);  unsqueeze_22 = None
        add_124: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_315, unsqueeze_23);  view_315 = unsqueeze_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_316: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_124, [-1, 16, 49, 49]);  add_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_11: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(view_316, [-1], True)
        sub_37: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_316, amax_11);  view_316 = None
        exp_11: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
        sum_12: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_31: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        convert_element_type_379: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_31, torch.bfloat16);  div_31 = None
        expand_46: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_379, [512, 16, 49, 49]);  convert_element_type_379 = None
        view_317: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_46, [8192, 49, 49]);  expand_46 = None
        expand_47: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_87, [512, 16, 49, 32]);  getitem_87 = None
        clone_128: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_318: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_128, [8192, 49, 32]);  clone_128 = None
        constant_pad_nd_default_44: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_317, [0, 7, 0, 7, 0, 0])
        constant_pad_nd_default_45: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_318, [0, 0, 0, 7, 0, 0])
        bmm_default_22: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_44, constant_pad_nd_default_45);  constant_pad_nd_default_44 = constant_pad_nd_default_45 = None
        slice_tensor_32: "bf16[8192, 49, 32][1792, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_22, 1, 0, -7);  bmm_default_22 = None
        view_319: "bf16[512, 16, 49, 32][28672, 1792, 32, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_32, [512, 16, 49, 32]);  slice_tensor_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_120: "bf16[512, 49, 16, 32][28672, 32, 1792, 1]cuda:0" = torch.ops.aten.permute.default(view_319, [0, 2, 1, 3]);  view_319 = None
        clone_129: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_120, memory_format = torch.contiguous_format);  permute_120 = None
        view_320: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_129, [512, 49, 512]);  clone_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        convert_element_type_382: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_179, torch.bfloat16);  primals_179 = None
        convert_element_type_383: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_178, torch.bfloat16);  primals_178 = None
        view_321: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_320, [25088, 512]);  view_320 = None
        permute_121: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_383, [1, 0]);  convert_element_type_383 = None
        addmm_45: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_382, view_321, permute_121);  convert_element_type_382 = None
        view_322: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_45, [512, 49, 512]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_323: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_322, [-1, 7, 7, 512]);  view_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_324: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_323, [-1, 2, 2, 7, 7, 512]);  view_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_122: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_324, [0, 1, 3, 2, 4, 5]);  view_324 = None
        clone_131: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
        view_325: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_131, [-1, 14, 14, 512]);  clone_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_34: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_325, [None, fmod_10]);  view_325 = None
        index_35: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_34, [None, None, fmod_10]);  index_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_20: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 20)
        inductor_random_default_25: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_20, 'rand');  inductor_lookup_seed_default_20 = None
        lt_20: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_25, 0.9521739110350609);  inductor_random_default_25 = None
        convert_element_type_387: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_20, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_32: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_387, 0.9521739110350609);  convert_element_type_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_117: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(index_35, div_32);  index_35 = div_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_127: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_303, mul_117);  mul_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_326: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_127, [128, -1, 512]);  add_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        convert_element_type_388: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_326, torch.float32)
        var_mean_26 = torch.ops.aten.var_mean.correction(convert_element_type_388, [2], correction = 0, keepdim = True)
        getitem_88: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_26[0]
        getitem_89: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_26[1];  var_mean_26 = None
        add_128: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_26: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
        sub_38: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_388, getitem_89);  convert_element_type_388 = None
        mul_118: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_26);  sub_38 = None
        mul_119: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_118, primals_180);  mul_118 = None
        add_129: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_119, primals_181);  mul_119 = primals_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_389: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_183, torch.bfloat16);  primals_183 = None
        convert_element_type_390: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_182, torch.bfloat16);  primals_182 = None
        convert_element_type_391: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_129, torch.bfloat16);  add_129 = None
        view_327: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_391, [25088, 512]);  convert_element_type_391 = None
        permute_123: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_390, [1, 0]);  convert_element_type_390 = None
        addmm_46: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_389, view_327, permute_123);  convert_element_type_389 = None
        view_328: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_395: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_328, torch.float32);  view_328 = None
        mul_120: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_395, 0.5)
        mul_121: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_395, 0.7071067811865476);  convert_element_type_395 = None
        erf_11: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_121);  mul_121 = None
        add_130: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_122: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_120, add_130);  mul_120 = add_130 = None
        convert_element_type_396: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_122, torch.bfloat16);  mul_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_397: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_185, torch.bfloat16);  primals_185 = None
        convert_element_type_398: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_184, torch.bfloat16);  primals_184 = None
        view_329: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_396, [25088, 2048]);  convert_element_type_396 = None
        permute_124: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_398, [1, 0]);  convert_element_type_398 = None
        addmm_47: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_397, view_329, permute_124);  convert_element_type_397 = None
        view_330: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_47, [128, 196, 512]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_21: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 21)
        inductor_random_default_24: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_21, 'rand');  inductor_lookup_seed_default_21 = None
        lt_21: "b8[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_24, 0.9521739110350609);  inductor_random_default_24 = None
        convert_element_type_402: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_21, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_33: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_402, 0.9521739110350609);  convert_element_type_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_123: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_330, div_33);  view_330 = div_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_131: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_326, mul_123);  mul_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_331: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_131, [128, 14, 14, 512]);  add_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        convert_element_type_403: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_331, torch.float32)
        var_mean_27 = torch.ops.aten.var_mean.correction(convert_element_type_403, [3], correction = 0, keepdim = True)
        getitem_90: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_27[0]
        getitem_91: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_27[1];  var_mean_27 = None
        add_132: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
        rsqrt_27: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
        sub_39: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_403, getitem_91);  convert_element_type_403 = None
        mul_124: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_27);  sub_39 = None
        mul_125: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_124, primals_186);  mul_124 = None
        add_133: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_125, primals_187);  mul_125 = primals_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_332: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_133, [128, 2, 7, 2, 7, 512]);  add_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_125: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_332, [0, 1, 3, 2, 4, 5]);  view_332 = None
        clone_134: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
        view_333: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_134, [-1, 7, 7, 512]);  clone_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_334: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_333, [-1, 49, 512]);  view_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        convert_element_type_404: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_189, torch.bfloat16);  primals_189 = None
        convert_element_type_405: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_188, torch.bfloat16);  primals_188 = None
        convert_element_type_406: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_334, torch.bfloat16);  view_334 = None
        view_335: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_406, [25088, 512]);  convert_element_type_406 = None
        permute_126: "bf16[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_405, [1, 0]);  convert_element_type_405 = None
        addmm_48: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_404, view_335, permute_126);  convert_element_type_404 = None
        view_336: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_48, [512, 49, 1536]);  addmm_48 = None
        view_337: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_336, [512, 49, 3, 16, -1]);  view_336 = None
        permute_127: "bf16[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_337, [2, 0, 3, 1, 4]);  view_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_12 = torch.ops.aten.unbind.int(permute_127);  permute_127 = None
        getitem_92: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_12[0]
        getitem_93: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_12[1]
        getitem_94: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_12[2];  unbind_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_126: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_92, 0.1767766952966369);  getitem_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_128: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_93, [0, 1, 3, 2]);  getitem_93 = None
        expand_48: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_126, [512, 16, 49, 32]);  mul_126 = None
        clone_135: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_338: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_135, [8192, 49, 32]);  clone_135 = None
        expand_49: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_128, [512, 16, 32, 49]);  permute_128 = None
        clone_136: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_339: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_136, [8192, 32, 49]);  clone_136 = None
        constant_pad_nd_default_42: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_338, [0, 0, 0, 7, 0, 0])
        constant_pad_nd_default_43: "bf16[8192, 32, 56][1792, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_339, [0, 7, 0, 0, 0, 0])
        bmm_default_21: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_42, constant_pad_nd_default_43);  constant_pad_nd_default_42 = constant_pad_nd_default_43 = None
        slice_tensor_30: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_21, 1, 0, -7)
        slice_tensor_31: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_30, 2, 0, -7);  slice_tensor_30 = None
        view_340: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_31, [512, 16, 49, 49]);  slice_tensor_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_341: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_191, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_36: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_190, [view_341]);  view_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_342: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_36, [49, 49, -1]);  index_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_129: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_342, [2, 0, 1]);  view_342 = None
        clone_137: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_24: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_137, 0);  clone_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_134: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_340, unsqueeze_24);  view_340 = unsqueeze_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_12: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_134, [-1], True)
        sub_40: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_134, amax_12);  add_134 = None
        exp_12: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_40);  sub_40 = None
        sum_13: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_34: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        convert_element_type_412: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_34, torch.bfloat16);  div_34 = None
        expand_50: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_412, [512, 16, 49, 49]);  convert_element_type_412 = None
        view_343: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_50, [8192, 49, 49]);  expand_50 = None
        expand_51: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_94, [512, 16, 49, 32]);  getitem_94 = None
        clone_139: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        view_344: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_139, [8192, 49, 32]);  clone_139 = None
        constant_pad_nd_default_40: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_343, [0, 7, 0, 7, 0, 0])
        constant_pad_nd_default_41: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_344, [0, 0, 0, 7, 0, 0])
        bmm_default_20: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_40, constant_pad_nd_default_41);  constant_pad_nd_default_40 = constant_pad_nd_default_41 = None
        slice_tensor_29: "bf16[8192, 49, 32][1792, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_20, 1, 0, -7);  bmm_default_20 = None
        view_345: "bf16[512, 16, 49, 32][28672, 1792, 32, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_29, [512, 16, 49, 32]);  slice_tensor_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_130: "bf16[512, 49, 16, 32][28672, 32, 1792, 1]cuda:0" = torch.ops.aten.permute.default(view_345, [0, 2, 1, 3]);  view_345 = None
        clone_140: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_130, memory_format = torch.contiguous_format);  permute_130 = None
        view_346: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_140, [512, 49, 512]);  clone_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        convert_element_type_415: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_193, torch.bfloat16);  primals_193 = None
        convert_element_type_416: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_192, torch.bfloat16);  primals_192 = None
        view_347: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_346, [25088, 512]);  view_346 = None
        permute_131: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_416, [1, 0]);  convert_element_type_416 = None
        addmm_49: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_415, view_347, permute_131);  convert_element_type_415 = None
        view_348: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_49, [512, 49, 512]);  addmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_349: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_348, [-1, 7, 7, 512]);  view_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_350: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_349, [-1, 2, 2, 7, 7, 512]);  view_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_132: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_350, [0, 1, 3, 2, 4, 5]);  view_350 = None
        clone_142: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_132, memory_format = torch.contiguous_format);  permute_132 = None
        view_351: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_142, [-1, 14, 14, 512]);  clone_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_22: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 22)
        inductor_random_default_23: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_22, 'rand');  inductor_lookup_seed_default_22 = None
        lt_22: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_23, 0.947826087474823);  inductor_random_default_23 = None
        convert_element_type_420: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_22, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_35: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_420, 0.947826087474823);  convert_element_type_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_127: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_351, div_35);  view_351 = div_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_135: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_331, mul_127);  mul_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_352: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_135, [128, -1, 512]);  add_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        convert_element_type_421: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_352, torch.float32)
        var_mean_28 = torch.ops.aten.var_mean.correction(convert_element_type_421, [2], correction = 0, keepdim = True)
        getitem_95: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_28[0]
        getitem_96: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_28[1];  var_mean_28 = None
        add_136: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_95, 1e-05);  getitem_95 = None
        rsqrt_28: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
        sub_41: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_421, getitem_96);  convert_element_type_421 = None
        mul_128: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_28);  sub_41 = None
        mul_129: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_128, primals_194);  mul_128 = None
        add_137: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_129, primals_195);  mul_129 = primals_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_422: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_197, torch.bfloat16);  primals_197 = None
        convert_element_type_423: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_196, torch.bfloat16);  primals_196 = None
        convert_element_type_424: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_137, torch.bfloat16);  add_137 = None
        view_353: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_424, [25088, 512]);  convert_element_type_424 = None
        permute_133: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_423, [1, 0]);  convert_element_type_423 = None
        addmm_50: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_422, view_353, permute_133);  convert_element_type_422 = None
        view_354: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_50, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_428: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_354, torch.float32);  view_354 = None
        mul_130: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_428, 0.5)
        mul_131: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_428, 0.7071067811865476);  convert_element_type_428 = None
        erf_12: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_131);  mul_131 = None
        add_138: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_132: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_130, add_138);  mul_130 = add_138 = None
        convert_element_type_429: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_132, torch.bfloat16);  mul_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_430: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_199, torch.bfloat16);  primals_199 = None
        convert_element_type_431: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_198, torch.bfloat16);  primals_198 = None
        view_355: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_429, [25088, 2048]);  convert_element_type_429 = None
        permute_134: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_431, [1, 0]);  convert_element_type_431 = None
        addmm_51: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_430, view_355, permute_134);  convert_element_type_430 = None
        view_356: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_51, [128, 196, 512]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_23: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 23)
        inductor_random_default_22: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_23, 'rand');  inductor_lookup_seed_default_23 = None
        lt_23: "b8[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_22, 0.947826087474823);  inductor_random_default_22 = None
        convert_element_type_435: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_23, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_36: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_435, 0.947826087474823);  convert_element_type_435 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_133: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_356, div_36);  view_356 = div_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_139: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_352, mul_133);  mul_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_357: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_139, [128, 14, 14, 512]);  add_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        convert_element_type_436: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_357, torch.float32)
        var_mean_29 = torch.ops.aten.var_mean.correction(convert_element_type_436, [3], correction = 0, keepdim = True)
        getitem_97: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_29[0]
        getitem_98: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_29[1];  var_mean_29 = None
        add_140: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_97, 1e-05);  getitem_97 = None
        rsqrt_29: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
        sub_42: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_436, getitem_98);  convert_element_type_436 = None
        mul_134: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_29);  sub_42 = None
        mul_135: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_134, primals_200);  mul_134 = None
        add_141: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_135, primals_201);  mul_135 = primals_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_37: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(add_141, [None, fmod_8]);  add_141 = None
        index_38: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_37, [None, None, fmod_8]);  index_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_358: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(index_38, [128, 2, 7, 2, 7, 512]);  index_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_135: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_358, [0, 1, 3, 2, 4, 5]);  view_358 = None
        clone_145: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None
        view_359: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_145, [-1, 7, 7, 512]);  clone_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_360: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_359, [-1, 49, 512]);  view_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        convert_element_type_437: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_204, torch.bfloat16);  primals_204 = None
        convert_element_type_438: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_203, torch.bfloat16);  primals_203 = None
        convert_element_type_439: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_360, torch.bfloat16);  view_360 = None
        view_361: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_439, [25088, 512]);  convert_element_type_439 = None
        permute_136: "bf16[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_438, [1, 0]);  convert_element_type_438 = None
        addmm_52: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_437, view_361, permute_136);  convert_element_type_437 = None
        view_362: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_52, [512, 49, 1536]);  addmm_52 = None
        view_363: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_362, [512, 49, 3, 16, -1]);  view_362 = None
        permute_137: "bf16[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_363, [2, 0, 3, 1, 4]);  view_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_13 = torch.ops.aten.unbind.int(permute_137);  permute_137 = None
        getitem_99: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_13[0]
        getitem_100: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_13[1]
        getitem_101: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_13[2];  unbind_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_136: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_99, 0.1767766952966369);  getitem_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_138: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_100, [0, 1, 3, 2]);  getitem_100 = None
        expand_52: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_136, [512, 16, 49, 32]);  mul_136 = None
        clone_146: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_364: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_146, [8192, 49, 32]);  clone_146 = None
        expand_53: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_138, [512, 16, 32, 49]);  permute_138 = None
        clone_147: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
        view_365: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_147, [8192, 32, 49]);  clone_147 = None
        constant_pad_nd_default_38: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_364, [0, 0, 0, 7, 0, 0])
        constant_pad_nd_default_39: "bf16[8192, 32, 56][1792, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_365, [0, 7, 0, 0, 0, 0])
        bmm_default_19: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_38, constant_pad_nd_default_39);  constant_pad_nd_default_38 = constant_pad_nd_default_39 = None
        slice_tensor_27: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_19, 1, 0, -7)
        slice_tensor_28: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_27, 2, 0, -7);  slice_tensor_27 = None
        view_366: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_28, [512, 16, 49, 49]);  slice_tensor_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_367: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_206, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_39: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_205, [view_367]);  view_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_368: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_39, [49, 49, -1]);  index_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_139: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_368, [2, 0, 1]);  view_368 = None
        clone_148: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_25: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_148, 0);  clone_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_144: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_366, unsqueeze_25);  view_366 = unsqueeze_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_369: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_144, [-1, 4, 16, 49, 49]);  add_144 = None
        unsqueeze_26: "f32[4, 1, 49, 49][2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_202, 1)
        unsqueeze_27: "f32[1, 4, 1, 49, 49][9604, 2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_26, 0);  unsqueeze_26 = None
        add_145: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_369, unsqueeze_27);  view_369 = unsqueeze_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_370: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_145, [-1, 16, 49, 49]);  add_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_13: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(view_370, [-1], True)
        sub_43: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_370, amax_13);  view_370 = None
        exp_13: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
        sum_14: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_37: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        convert_element_type_445: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_37, torch.bfloat16);  div_37 = None
        expand_54: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_445, [512, 16, 49, 49]);  convert_element_type_445 = None
        view_371: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_54, [8192, 49, 49]);  expand_54 = None
        expand_55: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_101, [512, 16, 49, 32]);  getitem_101 = None
        clone_150: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
        view_372: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_150, [8192, 49, 32]);  clone_150 = None
        constant_pad_nd_default_36: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_371, [0, 7, 0, 7, 0, 0])
        constant_pad_nd_default_37: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_372, [0, 0, 0, 7, 0, 0])
        bmm_default_18: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_36, constant_pad_nd_default_37);  constant_pad_nd_default_36 = constant_pad_nd_default_37 = None
        slice_tensor_26: "bf16[8192, 49, 32][1792, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_18, 1, 0, -7);  bmm_default_18 = None
        view_373: "bf16[512, 16, 49, 32][28672, 1792, 32, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_26, [512, 16, 49, 32]);  slice_tensor_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_140: "bf16[512, 49, 16, 32][28672, 32, 1792, 1]cuda:0" = torch.ops.aten.permute.default(view_373, [0, 2, 1, 3]);  view_373 = None
        clone_151: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format);  permute_140 = None
        view_374: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_151, [512, 49, 512]);  clone_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        convert_element_type_448: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_208, torch.bfloat16);  primals_208 = None
        convert_element_type_449: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_207, torch.bfloat16);  primals_207 = None
        view_375: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_374, [25088, 512]);  view_374 = None
        permute_141: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_449, [1, 0]);  convert_element_type_449 = None
        addmm_53: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_448, view_375, permute_141);  convert_element_type_448 = None
        view_376: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_53, [512, 49, 512]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_377: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_376, [-1, 7, 7, 512]);  view_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_378: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_377, [-1, 2, 2, 7, 7, 512]);  view_377 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_142: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_378, [0, 1, 3, 2, 4, 5]);  view_378 = None
        clone_153: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_142, memory_format = torch.contiguous_format);  permute_142 = None
        view_379: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_153, [-1, 14, 14, 512]);  clone_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_40: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_379, [None, fmod_10]);  view_379 = None
        index_41: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_40, [None, None, fmod_10]);  index_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_24: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 24)
        inductor_random_default_21: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_24, 'rand');  inductor_lookup_seed_default_24 = None
        lt_24: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_21, 0.9434782639145851);  inductor_random_default_21 = None
        convert_element_type_453: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_24, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_38: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_453, 0.9434782639145851);  convert_element_type_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_137: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(index_41, div_38);  index_41 = div_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_148: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_357, mul_137);  mul_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_380: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_148, [128, -1, 512]);  add_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        convert_element_type_454: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_380, torch.float32)
        var_mean_30 = torch.ops.aten.var_mean.correction(convert_element_type_454, [2], correction = 0, keepdim = True)
        getitem_102: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_30[0]
        getitem_103: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_30[1];  var_mean_30 = None
        add_149: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_102, 1e-05);  getitem_102 = None
        rsqrt_30: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
        sub_44: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_454, getitem_103);  convert_element_type_454 = None
        mul_138: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_30);  sub_44 = None
        mul_139: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_138, primals_209);  mul_138 = None
        add_150: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_139, primals_210);  mul_139 = primals_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_455: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_212, torch.bfloat16);  primals_212 = None
        convert_element_type_456: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_211, torch.bfloat16);  primals_211 = None
        convert_element_type_457: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_150, torch.bfloat16);  add_150 = None
        view_381: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_457, [25088, 512]);  convert_element_type_457 = None
        permute_143: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_456, [1, 0]);  convert_element_type_456 = None
        addmm_54: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_455, view_381, permute_143);  convert_element_type_455 = None
        view_382: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_54, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_461: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_382, torch.float32);  view_382 = None
        mul_140: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_461, 0.5)
        mul_141: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_461, 0.7071067811865476);  convert_element_type_461 = None
        erf_13: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_141);  mul_141 = None
        add_151: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_142: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_140, add_151);  mul_140 = add_151 = None
        convert_element_type_462: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_142, torch.bfloat16);  mul_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_463: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_214, torch.bfloat16);  primals_214 = None
        convert_element_type_464: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_213, torch.bfloat16);  primals_213 = None
        view_383: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_462, [25088, 2048]);  convert_element_type_462 = None
        permute_144: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_464, [1, 0]);  convert_element_type_464 = None
        addmm_55: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_463, view_383, permute_144);  convert_element_type_463 = None
        view_384: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_55, [128, 196, 512]);  addmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_25: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 25)
        inductor_random_default_20: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_25, 'rand');  inductor_lookup_seed_default_25 = None
        lt_25: "b8[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_20, 0.9434782639145851);  inductor_random_default_20 = None
        convert_element_type_468: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_25, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_39: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_468, 0.9434782639145851);  convert_element_type_468 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_143: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_384, div_39);  view_384 = div_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_152: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_380, mul_143);  mul_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_385: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_152, [128, 14, 14, 512]);  add_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        convert_element_type_469: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_385, torch.float32)
        var_mean_31 = torch.ops.aten.var_mean.correction(convert_element_type_469, [3], correction = 0, keepdim = True)
        getitem_104: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_31[0]
        getitem_105: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_31[1];  var_mean_31 = None
        add_153: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
        rsqrt_31: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
        sub_45: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_469, getitem_105);  convert_element_type_469 = None
        mul_144: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_31);  sub_45 = None
        mul_145: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_144, primals_215);  mul_144 = None
        add_154: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_145, primals_216);  mul_145 = primals_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_386: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_154, [128, 2, 7, 2, 7, 512]);  add_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_145: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_386, [0, 1, 3, 2, 4, 5]);  view_386 = None
        clone_156: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_145, memory_format = torch.contiguous_format);  permute_145 = None
        view_387: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_156, [-1, 7, 7, 512]);  clone_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_388: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_387, [-1, 49, 512]);  view_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        convert_element_type_470: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_218, torch.bfloat16);  primals_218 = None
        convert_element_type_471: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_217, torch.bfloat16);  primals_217 = None
        convert_element_type_472: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_388, torch.bfloat16);  view_388 = None
        view_389: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_472, [25088, 512]);  convert_element_type_472 = None
        permute_146: "bf16[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_471, [1, 0]);  convert_element_type_471 = None
        addmm_56: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_470, view_389, permute_146);  convert_element_type_470 = None
        view_390: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_56, [512, 49, 1536]);  addmm_56 = None
        view_391: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_390, [512, 49, 3, 16, -1]);  view_390 = None
        permute_147: "bf16[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_391, [2, 0, 3, 1, 4]);  view_391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_14 = torch.ops.aten.unbind.int(permute_147);  permute_147 = None
        getitem_106: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_14[0]
        getitem_107: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_14[1]
        getitem_108: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_14[2];  unbind_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_146: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_106, 0.1767766952966369);  getitem_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_148: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_107, [0, 1, 3, 2]);  getitem_107 = None
        expand_56: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_146, [512, 16, 49, 32]);  mul_146 = None
        clone_157: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_392: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_157, [8192, 49, 32]);  clone_157 = None
        expand_57: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_148, [512, 16, 32, 49]);  permute_148 = None
        clone_158: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_393: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_158, [8192, 32, 49]);  clone_158 = None
        constant_pad_nd_default_34: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_392, [0, 0, 0, 7, 0, 0])
        constant_pad_nd_default_35: "bf16[8192, 32, 56][1792, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_393, [0, 7, 0, 0, 0, 0])
        bmm_default_17: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_34, constant_pad_nd_default_35);  constant_pad_nd_default_34 = constant_pad_nd_default_35 = None
        slice_tensor_24: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_17, 1, 0, -7)
        slice_tensor_25: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_24, 2, 0, -7);  slice_tensor_24 = None
        view_394: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_25, [512, 16, 49, 49]);  slice_tensor_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_395: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_220, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_42: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_219, [view_395]);  view_395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_396: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_42, [49, 49, -1]);  index_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_149: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_396, [2, 0, 1]);  view_396 = None
        clone_159: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_149, memory_format = torch.contiguous_format);  permute_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_28: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_159, 0);  clone_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_155: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_394, unsqueeze_28);  view_394 = unsqueeze_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_14: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_155, [-1], True)
        sub_46: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_155, amax_14);  add_155 = None
        exp_14: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
        sum_15: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_40: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        convert_element_type_478: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_40, torch.bfloat16);  div_40 = None
        expand_58: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_478, [512, 16, 49, 49]);  convert_element_type_478 = None
        view_397: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_58, [8192, 49, 49]);  expand_58 = None
        expand_59: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_108, [512, 16, 49, 32]);  getitem_108 = None
        clone_161: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
        view_398: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_161, [8192, 49, 32]);  clone_161 = None
        constant_pad_nd_default_32: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_397, [0, 7, 0, 7, 0, 0])
        constant_pad_nd_default_33: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_398, [0, 0, 0, 7, 0, 0])
        bmm_default_16: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_32, constant_pad_nd_default_33);  constant_pad_nd_default_32 = constant_pad_nd_default_33 = None
        slice_tensor_23: "bf16[8192, 49, 32][1792, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_16, 1, 0, -7);  bmm_default_16 = None
        view_399: "bf16[512, 16, 49, 32][28672, 1792, 32, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_23, [512, 16, 49, 32]);  slice_tensor_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_150: "bf16[512, 49, 16, 32][28672, 32, 1792, 1]cuda:0" = torch.ops.aten.permute.default(view_399, [0, 2, 1, 3]);  view_399 = None
        clone_162: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
        view_400: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_162, [512, 49, 512]);  clone_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        convert_element_type_481: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_222, torch.bfloat16);  primals_222 = None
        convert_element_type_482: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_221, torch.bfloat16);  primals_221 = None
        view_401: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_400, [25088, 512]);  view_400 = None
        permute_151: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_482, [1, 0]);  convert_element_type_482 = None
        addmm_57: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_481, view_401, permute_151);  convert_element_type_481 = None
        view_402: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_57, [512, 49, 512]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_403: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_402, [-1, 7, 7, 512]);  view_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_404: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_403, [-1, 2, 2, 7, 7, 512]);  view_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_152: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_404, [0, 1, 3, 2, 4, 5]);  view_404 = None
        clone_164: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_152, memory_format = torch.contiguous_format);  permute_152 = None
        view_405: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_164, [-1, 14, 14, 512]);  clone_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_26: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 26)
        inductor_random_default_19: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_26, 'rand');  inductor_lookup_seed_default_26 = None
        lt_26: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_19, 0.9391304366290569);  inductor_random_default_19 = None
        convert_element_type_486: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_26, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_41: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_486, 0.9391304366290569);  convert_element_type_486 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_147: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_405, div_41);  view_405 = div_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_156: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_385, mul_147);  mul_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_406: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_156, [128, -1, 512]);  add_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        convert_element_type_487: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_406, torch.float32)
        var_mean_32 = torch.ops.aten.var_mean.correction(convert_element_type_487, [2], correction = 0, keepdim = True)
        getitem_109: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_32[0]
        getitem_110: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_32[1];  var_mean_32 = None
        add_157: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_109, 1e-05);  getitem_109 = None
        rsqrt_32: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
        sub_47: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_487, getitem_110);  convert_element_type_487 = None
        mul_148: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_32);  sub_47 = None
        mul_149: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_148, primals_223);  mul_148 = None
        add_158: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_149, primals_224);  mul_149 = primals_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_488: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_226, torch.bfloat16);  primals_226 = None
        convert_element_type_489: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_225, torch.bfloat16);  primals_225 = None
        convert_element_type_490: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_158, torch.bfloat16);  add_158 = None
        view_407: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_490, [25088, 512]);  convert_element_type_490 = None
        permute_153: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_489, [1, 0]);  convert_element_type_489 = None
        addmm_58: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_488, view_407, permute_153);  convert_element_type_488 = None
        view_408: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_58, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_494: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_408, torch.float32);  view_408 = None
        mul_150: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_494, 0.5)
        mul_151: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_494, 0.7071067811865476);  convert_element_type_494 = None
        erf_14: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_151);  mul_151 = None
        add_159: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_152: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_150, add_159);  mul_150 = add_159 = None
        convert_element_type_495: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_152, torch.bfloat16);  mul_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_496: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_228, torch.bfloat16);  primals_228 = None
        convert_element_type_497: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_227, torch.bfloat16);  primals_227 = None
        view_409: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_495, [25088, 2048]);  convert_element_type_495 = None
        permute_154: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_497, [1, 0]);  convert_element_type_497 = None
        addmm_59: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_496, view_409, permute_154);  convert_element_type_496 = None
        view_410: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_59, [128, 196, 512]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_27: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 27)
        inductor_random_default_18: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_27, 'rand');  inductor_lookup_seed_default_27 = None
        lt_27: "b8[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_18, 0.9391304366290569);  inductor_random_default_18 = None
        convert_element_type_501: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_27, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_42: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_501, 0.9391304366290569);  convert_element_type_501 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_153: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_410, div_42);  view_410 = div_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_160: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_406, mul_153);  mul_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_411: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_160, [128, 14, 14, 512]);  add_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        convert_element_type_502: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_411, torch.float32)
        var_mean_33 = torch.ops.aten.var_mean.correction(convert_element_type_502, [3], correction = 0, keepdim = True)
        getitem_111: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_33[0]
        getitem_112: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_33[1];  var_mean_33 = None
        add_161: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_111, 1e-05);  getitem_111 = None
        rsqrt_33: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
        sub_48: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_502, getitem_112);  convert_element_type_502 = None
        mul_154: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_33);  sub_48 = None
        mul_155: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_154, primals_229);  mul_154 = None
        add_162: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_155, primals_230);  mul_155 = primals_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_43: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(add_162, [None, fmod_8]);  add_162 = None
        index_44: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_43, [None, None, fmod_8]);  index_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_412: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(index_44, [128, 2, 7, 2, 7, 512]);  index_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_155: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_412, [0, 1, 3, 2, 4, 5]);  view_412 = None
        clone_167: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
        view_413: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_167, [-1, 7, 7, 512]);  clone_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_414: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_413, [-1, 49, 512]);  view_413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        convert_element_type_503: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_233, torch.bfloat16);  primals_233 = None
        convert_element_type_504: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_232, torch.bfloat16);  primals_232 = None
        convert_element_type_505: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_414, torch.bfloat16);  view_414 = None
        view_415: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_505, [25088, 512]);  convert_element_type_505 = None
        permute_156: "bf16[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_504, [1, 0]);  convert_element_type_504 = None
        addmm_60: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_503, view_415, permute_156);  convert_element_type_503 = None
        view_416: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_60, [512, 49, 1536]);  addmm_60 = None
        view_417: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_416, [512, 49, 3, 16, -1]);  view_416 = None
        permute_157: "bf16[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_417, [2, 0, 3, 1, 4]);  view_417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_15 = torch.ops.aten.unbind.int(permute_157);  permute_157 = None
        getitem_113: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_15[0]
        getitem_114: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_15[1]
        getitem_115: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_15[2];  unbind_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_156: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_113, 0.1767766952966369);  getitem_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_158: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_114, [0, 1, 3, 2]);  getitem_114 = None
        expand_60: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_156, [512, 16, 49, 32]);  mul_156 = None
        clone_168: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_418: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_168, [8192, 49, 32]);  clone_168 = None
        expand_61: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_158, [512, 16, 32, 49]);  permute_158 = None
        clone_169: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
        view_419: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_169, [8192, 32, 49]);  clone_169 = None
        constant_pad_nd_default_30: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_418, [0, 0, 0, 7, 0, 0])
        constant_pad_nd_default_31: "bf16[8192, 32, 56][1792, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_419, [0, 7, 0, 0, 0, 0])
        bmm_default_15: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_30, constant_pad_nd_default_31);  constant_pad_nd_default_30 = constant_pad_nd_default_31 = None
        slice_tensor_21: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_15, 1, 0, -7)
        slice_tensor_22: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_21, 2, 0, -7);  slice_tensor_21 = None
        view_420: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_22, [512, 16, 49, 49]);  slice_tensor_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_421: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_235, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_45: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_234, [view_421]);  view_421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_422: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_45, [49, 49, -1]);  index_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_159: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_422, [2, 0, 1]);  view_422 = None
        clone_170: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_29: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_170, 0);  clone_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_165: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_420, unsqueeze_29);  view_420 = unsqueeze_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_423: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_165, [-1, 4, 16, 49, 49]);  add_165 = None
        unsqueeze_30: "f32[4, 1, 49, 49][2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_231, 1)
        unsqueeze_31: "f32[1, 4, 1, 49, 49][9604, 2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_30, 0);  unsqueeze_30 = None
        add_166: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_423, unsqueeze_31);  view_423 = unsqueeze_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_424: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_166, [-1, 16, 49, 49]);  add_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_15: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(view_424, [-1], True)
        sub_49: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_424, amax_15);  view_424 = None
        exp_15: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
        sum_16: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_43: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        convert_element_type_511: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_43, torch.bfloat16);  div_43 = None
        expand_62: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_511, [512, 16, 49, 49]);  convert_element_type_511 = None
        view_425: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_62, [8192, 49, 49]);  expand_62 = None
        expand_63: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_115, [512, 16, 49, 32]);  getitem_115 = None
        clone_172: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
        view_426: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_172, [8192, 49, 32]);  clone_172 = None
        constant_pad_nd_default_28: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_425, [0, 7, 0, 7, 0, 0])
        constant_pad_nd_default_29: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_426, [0, 0, 0, 7, 0, 0])
        bmm_default_14: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_28, constant_pad_nd_default_29);  constant_pad_nd_default_28 = constant_pad_nd_default_29 = None
        slice_tensor_20: "bf16[8192, 49, 32][1792, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_14, 1, 0, -7);  bmm_default_14 = None
        view_427: "bf16[512, 16, 49, 32][28672, 1792, 32, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_20, [512, 16, 49, 32]);  slice_tensor_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_160: "bf16[512, 49, 16, 32][28672, 32, 1792, 1]cuda:0" = torch.ops.aten.permute.default(view_427, [0, 2, 1, 3]);  view_427 = None
        clone_173: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
        view_428: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_173, [512, 49, 512]);  clone_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        convert_element_type_514: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_237, torch.bfloat16);  primals_237 = None
        convert_element_type_515: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_236, torch.bfloat16);  primals_236 = None
        view_429: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_428, [25088, 512]);  view_428 = None
        permute_161: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_515, [1, 0]);  convert_element_type_515 = None
        addmm_61: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_514, view_429, permute_161);  convert_element_type_514 = None
        view_430: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_61, [512, 49, 512]);  addmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_431: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_430, [-1, 7, 7, 512]);  view_430 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_432: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_431, [-1, 2, 2, 7, 7, 512]);  view_431 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_162: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_432, [0, 1, 3, 2, 4, 5]);  view_432 = None
        clone_175: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_162, memory_format = torch.contiguous_format);  permute_162 = None
        view_433: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_175, [-1, 14, 14, 512]);  clone_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_46: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_433, [None, fmod_10]);  view_433 = None
        index_47: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_46, [None, None, fmod_10]);  index_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_28: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 28)
        inductor_random_default_17: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_28, 'rand');  inductor_lookup_seed_default_28 = None
        lt_28: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_17, 0.9347826093435287);  inductor_random_default_17 = None
        convert_element_type_519: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_28, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_44: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_519, 0.9347826093435287);  convert_element_type_519 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_157: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(index_47, div_44);  index_47 = div_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_169: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_411, mul_157);  mul_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_434: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_169, [128, -1, 512]);  add_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        convert_element_type_520: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_434, torch.float32)
        var_mean_34 = torch.ops.aten.var_mean.correction(convert_element_type_520, [2], correction = 0, keepdim = True)
        getitem_116: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_34[0]
        getitem_117: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_34[1];  var_mean_34 = None
        add_170: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_116, 1e-05);  getitem_116 = None
        rsqrt_34: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        sub_50: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_520, getitem_117);  convert_element_type_520 = None
        mul_158: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_34);  sub_50 = None
        mul_159: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_158, primals_238);  mul_158 = None
        add_171: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_159, primals_239);  mul_159 = primals_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_521: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_241, torch.bfloat16);  primals_241 = None
        convert_element_type_522: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_240, torch.bfloat16);  primals_240 = None
        convert_element_type_523: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_171, torch.bfloat16);  add_171 = None
        view_435: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_523, [25088, 512]);  convert_element_type_523 = None
        permute_163: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_522, [1, 0]);  convert_element_type_522 = None
        addmm_62: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_521, view_435, permute_163);  convert_element_type_521 = None
        view_436: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_62, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_527: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_436, torch.float32);  view_436 = None
        mul_160: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_527, 0.5)
        mul_161: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_527, 0.7071067811865476);  convert_element_type_527 = None
        erf_15: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_161);  mul_161 = None
        add_172: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_162: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_160, add_172);  mul_160 = add_172 = None
        convert_element_type_528: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_162, torch.bfloat16);  mul_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_529: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_243, torch.bfloat16);  primals_243 = None
        convert_element_type_530: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_242, torch.bfloat16);  primals_242 = None
        view_437: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_528, [25088, 2048]);  convert_element_type_528 = None
        permute_164: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_530, [1, 0]);  convert_element_type_530 = None
        addmm_63: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_529, view_437, permute_164);  convert_element_type_529 = None
        view_438: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_63, [128, 196, 512]);  addmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_29: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 29)
        inductor_random_default_16: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_29, 'rand');  inductor_lookup_seed_default_29 = None
        lt_29: "b8[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_16, 0.9347826093435287);  inductor_random_default_16 = None
        convert_element_type_534: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_29, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_45: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_534, 0.9347826093435287);  convert_element_type_534 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_163: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_438, div_45);  view_438 = div_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_173: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_434, mul_163);  mul_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_439: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_173, [128, 14, 14, 512]);  add_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        convert_element_type_535: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_439, torch.float32)
        var_mean_35 = torch.ops.aten.var_mean.correction(convert_element_type_535, [3], correction = 0, keepdim = True)
        getitem_118: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_35[0]
        getitem_119: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_35[1];  var_mean_35 = None
        add_174: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_118, 1e-05);  getitem_118 = None
        rsqrt_35: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        sub_51: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_535, getitem_119);  convert_element_type_535 = None
        mul_164: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_35);  sub_51 = None
        mul_165: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_164, primals_244);  mul_164 = None
        add_175: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_165, primals_245);  mul_165 = primals_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_440: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_175, [128, 2, 7, 2, 7, 512]);  add_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_165: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_440, [0, 1, 3, 2, 4, 5]);  view_440 = None
        clone_178: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_165, memory_format = torch.contiguous_format);  permute_165 = None
        view_441: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_178, [-1, 7, 7, 512]);  clone_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_442: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_441, [-1, 49, 512]);  view_441 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        convert_element_type_536: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_247, torch.bfloat16);  primals_247 = None
        convert_element_type_537: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_246, torch.bfloat16);  primals_246 = None
        convert_element_type_538: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_442, torch.bfloat16);  view_442 = None
        view_443: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_538, [25088, 512]);  convert_element_type_538 = None
        permute_166: "bf16[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_537, [1, 0]);  convert_element_type_537 = None
        addmm_64: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_536, view_443, permute_166);  convert_element_type_536 = None
        view_444: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_64, [512, 49, 1536]);  addmm_64 = None
        view_445: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_444, [512, 49, 3, 16, -1]);  view_444 = None
        permute_167: "bf16[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_445, [2, 0, 3, 1, 4]);  view_445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_16 = torch.ops.aten.unbind.int(permute_167);  permute_167 = None
        getitem_120: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_16[0]
        getitem_121: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_16[1]
        getitem_122: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_16[2];  unbind_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_166: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_120, 0.1767766952966369);  getitem_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_168: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_121, [0, 1, 3, 2]);  getitem_121 = None
        expand_64: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_166, [512, 16, 49, 32]);  mul_166 = None
        clone_179: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
        view_446: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_179, [8192, 49, 32]);  clone_179 = None
        expand_65: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_168, [512, 16, 32, 49]);  permute_168 = None
        clone_180: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        view_447: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_180, [8192, 32, 49]);  clone_180 = None
        constant_pad_nd_default_26: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_446, [0, 0, 0, 7, 0, 0])
        constant_pad_nd_default_27: "bf16[8192, 32, 56][1792, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_447, [0, 7, 0, 0, 0, 0])
        bmm_default_13: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_26, constant_pad_nd_default_27);  constant_pad_nd_default_26 = constant_pad_nd_default_27 = None
        slice_tensor_18: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_13, 1, 0, -7)
        slice_tensor_19: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_18, 2, 0, -7);  slice_tensor_18 = None
        view_448: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_19, [512, 16, 49, 49]);  slice_tensor_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_449: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_249, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_48: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_248, [view_449]);  view_449 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_450: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_48, [49, 49, -1]);  index_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_169: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_450, [2, 0, 1]);  view_450 = None
        clone_181: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_169, memory_format = torch.contiguous_format);  permute_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_32: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_181, 0);  clone_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_176: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_448, unsqueeze_32);  view_448 = unsqueeze_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_16: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_176, [-1], True)
        sub_52: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_176, amax_16);  add_176 = None
        exp_16: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
        sum_17: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_46: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        convert_element_type_544: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_46, torch.bfloat16);  div_46 = None
        expand_66: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_544, [512, 16, 49, 49]);  convert_element_type_544 = None
        view_451: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_66, [8192, 49, 49]);  expand_66 = None
        expand_67: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_122, [512, 16, 49, 32]);  getitem_122 = None
        clone_183: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
        view_452: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_183, [8192, 49, 32]);  clone_183 = None
        constant_pad_nd_default_24: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_451, [0, 7, 0, 7, 0, 0])
        constant_pad_nd_default_25: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_452, [0, 0, 0, 7, 0, 0])
        bmm_default_12: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_24, constant_pad_nd_default_25);  constant_pad_nd_default_24 = constant_pad_nd_default_25 = None
        slice_tensor_17: "bf16[8192, 49, 32][1792, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_12, 1, 0, -7);  bmm_default_12 = None
        view_453: "bf16[512, 16, 49, 32][28672, 1792, 32, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_17, [512, 16, 49, 32]);  slice_tensor_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_170: "bf16[512, 49, 16, 32][28672, 32, 1792, 1]cuda:0" = torch.ops.aten.permute.default(view_453, [0, 2, 1, 3]);  view_453 = None
        clone_184: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
        view_454: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_184, [512, 49, 512]);  clone_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        convert_element_type_547: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_251, torch.bfloat16);  primals_251 = None
        convert_element_type_548: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_250, torch.bfloat16);  primals_250 = None
        view_455: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_454, [25088, 512]);  view_454 = None
        permute_171: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_548, [1, 0]);  convert_element_type_548 = None
        addmm_65: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_547, view_455, permute_171);  convert_element_type_547 = None
        view_456: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_65, [512, 49, 512]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_457: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_456, [-1, 7, 7, 512]);  view_456 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_458: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_457, [-1, 2, 2, 7, 7, 512]);  view_457 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_172: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_458, [0, 1, 3, 2, 4, 5]);  view_458 = None
        clone_186: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
        view_459: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_186, [-1, 14, 14, 512]);  clone_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_30: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 30)
        inductor_random_default_15: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_30, 'rand');  inductor_lookup_seed_default_30 = None
        lt_30: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_15, 0.9304347857832909);  inductor_random_default_15 = None
        convert_element_type_552: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_30, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_47: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_552, 0.9304347857832909);  convert_element_type_552 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_167: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_459, div_47);  view_459 = div_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_177: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_439, mul_167);  mul_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_460: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_177, [128, -1, 512]);  add_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        convert_element_type_553: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_460, torch.float32)
        var_mean_36 = torch.ops.aten.var_mean.correction(convert_element_type_553, [2], correction = 0, keepdim = True)
        getitem_123: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_36[0]
        getitem_124: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_36[1];  var_mean_36 = None
        add_178: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_123, 1e-05);  getitem_123 = None
        rsqrt_36: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
        sub_53: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_553, getitem_124);  convert_element_type_553 = None
        mul_168: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_36);  sub_53 = None
        mul_169: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_168, primals_252);  mul_168 = None
        add_179: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_169, primals_253);  mul_169 = primals_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_554: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_255, torch.bfloat16);  primals_255 = None
        convert_element_type_555: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_254, torch.bfloat16);  primals_254 = None
        convert_element_type_556: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_179, torch.bfloat16);  add_179 = None
        view_461: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_556, [25088, 512]);  convert_element_type_556 = None
        permute_173: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_555, [1, 0]);  convert_element_type_555 = None
        addmm_66: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_554, view_461, permute_173);  convert_element_type_554 = None
        view_462: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_66, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_560: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_462, torch.float32);  view_462 = None
        mul_170: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_560, 0.5)
        mul_171: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_560, 0.7071067811865476);  convert_element_type_560 = None
        erf_16: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_171);  mul_171 = None
        add_180: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_172: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_170, add_180);  mul_170 = add_180 = None
        convert_element_type_561: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_172, torch.bfloat16);  mul_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_562: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_257, torch.bfloat16);  primals_257 = None
        convert_element_type_563: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_256, torch.bfloat16);  primals_256 = None
        view_463: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_561, [25088, 2048]);  convert_element_type_561 = None
        permute_174: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_563, [1, 0]);  convert_element_type_563 = None
        addmm_67: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_562, view_463, permute_174);  convert_element_type_562 = None
        view_464: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_67, [128, 196, 512]);  addmm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_31: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 31)
        inductor_random_default_14: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_31, 'rand');  inductor_lookup_seed_default_31 = None
        lt_31: "b8[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_14, 0.9304347857832909);  inductor_random_default_14 = None
        convert_element_type_567: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_31, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_48: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_567, 0.9304347857832909);  convert_element_type_567 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_173: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_464, div_48);  view_464 = div_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_181: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_460, mul_173);  mul_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_465: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_181, [128, 14, 14, 512]);  add_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        convert_element_type_568: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_465, torch.float32)
        var_mean_37 = torch.ops.aten.var_mean.correction(convert_element_type_568, [3], correction = 0, keepdim = True)
        getitem_125: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_37[0]
        getitem_126: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_37[1];  var_mean_37 = None
        add_182: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_125, 1e-05);  getitem_125 = None
        rsqrt_37: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
        sub_54: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_568, getitem_126);  convert_element_type_568 = None
        mul_174: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_37);  sub_54 = None
        mul_175: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_174, primals_258);  mul_174 = None
        add_183: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_175, primals_259);  mul_175 = primals_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_49: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(add_183, [None, fmod_8]);  add_183 = None
        index_50: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_49, [None, None, fmod_8]);  index_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_466: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(index_50, [128, 2, 7, 2, 7, 512]);  index_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_175: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_466, [0, 1, 3, 2, 4, 5]);  view_466 = None
        clone_189: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_175, memory_format = torch.contiguous_format);  permute_175 = None
        view_467: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_189, [-1, 7, 7, 512]);  clone_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_468: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_467, [-1, 49, 512]);  view_467 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        convert_element_type_569: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_262, torch.bfloat16);  primals_262 = None
        convert_element_type_570: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_261, torch.bfloat16);  primals_261 = None
        convert_element_type_571: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_468, torch.bfloat16);  view_468 = None
        view_469: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_571, [25088, 512]);  convert_element_type_571 = None
        permute_176: "bf16[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_570, [1, 0]);  convert_element_type_570 = None
        addmm_68: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_569, view_469, permute_176);  convert_element_type_569 = None
        view_470: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_68, [512, 49, 1536]);  addmm_68 = None
        view_471: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_470, [512, 49, 3, 16, -1]);  view_470 = None
        permute_177: "bf16[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_471, [2, 0, 3, 1, 4]);  view_471 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_17 = torch.ops.aten.unbind.int(permute_177);  permute_177 = None
        getitem_127: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_17[0]
        getitem_128: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_17[1]
        getitem_129: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_17[2];  unbind_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_176: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_127, 0.1767766952966369);  getitem_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_178: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_128, [0, 1, 3, 2]);  getitem_128 = None
        expand_68: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_176, [512, 16, 49, 32]);  mul_176 = None
        clone_190: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
        view_472: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_190, [8192, 49, 32]);  clone_190 = None
        expand_69: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_178, [512, 16, 32, 49]);  permute_178 = None
        clone_191: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
        view_473: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_191, [8192, 32, 49]);  clone_191 = None
        constant_pad_nd_default_22: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_472, [0, 0, 0, 7, 0, 0])
        constant_pad_nd_default_23: "bf16[8192, 32, 56][1792, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_473, [0, 7, 0, 0, 0, 0])
        bmm_default_11: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_22, constant_pad_nd_default_23);  constant_pad_nd_default_22 = constant_pad_nd_default_23 = None
        slice_tensor_15: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_11, 1, 0, -7)
        slice_tensor_16: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_15, 2, 0, -7);  slice_tensor_15 = None
        view_474: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_16, [512, 16, 49, 49]);  slice_tensor_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_475: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_264, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_51: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_263, [view_475]);  view_475 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_476: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_51, [49, 49, -1]);  index_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_179: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_476, [2, 0, 1]);  view_476 = None
        clone_192: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_33: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_192, 0);  clone_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_186: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_474, unsqueeze_33);  view_474 = unsqueeze_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_477: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_186, [-1, 4, 16, 49, 49]);  add_186 = None
        unsqueeze_34: "f32[4, 1, 49, 49][2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_260, 1)
        unsqueeze_35: "f32[1, 4, 1, 49, 49][9604, 2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_34, 0);  unsqueeze_34 = None
        add_187: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_477, unsqueeze_35);  view_477 = unsqueeze_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_478: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_187, [-1, 16, 49, 49]);  add_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_17: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(view_478, [-1], True)
        sub_55: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_478, amax_17);  view_478 = None
        exp_17: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
        sum_18: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_49: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        convert_element_type_577: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_49, torch.bfloat16);  div_49 = None
        expand_70: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_577, [512, 16, 49, 49]);  convert_element_type_577 = None
        view_479: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_70, [8192, 49, 49]);  expand_70 = None
        expand_71: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_129, [512, 16, 49, 32]);  getitem_129 = None
        clone_194: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
        view_480: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_194, [8192, 49, 32]);  clone_194 = None
        constant_pad_nd_default_20: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_479, [0, 7, 0, 7, 0, 0])
        constant_pad_nd_default_21: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_480, [0, 0, 0, 7, 0, 0])
        bmm_default_10: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_20, constant_pad_nd_default_21);  constant_pad_nd_default_20 = constant_pad_nd_default_21 = None
        slice_tensor_14: "bf16[8192, 49, 32][1792, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_10, 1, 0, -7);  bmm_default_10 = None
        view_481: "bf16[512, 16, 49, 32][28672, 1792, 32, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_14, [512, 16, 49, 32]);  slice_tensor_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_180: "bf16[512, 49, 16, 32][28672, 32, 1792, 1]cuda:0" = torch.ops.aten.permute.default(view_481, [0, 2, 1, 3]);  view_481 = None
        clone_195: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_180, memory_format = torch.contiguous_format);  permute_180 = None
        view_482: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_195, [512, 49, 512]);  clone_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        convert_element_type_580: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_266, torch.bfloat16);  primals_266 = None
        convert_element_type_581: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_265, torch.bfloat16);  primals_265 = None
        view_483: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_482, [25088, 512]);  view_482 = None
        permute_181: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_581, [1, 0]);  convert_element_type_581 = None
        addmm_69: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_580, view_483, permute_181);  convert_element_type_580 = None
        view_484: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_69, [512, 49, 512]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_485: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_484, [-1, 7, 7, 512]);  view_484 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_486: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_485, [-1, 2, 2, 7, 7, 512]);  view_485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_182: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_486, [0, 1, 3, 2, 4, 5]);  view_486 = None
        clone_197: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_182, memory_format = torch.contiguous_format);  permute_182 = None
        view_487: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_197, [-1, 14, 14, 512]);  clone_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_52: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_487, [None, fmod_10]);  view_487 = None
        index_53: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_52, [None, None, fmod_10]);  index_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_32: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 32)
        inductor_random_default_13: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_32, 'rand');  inductor_lookup_seed_default_32 = None
        lt_32: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_13, 0.9260869547724724);  inductor_random_default_13 = None
        convert_element_type_585: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_32, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_50: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_585, 0.9260869547724724);  convert_element_type_585 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_177: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(index_53, div_50);  index_53 = div_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_190: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_465, mul_177);  mul_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_488: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_190, [128, -1, 512]);  add_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        convert_element_type_586: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_488, torch.float32)
        var_mean_38 = torch.ops.aten.var_mean.correction(convert_element_type_586, [2], correction = 0, keepdim = True)
        getitem_130: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_38[0]
        getitem_131: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_38[1];  var_mean_38 = None
        add_191: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_38: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
        sub_56: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_586, getitem_131);  convert_element_type_586 = None
        mul_178: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_38);  sub_56 = None
        mul_179: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_178, primals_267);  mul_178 = None
        add_192: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_179, primals_268);  mul_179 = primals_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_587: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_270, torch.bfloat16);  primals_270 = None
        convert_element_type_588: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_269, torch.bfloat16);  primals_269 = None
        convert_element_type_589: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_192, torch.bfloat16);  add_192 = None
        view_489: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_589, [25088, 512]);  convert_element_type_589 = None
        permute_183: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_588, [1, 0]);  convert_element_type_588 = None
        addmm_70: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_587, view_489, permute_183);  convert_element_type_587 = None
        view_490: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_70, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_593: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_490, torch.float32);  view_490 = None
        mul_180: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_593, 0.5)
        mul_181: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_593, 0.7071067811865476);  convert_element_type_593 = None
        erf_17: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_181);  mul_181 = None
        add_193: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_182: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_180, add_193);  mul_180 = add_193 = None
        convert_element_type_594: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_182, torch.bfloat16);  mul_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_595: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_272, torch.bfloat16);  primals_272 = None
        convert_element_type_596: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_271, torch.bfloat16);  primals_271 = None
        view_491: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_594, [25088, 2048]);  convert_element_type_594 = None
        permute_184: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_596, [1, 0]);  convert_element_type_596 = None
        addmm_71: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_595, view_491, permute_184);  convert_element_type_595 = None
        view_492: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_71, [128, 196, 512]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_33: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 33)
        inductor_random_default_12: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_33, 'rand');  inductor_lookup_seed_default_33 = None
        lt_33: "b8[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_12, 0.9260869547724724);  inductor_random_default_12 = None
        convert_element_type_600: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_33, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_51: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_600, 0.9260869547724724);  convert_element_type_600 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_183: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_492, div_51);  view_492 = div_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_194: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_488, mul_183);  mul_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_493: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_194, [128, 14, 14, 512]);  add_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        convert_element_type_601: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_493, torch.float32)
        var_mean_39 = torch.ops.aten.var_mean.correction(convert_element_type_601, [3], correction = 0, keepdim = True)
        getitem_132: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_39[0]
        getitem_133: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_39[1];  var_mean_39 = None
        add_195: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_132, 1e-05);  getitem_132 = None
        rsqrt_39: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_195);  add_195 = None
        sub_57: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_601, getitem_133);  convert_element_type_601 = None
        mul_184: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_39);  sub_57 = None
        mul_185: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_184, primals_273);  mul_184 = None
        add_196: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_185, primals_274);  mul_185 = primals_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_494: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_196, [128, 2, 7, 2, 7, 512]);  add_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_185: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_494, [0, 1, 3, 2, 4, 5]);  view_494 = None
        clone_200: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_185, memory_format = torch.contiguous_format);  permute_185 = None
        view_495: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_200, [-1, 7, 7, 512]);  clone_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_496: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_495, [-1, 49, 512]);  view_495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        convert_element_type_602: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_276, torch.bfloat16);  primals_276 = None
        convert_element_type_603: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_275, torch.bfloat16);  primals_275 = None
        convert_element_type_604: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_496, torch.bfloat16);  view_496 = None
        view_497: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_604, [25088, 512]);  convert_element_type_604 = None
        permute_186: "bf16[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_603, [1, 0]);  convert_element_type_603 = None
        addmm_72: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_602, view_497, permute_186);  convert_element_type_602 = None
        view_498: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_72, [512, 49, 1536]);  addmm_72 = None
        view_499: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_498, [512, 49, 3, 16, -1]);  view_498 = None
        permute_187: "bf16[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_499, [2, 0, 3, 1, 4]);  view_499 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_18 = torch.ops.aten.unbind.int(permute_187);  permute_187 = None
        getitem_134: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_18[0]
        getitem_135: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_18[1]
        getitem_136: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_18[2];  unbind_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_186: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_134, 0.1767766952966369);  getitem_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_188: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_135, [0, 1, 3, 2]);  getitem_135 = None
        expand_72: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_186, [512, 16, 49, 32]);  mul_186 = None
        clone_201: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
        view_500: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_201, [8192, 49, 32]);  clone_201 = None
        expand_73: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_188, [512, 16, 32, 49]);  permute_188 = None
        clone_202: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
        view_501: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_202, [8192, 32, 49]);  clone_202 = None
        constant_pad_nd_default_18: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_500, [0, 0, 0, 7, 0, 0])
        constant_pad_nd_default_19: "bf16[8192, 32, 56][1792, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_501, [0, 7, 0, 0, 0, 0])
        bmm_default_9: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_18, constant_pad_nd_default_19);  constant_pad_nd_default_18 = constant_pad_nd_default_19 = None
        slice_tensor_12: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_9, 1, 0, -7)
        slice_tensor_13: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_12, 2, 0, -7);  slice_tensor_12 = None
        view_502: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_13, [512, 16, 49, 49]);  slice_tensor_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_503: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_278, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_54: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_277, [view_503]);  view_503 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_504: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_54, [49, 49, -1]);  index_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_189: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_504, [2, 0, 1]);  view_504 = None
        clone_203: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_36: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_203, 0);  clone_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_197: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_502, unsqueeze_36);  view_502 = unsqueeze_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_18: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_197, [-1], True)
        sub_58: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_197, amax_18);  add_197 = None
        exp_18: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_58);  sub_58 = None
        sum_19: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_52: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        convert_element_type_610: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_52, torch.bfloat16);  div_52 = None
        expand_74: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_610, [512, 16, 49, 49]);  convert_element_type_610 = None
        view_505: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_74, [8192, 49, 49]);  expand_74 = None
        expand_75: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_136, [512, 16, 49, 32]);  getitem_136 = None
        clone_205: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
        view_506: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_205, [8192, 49, 32]);  clone_205 = None
        constant_pad_nd_default_16: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_505, [0, 7, 0, 7, 0, 0])
        constant_pad_nd_default_17: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_506, [0, 0, 0, 7, 0, 0])
        bmm_default_8: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_16, constant_pad_nd_default_17);  constant_pad_nd_default_16 = constant_pad_nd_default_17 = None
        slice_tensor_11: "bf16[8192, 49, 32][1792, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_8, 1, 0, -7);  bmm_default_8 = None
        view_507: "bf16[512, 16, 49, 32][28672, 1792, 32, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_11, [512, 16, 49, 32]);  slice_tensor_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_190: "bf16[512, 49, 16, 32][28672, 32, 1792, 1]cuda:0" = torch.ops.aten.permute.default(view_507, [0, 2, 1, 3]);  view_507 = None
        clone_206: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_190, memory_format = torch.contiguous_format);  permute_190 = None
        view_508: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_206, [512, 49, 512]);  clone_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        convert_element_type_613: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_280, torch.bfloat16);  primals_280 = None
        convert_element_type_614: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_279, torch.bfloat16);  primals_279 = None
        view_509: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_508, [25088, 512]);  view_508 = None
        permute_191: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_614, [1, 0]);  convert_element_type_614 = None
        addmm_73: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_613, view_509, permute_191);  convert_element_type_613 = None
        view_510: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_73, [512, 49, 512]);  addmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_511: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_510, [-1, 7, 7, 512]);  view_510 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_512: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_511, [-1, 2, 2, 7, 7, 512]);  view_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_192: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_512, [0, 1, 3, 2, 4, 5]);  view_512 = None
        clone_208: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_192, memory_format = torch.contiguous_format);  permute_192 = None
        view_513: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_208, [-1, 14, 14, 512]);  clone_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_34: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 34)
        inductor_random_default_11: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_34, 'rand');  inductor_lookup_seed_default_34 = None
        lt_34: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_11, 0.9217391312122345);  inductor_random_default_11 = None
        convert_element_type_618: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_34, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_53: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_618, 0.9217391312122345);  convert_element_type_618 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_187: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_513, div_53);  view_513 = div_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_198: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_493, mul_187);  mul_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_514: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_198, [128, -1, 512]);  add_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        convert_element_type_619: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_514, torch.float32)
        var_mean_40 = torch.ops.aten.var_mean.correction(convert_element_type_619, [2], correction = 0, keepdim = True)
        getitem_137: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_40[0]
        getitem_138: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_40[1];  var_mean_40 = None
        add_199: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_137, 1e-05);  getitem_137 = None
        rsqrt_40: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
        sub_59: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_619, getitem_138);  convert_element_type_619 = None
        mul_188: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_40);  sub_59 = None
        mul_189: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_188, primals_281);  mul_188 = None
        add_200: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_189, primals_282);  mul_189 = primals_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_620: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_284, torch.bfloat16);  primals_284 = None
        convert_element_type_621: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_283, torch.bfloat16);  primals_283 = None
        convert_element_type_622: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_200, torch.bfloat16);  add_200 = None
        view_515: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_622, [25088, 512]);  convert_element_type_622 = None
        permute_193: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_621, [1, 0]);  convert_element_type_621 = None
        addmm_74: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_620, view_515, permute_193);  convert_element_type_620 = None
        view_516: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_74, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_626: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_516, torch.float32);  view_516 = None
        mul_190: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_626, 0.5)
        mul_191: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_626, 0.7071067811865476);  convert_element_type_626 = None
        erf_18: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_191);  mul_191 = None
        add_201: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_192: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_190, add_201);  mul_190 = add_201 = None
        convert_element_type_627: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_192, torch.bfloat16);  mul_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_628: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_286, torch.bfloat16);  primals_286 = None
        convert_element_type_629: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_285, torch.bfloat16);  primals_285 = None
        view_517: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_627, [25088, 2048]);  convert_element_type_627 = None
        permute_194: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_629, [1, 0]);  convert_element_type_629 = None
        addmm_75: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_628, view_517, permute_194);  convert_element_type_628 = None
        view_518: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_75, [128, 196, 512]);  addmm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_35: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 35)
        inductor_random_default_10: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_35, 'rand');  inductor_lookup_seed_default_35 = None
        lt_35: "b8[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_10, 0.9217391312122345);  inductor_random_default_10 = None
        convert_element_type_633: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_35, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_54: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_633, 0.9217391312122345);  convert_element_type_633 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_193: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_518, div_54);  view_518 = div_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_202: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_514, mul_193);  mul_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_519: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_202, [128, 14, 14, 512]);  add_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        convert_element_type_634: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_519, torch.float32)
        var_mean_41 = torch.ops.aten.var_mean.correction(convert_element_type_634, [3], correction = 0, keepdim = True)
        getitem_139: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_41[0]
        getitem_140: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_41[1];  var_mean_41 = None
        add_203: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_139, 1e-05);  getitem_139 = None
        rsqrt_41: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
        sub_60: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_634, getitem_140);  convert_element_type_634 = None
        mul_194: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_41);  sub_60 = None
        mul_195: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_194, primals_287);  mul_194 = None
        add_204: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_195, primals_288);  mul_195 = primals_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_55: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(add_204, [None, fmod_8]);  add_204 = None
        index_56: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_55, [None, None, fmod_8]);  index_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_520: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(index_56, [128, 2, 7, 2, 7, 512]);  index_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_195: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_520, [0, 1, 3, 2, 4, 5]);  view_520 = None
        clone_211: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_195, memory_format = torch.contiguous_format);  permute_195 = None
        view_521: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_211, [-1, 7, 7, 512]);  clone_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_522: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_521, [-1, 49, 512]);  view_521 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        convert_element_type_635: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_291, torch.bfloat16);  primals_291 = None
        convert_element_type_636: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_290, torch.bfloat16);  primals_290 = None
        convert_element_type_637: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_522, torch.bfloat16);  view_522 = None
        view_523: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_637, [25088, 512]);  convert_element_type_637 = None
        permute_196: "bf16[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_636, [1, 0]);  convert_element_type_636 = None
        addmm_76: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_635, view_523, permute_196);  convert_element_type_635 = None
        view_524: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_76, [512, 49, 1536]);  addmm_76 = None
        view_525: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_524, [512, 49, 3, 16, -1]);  view_524 = None
        permute_197: "bf16[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_525, [2, 0, 3, 1, 4]);  view_525 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_19 = torch.ops.aten.unbind.int(permute_197);  permute_197 = None
        getitem_141: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_19[0]
        getitem_142: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_19[1]
        getitem_143: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_19[2];  unbind_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_196: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_141, 0.1767766952966369);  getitem_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_198: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_142, [0, 1, 3, 2]);  getitem_142 = None
        expand_76: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_196, [512, 16, 49, 32]);  mul_196 = None
        clone_212: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
        view_526: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_212, [8192, 49, 32]);  clone_212 = None
        expand_77: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_198, [512, 16, 32, 49]);  permute_198 = None
        clone_213: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
        view_527: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_213, [8192, 32, 49]);  clone_213 = None
        constant_pad_nd_default_14: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_526, [0, 0, 0, 7, 0, 0])
        constant_pad_nd_default_15: "bf16[8192, 32, 56][1792, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_527, [0, 7, 0, 0, 0, 0])
        bmm_default_7: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_14, constant_pad_nd_default_15);  constant_pad_nd_default_14 = constant_pad_nd_default_15 = None
        slice_tensor_9: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_7, 1, 0, -7)
        slice_tensor_10: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_9, 2, 0, -7);  slice_tensor_9 = None
        view_528: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_10, [512, 16, 49, 49]);  slice_tensor_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_529: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_293, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_57: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_292, [view_529]);  view_529 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_530: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_57, [49, 49, -1]);  index_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_199: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_530, [2, 0, 1]);  view_530 = None
        clone_214: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_37: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_214, 0);  clone_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_207: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_528, unsqueeze_37);  view_528 = unsqueeze_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_531: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_207, [-1, 4, 16, 49, 49]);  add_207 = None
        unsqueeze_38: "f32[4, 1, 49, 49][2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_289, 1)
        unsqueeze_39: "f32[1, 4, 1, 49, 49][9604, 2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_38, 0);  unsqueeze_38 = None
        add_208: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_531, unsqueeze_39);  view_531 = unsqueeze_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_532: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_208, [-1, 16, 49, 49]);  add_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_19: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(view_532, [-1], True)
        sub_61: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_532, amax_19);  view_532 = None
        exp_19: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_61);  sub_61 = None
        sum_20: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_55: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        convert_element_type_643: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_55, torch.bfloat16);  div_55 = None
        expand_78: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_643, [512, 16, 49, 49]);  convert_element_type_643 = None
        view_533: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_78, [8192, 49, 49]);  expand_78 = None
        expand_79: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_143, [512, 16, 49, 32]);  getitem_143 = None
        clone_216: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_79, memory_format = torch.contiguous_format);  expand_79 = None
        view_534: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_216, [8192, 49, 32]);  clone_216 = None
        constant_pad_nd_default_12: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_533, [0, 7, 0, 7, 0, 0])
        constant_pad_nd_default_13: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_534, [0, 0, 0, 7, 0, 0])
        bmm_default_6: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_12, constant_pad_nd_default_13);  constant_pad_nd_default_12 = constant_pad_nd_default_13 = None
        slice_tensor_8: "bf16[8192, 49, 32][1792, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_6, 1, 0, -7);  bmm_default_6 = None
        view_535: "bf16[512, 16, 49, 32][28672, 1792, 32, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_8, [512, 16, 49, 32]);  slice_tensor_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_200: "bf16[512, 49, 16, 32][28672, 32, 1792, 1]cuda:0" = torch.ops.aten.permute.default(view_535, [0, 2, 1, 3]);  view_535 = None
        clone_217: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_200, memory_format = torch.contiguous_format);  permute_200 = None
        view_536: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_217, [512, 49, 512]);  clone_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        convert_element_type_646: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_295, torch.bfloat16);  primals_295 = None
        convert_element_type_647: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_294, torch.bfloat16);  primals_294 = None
        view_537: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_536, [25088, 512]);  view_536 = None
        permute_201: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_647, [1, 0]);  convert_element_type_647 = None
        addmm_77: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_646, view_537, permute_201);  convert_element_type_646 = None
        view_538: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_77, [512, 49, 512]);  addmm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_539: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_538, [-1, 7, 7, 512]);  view_538 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_540: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_539, [-1, 2, 2, 7, 7, 512]);  view_539 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_202: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_540, [0, 1, 3, 2, 4, 5]);  view_540 = None
        clone_219: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_202, memory_format = torch.contiguous_format);  permute_202 = None
        view_541: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_219, [-1, 14, 14, 512]);  clone_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_58: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_541, [None, fmod_10]);  view_541 = None
        index_59: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_58, [None, None, fmod_10]);  index_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_36: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 36)
        inductor_random_default_9: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_36, 'rand');  inductor_lookup_seed_default_36 = None
        lt_36: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_9, 0.917391300201416);  inductor_random_default_9 = None
        convert_element_type_651: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_36, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_56: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_651, 0.917391300201416);  convert_element_type_651 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_197: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(index_59, div_56);  index_59 = div_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_211: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_519, mul_197);  mul_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_542: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_211, [128, -1, 512]);  add_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        convert_element_type_652: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_542, torch.float32)
        var_mean_42 = torch.ops.aten.var_mean.correction(convert_element_type_652, [2], correction = 0, keepdim = True)
        getitem_144: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_42[0]
        getitem_145: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_42[1];  var_mean_42 = None
        add_212: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_144, 1e-05);  getitem_144 = None
        rsqrt_42: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_212);  add_212 = None
        sub_62: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_652, getitem_145);  convert_element_type_652 = None
        mul_198: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_42);  sub_62 = None
        mul_199: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_198, primals_296);  mul_198 = None
        add_213: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_199, primals_297);  mul_199 = primals_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_653: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_299, torch.bfloat16);  primals_299 = None
        convert_element_type_654: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_298, torch.bfloat16);  primals_298 = None
        convert_element_type_655: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_213, torch.bfloat16);  add_213 = None
        view_543: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_655, [25088, 512]);  convert_element_type_655 = None
        permute_203: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_654, [1, 0]);  convert_element_type_654 = None
        addmm_78: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_653, view_543, permute_203);  convert_element_type_653 = None
        view_544: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_78, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_659: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_544, torch.float32);  view_544 = None
        mul_200: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_659, 0.5)
        mul_201: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_659, 0.7071067811865476);  convert_element_type_659 = None
        erf_19: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_201);  mul_201 = None
        add_214: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_202: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_200, add_214);  mul_200 = add_214 = None
        convert_element_type_660: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_202, torch.bfloat16);  mul_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_661: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_301, torch.bfloat16);  primals_301 = None
        convert_element_type_662: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_300, torch.bfloat16);  primals_300 = None
        view_545: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_660, [25088, 2048]);  convert_element_type_660 = None
        permute_204: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_662, [1, 0]);  convert_element_type_662 = None
        addmm_79: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_661, view_545, permute_204);  convert_element_type_661 = None
        view_546: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_79, [128, 196, 512]);  addmm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_37: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 37)
        inductor_random_default_8: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_37, 'rand');  inductor_lookup_seed_default_37 = None
        lt_37: "b8[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_8, 0.917391300201416);  inductor_random_default_8 = None
        convert_element_type_666: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_37, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_57: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_666, 0.917391300201416);  convert_element_type_666 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_203: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_546, div_57);  view_546 = div_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_215: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_542, mul_203);  mul_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_547: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_215, [128, 14, 14, 512]);  add_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        convert_element_type_667: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_547, torch.float32)
        var_mean_43 = torch.ops.aten.var_mean.correction(convert_element_type_667, [3], correction = 0, keepdim = True)
        getitem_146: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_43[0]
        getitem_147: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_43[1];  var_mean_43 = None
        add_216: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_146, 1e-05);  getitem_146 = None
        rsqrt_43: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_216);  add_216 = None
        sub_63: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_667, getitem_147);  convert_element_type_667 = None
        mul_204: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_43);  sub_63 = None
        mul_205: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_204, primals_302);  mul_204 = None
        add_217: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_205, primals_303);  mul_205 = primals_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_548: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_217, [128, 2, 7, 2, 7, 512]);  add_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_205: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_548, [0, 1, 3, 2, 4, 5]);  view_548 = None
        clone_222: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
        view_549: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_222, [-1, 7, 7, 512]);  clone_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_550: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_549, [-1, 49, 512]);  view_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        convert_element_type_668: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_305, torch.bfloat16);  primals_305 = None
        convert_element_type_669: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_304, torch.bfloat16);  primals_304 = None
        convert_element_type_670: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_550, torch.bfloat16);  view_550 = None
        view_551: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_670, [25088, 512]);  convert_element_type_670 = None
        permute_206: "bf16[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_669, [1, 0]);  convert_element_type_669 = None
        addmm_80: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_668, view_551, permute_206);  convert_element_type_668 = None
        view_552: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_80, [512, 49, 1536]);  addmm_80 = None
        view_553: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_552, [512, 49, 3, 16, -1]);  view_552 = None
        permute_207: "bf16[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_553, [2, 0, 3, 1, 4]);  view_553 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_20 = torch.ops.aten.unbind.int(permute_207);  permute_207 = None
        getitem_148: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_20[0]
        getitem_149: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_20[1]
        getitem_150: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_20[2];  unbind_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_206: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_148, 0.1767766952966369);  getitem_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_208: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_149, [0, 1, 3, 2]);  getitem_149 = None
        expand_80: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_206, [512, 16, 49, 32]);  mul_206 = None
        clone_223: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
        view_554: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_223, [8192, 49, 32]);  clone_223 = None
        expand_81: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_208, [512, 16, 32, 49]);  permute_208 = None
        clone_224: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
        view_555: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_224, [8192, 32, 49]);  clone_224 = None
        constant_pad_nd_default_10: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_554, [0, 0, 0, 7, 0, 0])
        constant_pad_nd_default_11: "bf16[8192, 32, 56][1792, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_555, [0, 7, 0, 0, 0, 0])
        bmm_default_5: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_10, constant_pad_nd_default_11);  constant_pad_nd_default_10 = constant_pad_nd_default_11 = None
        slice_tensor_6: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_5, 1, 0, -7)
        slice_tensor_7: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_6, 2, 0, -7);  slice_tensor_6 = None
        view_556: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_7, [512, 16, 49, 49]);  slice_tensor_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_557: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_307, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_60: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_306, [view_557]);  view_557 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_558: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_60, [49, 49, -1]);  index_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_209: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_558, [2, 0, 1]);  view_558 = None
        clone_225: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_209, memory_format = torch.contiguous_format);  permute_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_40: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_225, 0);  clone_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_218: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_556, unsqueeze_40);  view_556 = unsqueeze_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_20: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_218, [-1], True)
        sub_64: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_218, amax_20);  add_218 = None
        exp_20: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_64);  sub_64 = None
        sum_21: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_58: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        convert_element_type_676: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_58, torch.bfloat16);  div_58 = None
        expand_82: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_676, [512, 16, 49, 49]);  convert_element_type_676 = None
        view_559: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_82, [8192, 49, 49]);  expand_82 = None
        expand_83: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_150, [512, 16, 49, 32]);  getitem_150 = None
        clone_227: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_83, memory_format = torch.contiguous_format);  expand_83 = None
        view_560: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_227, [8192, 49, 32]);  clone_227 = None
        constant_pad_nd_default_8: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_559, [0, 7, 0, 7, 0, 0])
        constant_pad_nd_default_9: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_560, [0, 0, 0, 7, 0, 0])
        bmm_default_4: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_8, constant_pad_nd_default_9);  constant_pad_nd_default_8 = constant_pad_nd_default_9 = None
        slice_tensor_5: "bf16[8192, 49, 32][1792, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_4, 1, 0, -7);  bmm_default_4 = None
        view_561: "bf16[512, 16, 49, 32][28672, 1792, 32, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_5, [512, 16, 49, 32]);  slice_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_210: "bf16[512, 49, 16, 32][28672, 32, 1792, 1]cuda:0" = torch.ops.aten.permute.default(view_561, [0, 2, 1, 3]);  view_561 = None
        clone_228: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None
        view_562: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_228, [512, 49, 512]);  clone_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        convert_element_type_679: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_309, torch.bfloat16);  primals_309 = None
        convert_element_type_680: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_308, torch.bfloat16);  primals_308 = None
        view_563: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_562, [25088, 512]);  view_562 = None
        permute_211: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_680, [1, 0]);  convert_element_type_680 = None
        addmm_81: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_679, view_563, permute_211);  convert_element_type_679 = None
        view_564: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_81, [512, 49, 512]);  addmm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_565: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_564, [-1, 7, 7, 512]);  view_564 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_566: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_565, [-1, 2, 2, 7, 7, 512]);  view_565 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_212: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_566, [0, 1, 3, 2, 4, 5]);  view_566 = None
        clone_230: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_212, memory_format = torch.contiguous_format);  permute_212 = None
        view_567: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_230, [-1, 14, 14, 512]);  clone_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_38: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 38)
        inductor_random_default_7: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_38, 'rand');  inductor_lookup_seed_default_38 = None
        lt_38: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_7, 0.9130434766411781);  inductor_random_default_7 = None
        convert_element_type_684: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_38, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_59: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_684, 0.9130434766411781);  convert_element_type_684 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_207: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_567, div_59);  view_567 = div_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_219: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_547, mul_207);  mul_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_568: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_219, [128, -1, 512]);  add_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        convert_element_type_685: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_568, torch.float32)
        var_mean_44 = torch.ops.aten.var_mean.correction(convert_element_type_685, [2], correction = 0, keepdim = True)
        getitem_151: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_44[0]
        getitem_152: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_44[1];  var_mean_44 = None
        add_220: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_151, 1e-05);  getitem_151 = None
        rsqrt_44: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_220);  add_220 = None
        sub_65: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_685, getitem_152);  convert_element_type_685 = None
        mul_208: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_44);  sub_65 = None
        mul_209: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_208, primals_310);  mul_208 = None
        add_221: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_209, primals_311);  mul_209 = primals_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_686: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_313, torch.bfloat16);  primals_313 = None
        convert_element_type_687: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_312, torch.bfloat16);  primals_312 = None
        convert_element_type_688: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_221, torch.bfloat16);  add_221 = None
        view_569: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_688, [25088, 512]);  convert_element_type_688 = None
        permute_213: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_687, [1, 0]);  convert_element_type_687 = None
        addmm_82: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_686, view_569, permute_213);  convert_element_type_686 = None
        view_570: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_82, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_692: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_570, torch.float32);  view_570 = None
        mul_210: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_692, 0.5)
        mul_211: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_692, 0.7071067811865476);  convert_element_type_692 = None
        erf_20: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_211);  mul_211 = None
        add_222: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_212: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_210, add_222);  mul_210 = add_222 = None
        convert_element_type_693: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_212, torch.bfloat16);  mul_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_694: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_315, torch.bfloat16);  primals_315 = None
        convert_element_type_695: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_314, torch.bfloat16);  primals_314 = None
        view_571: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_693, [25088, 2048]);  convert_element_type_693 = None
        permute_214: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_695, [1, 0]);  convert_element_type_695 = None
        addmm_83: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_694, view_571, permute_214);  convert_element_type_694 = None
        view_572: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_83, [128, 196, 512]);  addmm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_39: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 39)
        inductor_random_default_6: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_39, 'rand');  inductor_lookup_seed_default_39 = None
        lt_39: "b8[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_6, 0.9130434766411781);  inductor_random_default_6 = None
        convert_element_type_699: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_39, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_60: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_699, 0.9130434766411781);  convert_element_type_699 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_213: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_572, div_60);  view_572 = div_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_223: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_568, mul_213);  mul_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_573: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_223, [128, 14, 14, 512]);  add_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        convert_element_type_700: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_573, torch.float32)
        var_mean_45 = torch.ops.aten.var_mean.correction(convert_element_type_700, [3], correction = 0, keepdim = True)
        getitem_153: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_45[0]
        getitem_154: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = var_mean_45[1];  var_mean_45 = None
        add_224: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_153, 1e-05);  getitem_153 = None
        rsqrt_45: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_224);  add_224 = None
        sub_66: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_700, getitem_154);  convert_element_type_700 = None
        mul_214: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_45);  sub_66 = None
        mul_215: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_214, primals_316);  mul_214 = None
        add_225: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_215, primals_317);  mul_215 = primals_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_61: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(add_225, [None, fmod_8]);  add_225 = None
        index_62: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_61, [None, None, fmod_8]);  index_61 = fmod_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_574: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(index_62, [128, 2, 7, 2, 7, 512]);  index_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_215: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_574, [0, 1, 3, 2, 4, 5]);  view_574 = None
        clone_233: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_215, memory_format = torch.contiguous_format);  permute_215 = None
        view_575: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_233, [-1, 7, 7, 512]);  clone_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_576: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_575, [-1, 49, 512]);  view_575 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        convert_element_type_701: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_320, torch.bfloat16);  primals_320 = None
        convert_element_type_702: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_319, torch.bfloat16);  primals_319 = None
        convert_element_type_703: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_576, torch.bfloat16);  view_576 = None
        view_577: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_703, [25088, 512]);  convert_element_type_703 = None
        permute_216: "bf16[512, 1536][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_702, [1, 0]);  convert_element_type_702 = None
        addmm_84: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_701, view_577, permute_216);  convert_element_type_701 = None
        view_578: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_84, [512, 49, 1536]);  addmm_84 = None
        view_579: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_578, [512, 49, 3, 16, -1]);  view_578 = None
        permute_217: "bf16[3, 512, 16, 49, 32][512, 75264, 32, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_579, [2, 0, 3, 1, 4]);  view_579 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_21 = torch.ops.aten.unbind.int(permute_217);  permute_217 = None
        getitem_155: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_21[0]
        getitem_156: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_21[1]
        getitem_157: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = unbind_21[2];  unbind_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_216: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_155, 0.1767766952966369);  getitem_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_218: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.permute.default(getitem_156, [0, 1, 3, 2]);  getitem_156 = None
        expand_84: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.expand.default(mul_216, [512, 16, 49, 32]);  mul_216 = None
        clone_234: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
        view_580: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_234, [8192, 49, 32]);  clone_234 = None
        expand_85: "bf16[512, 16, 32, 49][75264, 32, 1, 1536]cuda:0" = torch.ops.aten.expand.default(permute_218, [512, 16, 32, 49]);  permute_218 = None
        clone_235: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_85, memory_format = torch.contiguous_format);  expand_85 = None
        view_581: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_235, [8192, 32, 49]);  clone_235 = None
        constant_pad_nd_default_6: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_580, [0, 0, 0, 7, 0, 0])
        constant_pad_nd_default_7: "bf16[8192, 32, 56][1792, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_581, [0, 7, 0, 0, 0, 0])
        bmm_default_3: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_6, constant_pad_nd_default_7);  constant_pad_nd_default_6 = constant_pad_nd_default_7 = None
        slice_tensor_3: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_3, 1, 0, -7)
        slice_tensor_4: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_3, 2, 0, -7);  slice_tensor_3 = None
        view_582: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_4, [512, 16, 49, 49]);  slice_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_583: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_322, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_63: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_321, [view_583]);  view_583 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_584: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_63, [49, 49, -1]);  index_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_219: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_584, [2, 0, 1]);  view_584 = None
        clone_236: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_219, memory_format = torch.contiguous_format);  permute_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_41: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_236, 0);  clone_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_228: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_582, unsqueeze_41);  view_582 = unsqueeze_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_585: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_228, [-1, 4, 16, 49, 49]);  add_228 = None
        unsqueeze_42: "f32[4, 1, 49, 49][2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_318, 1)
        unsqueeze_43: "f32[1, 4, 1, 49, 49][9604, 2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_42, 0);  unsqueeze_42 = None
        add_229: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_585, unsqueeze_43);  view_585 = unsqueeze_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_586: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_229, [-1, 16, 49, 49]);  add_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_21: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(view_586, [-1], True)
        sub_67: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_586, amax_21);  view_586 = None
        exp_21: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_67);  sub_67 = None
        sum_22: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_61: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        convert_element_type_709: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_61, torch.bfloat16);  div_61 = None
        expand_86: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_709, [512, 16, 49, 49]);  convert_element_type_709 = None
        view_587: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_86, [8192, 49, 49]);  expand_86 = None
        expand_87: "bf16[512, 16, 49, 32][75264, 32, 1536, 1]cuda:0" = torch.ops.aten.expand.default(getitem_157, [512, 16, 49, 32]);  getitem_157 = None
        clone_238: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_87, memory_format = torch.contiguous_format);  expand_87 = None
        view_588: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_238, [8192, 49, 32]);  clone_238 = None
        constant_pad_nd_default_4: "bf16[8192, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_587, [0, 7, 0, 7, 0, 0])
        constant_pad_nd_default_5: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_588, [0, 0, 0, 7, 0, 0])
        bmm_default_2: "bf16[8192, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_4, constant_pad_nd_default_5);  constant_pad_nd_default_4 = constant_pad_nd_default_5 = None
        slice_tensor_2: "bf16[8192, 49, 32][1792, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_2, 1, 0, -7);  bmm_default_2 = None
        view_589: "bf16[512, 16, 49, 32][28672, 1792, 32, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_2, [512, 16, 49, 32]);  slice_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_220: "bf16[512, 49, 16, 32][28672, 32, 1792, 1]cuda:0" = torch.ops.aten.permute.default(view_589, [0, 2, 1, 3]);  view_589 = None
        clone_239: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_220, memory_format = torch.contiguous_format);  permute_220 = None
        view_590: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_239, [512, 49, 512]);  clone_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        convert_element_type_712: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_324, torch.bfloat16);  primals_324 = None
        convert_element_type_713: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_323, torch.bfloat16);  primals_323 = None
        view_591: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_590, [25088, 512]);  view_590 = None
        permute_221: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_713, [1, 0]);  convert_element_type_713 = None
        addmm_85: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_712, view_591, permute_221);  convert_element_type_712 = None
        view_592: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_85, [512, 49, 512]);  addmm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_593: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_592, [-1, 7, 7, 512]);  view_592 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_594: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_593, [-1, 2, 2, 7, 7, 512]);  view_593 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_222: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_594, [0, 1, 3, 2, 4, 5]);  view_594 = None
        clone_241: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_222, memory_format = torch.contiguous_format);  permute_222 = None
        view_595: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_241, [-1, 14, 14, 512]);  clone_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_64: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_595, [None, fmod_10]);  view_595 = None
        index_65: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_64, [None, None, fmod_10]);  index_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_40: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 40)
        inductor_random_default_5: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_40, 'rand');  inductor_lookup_seed_default_40 = None
        lt_40: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_5, 0.9086956530809402);  inductor_random_default_5 = None
        convert_element_type_717: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_40, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_62: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_717, 0.9086956530809402);  convert_element_type_717 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_217: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(index_65, div_62);  index_65 = div_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_232: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_573, mul_217);  mul_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_596: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_232, [128, -1, 512]);  add_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        convert_element_type_718: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_596, torch.float32)
        var_mean_46 = torch.ops.aten.var_mean.correction(convert_element_type_718, [2], correction = 0, keepdim = True)
        getitem_158: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_46[0]
        getitem_159: "f32[128, 196, 1][196, 1, 1]cuda:0" = var_mean_46[1];  var_mean_46 = None
        add_233: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_158, 1e-05);  getitem_158 = None
        rsqrt_46: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_233);  add_233 = None
        sub_68: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_718, getitem_159);  convert_element_type_718 = None
        mul_218: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_46);  sub_68 = None
        mul_219: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_218, primals_325);  mul_218 = None
        add_234: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_219, primals_326);  mul_219 = primals_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_719: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_328, torch.bfloat16);  primals_328 = None
        convert_element_type_720: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_327, torch.bfloat16);  primals_327 = None
        convert_element_type_721: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_234, torch.bfloat16);  add_234 = None
        view_597: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_721, [25088, 512]);  convert_element_type_721 = None
        permute_223: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_720, [1, 0]);  convert_element_type_720 = None
        addmm_86: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_719, view_597, permute_223);  convert_element_type_719 = None
        view_598: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_86, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_725: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_598, torch.float32);  view_598 = None
        mul_220: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_725, 0.5)
        mul_221: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_725, 0.7071067811865476);  convert_element_type_725 = None
        erf_21: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_221);  mul_221 = None
        add_235: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_222: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_220, add_235);  mul_220 = add_235 = None
        convert_element_type_726: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_222, torch.bfloat16);  mul_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_727: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_330, torch.bfloat16);  primals_330 = None
        convert_element_type_728: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_329, torch.bfloat16);  primals_329 = None
        view_599: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_726, [25088, 2048]);  convert_element_type_726 = None
        permute_224: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_728, [1, 0]);  convert_element_type_728 = None
        addmm_87: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_727, view_599, permute_224);  convert_element_type_727 = None
        view_600: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_87, [128, 196, 512]);  addmm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_41: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 41)
        inductor_random_default_4: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_41, 'rand');  inductor_lookup_seed_default_41 = None
        lt_41: "b8[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_4, 0.9086956530809402);  inductor_random_default_4 = None
        convert_element_type_732: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_41, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_63: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_732, 0.9086956530809402);  convert_element_type_732 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_223: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_600, div_63);  view_600 = div_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_236: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_596, mul_223);  mul_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_601: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_236, [128, 14, 14, 512]);  add_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:539 in forward, code: x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        view_602: "bf16[128, 7, 2, 7, 2, 512][100352, 14336, 7168, 1024, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_601, [128, 7, 2, 7, 2, 512]);  view_601 = None
        permute_225: "bf16[128, 7, 7, 2, 2, 512][100352, 14336, 1024, 512, 7168, 1]cuda:0" = torch.ops.aten.permute.default(view_602, [0, 1, 3, 4, 2, 5]);  view_602 = None
        clone_244: "bf16[128, 7, 7, 2, 2, 512][100352, 14336, 2048, 1024, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_225, memory_format = torch.contiguous_format);  permute_225 = None
        view_603: "bf16[128, 7, 7, 2048][100352, 14336, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_244, [128, 7, 7, 2048]);  clone_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:540 in forward, code: x = self.norm(x)
        convert_element_type_733: "f32[128, 7, 7, 2048][100352, 14336, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_603, torch.float32)
        var_mean_47 = torch.ops.aten.var_mean.correction(convert_element_type_733, [3], correction = 0, keepdim = True)
        getitem_160: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = var_mean_47[0]
        getitem_161: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = var_mean_47[1];  var_mean_47 = None
        add_237: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_160, 1e-05);  getitem_160 = None
        rsqrt_47: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_237);  add_237 = None
        sub_69: "f32[128, 7, 7, 2048][100352, 14336, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_733, getitem_161);  convert_element_type_733 = None
        mul_224: "f32[128, 7, 7, 2048][100352, 14336, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_47);  sub_69 = None
        mul_225: "f32[128, 7, 7, 2048][100352, 14336, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_224, primals_331);  mul_224 = None
        add_238: "f32[128, 7, 7, 2048][100352, 14336, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_225, primals_332);  mul_225 = primals_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:541 in forward, code: x = self.reduction(x)
        convert_element_type_734: "bf16[1024, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_333, torch.bfloat16);  primals_333 = None
        convert_element_type_735: "bf16[128, 7, 7, 2048][100352, 14336, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_238, torch.bfloat16);  add_238 = None
        permute_226: "bf16[2048, 1024][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_734, [1, 0]);  convert_element_type_734 = None
        view_604: "bf16[6272, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_735, [6272, 2048]);  convert_element_type_735 = None
        mm_2: "bf16[6272, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_604, permute_226)
        view_605: "bf16[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [128, 7, 7, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        convert_element_type_738: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_605, torch.float32)
        var_mean_48 = torch.ops.aten.var_mean.correction(convert_element_type_738, [3], correction = 0, keepdim = True)
        getitem_162: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = var_mean_48[0]
        getitem_163: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = var_mean_48[1];  var_mean_48 = None
        add_239: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_162, 1e-05);  getitem_162 = None
        rsqrt_48: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_239);  add_239 = None
        sub_70: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_738, getitem_163);  convert_element_type_738 = None
        mul_226: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_48);  sub_70 = None
        mul_227: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_226, primals_334);  mul_226 = None
        add_240: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_227, primals_335);  mul_227 = primals_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_606: "f32[128, 1, 7, 1, 7, 1024][50176, 50176, 7168, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(add_240, [128, 1, 7, 1, 7, 1024]);  add_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_227: "f32[128, 1, 1, 7, 7, 1024][50176, 50176, 7168, 7168, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_606, [0, 1, 3, 2, 4, 5]);  view_606 = None
        view_607: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_227, [-1, 7, 7, 1024]);  permute_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_608: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_607, [-1, 49, 1024]);  view_607 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        convert_element_type_739: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_337, torch.bfloat16);  primals_337 = None
        convert_element_type_740: "bf16[3072, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_336, torch.bfloat16);  primals_336 = None
        convert_element_type_741: "bf16[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_608, torch.bfloat16);  view_608 = None
        view_609: "bf16[6272, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_741, [6272, 1024]);  convert_element_type_741 = None
        permute_228: "bf16[1024, 3072][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_740, [1, 0]);  convert_element_type_740 = None
        addmm_88: "bf16[6272, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_739, view_609, permute_228);  convert_element_type_739 = None
        view_610: "bf16[128, 49, 3072][150528, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_88, [128, 49, 3072]);  addmm_88 = None
        view_611: "bf16[128, 49, 3, 32, 32][150528, 3072, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_610, [128, 49, 3, 32, -1]);  view_610 = None
        permute_229: "bf16[3, 128, 32, 49, 32][1024, 150528, 32, 3072, 1]cuda:0" = torch.ops.aten.permute.default(view_611, [2, 0, 3, 1, 4]);  view_611 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_22 = torch.ops.aten.unbind.int(permute_229);  permute_229 = None
        getitem_164: "bf16[128, 32, 49, 32][150528, 32, 3072, 1]cuda:0" = unbind_22[0]
        getitem_165: "bf16[128, 32, 49, 32][150528, 32, 3072, 1]cuda:0" = unbind_22[1]
        getitem_166: "bf16[128, 32, 49, 32][150528, 32, 3072, 1]cuda:0" = unbind_22[2];  unbind_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_228: "bf16[128, 32, 49, 32][50176, 32, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_164, 0.1767766952966369);  getitem_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_230: "bf16[128, 32, 32, 49][150528, 32, 1, 3072]cuda:0" = torch.ops.aten.permute.default(getitem_165, [0, 1, 3, 2]);  getitem_165 = None
        expand_88: "bf16[128, 32, 49, 32][50176, 32, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_228, [128, 32, 49, 32]);  mul_228 = None
        clone_245: "bf16[128, 32, 49, 32][50176, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
        view_612: "bf16[4096, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_245, [4096, 49, 32]);  clone_245 = None
        expand_89: "bf16[128, 32, 32, 49][150528, 32, 1, 3072]cuda:0" = torch.ops.aten.expand.default(permute_230, [128, 32, 32, 49]);  permute_230 = None
        clone_246: "bf16[128, 32, 32, 49][50176, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
        view_613: "bf16[4096, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_246, [4096, 32, 49]);  clone_246 = None
        bmm_44: "bf16[4096, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_612, view_613)
        view_614: "bf16[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_44, [128, 32, 49, 49])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_615: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_339, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_66: "f32[2401, 32][32, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_338, [view_615]);  view_615 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_616: "f32[49, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(index_66, [49, 49, -1]);  index_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_231: "f32[32, 49, 49][1, 1568, 32]cuda:0" = torch.ops.aten.permute.default(view_616, [2, 0, 1]);  view_616 = None
        clone_247: "f32[32, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_231, memory_format = torch.contiguous_format);  permute_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_44: "f32[1, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_247, 0);  clone_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_241: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_614, unsqueeze_44);  view_614 = unsqueeze_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_22: "f32[128, 32, 49, 1][1568, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_241, [-1], True)
        sub_71: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_241, amax_22);  add_241 = None
        exp_22: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_71);  sub_71 = None
        sum_23: "f32[128, 32, 49, 1][1568, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_64: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        convert_element_type_747: "bf16[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_64, torch.bfloat16);  div_64 = None
        expand_90: "bf16[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_747, [128, 32, 49, 49]);  convert_element_type_747 = None
        view_617: "bf16[4096, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_90, [4096, 49, 49]);  expand_90 = None
        expand_91: "bf16[128, 32, 49, 32][150528, 32, 3072, 1]cuda:0" = torch.ops.aten.expand.default(getitem_166, [128, 32, 49, 32]);  getitem_166 = None
        clone_249: "bf16[128, 32, 49, 32][50176, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_91, memory_format = torch.contiguous_format);  expand_91 = None
        view_618: "bf16[4096, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_249, [4096, 49, 32]);  clone_249 = None
        constant_pad_nd_default_2: "bf16[4096, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_617, [0, 7, 0, 7, 0, 0])
        constant_pad_nd_default_3: "bf16[4096, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_618, [0, 0, 0, 7, 0, 0])
        bmm_default_1: "bf16[4096, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default_2, constant_pad_nd_default_3);  constant_pad_nd_default_2 = constant_pad_nd_default_3 = None
        slice_tensor_1: "bf16[4096, 49, 32][1792, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_1, 1, 0, -7);  bmm_default_1 = None
        view_619: "bf16[128, 32, 49, 32][57344, 1792, 32, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_1, [128, 32, 49, 32]);  slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_232: "bf16[128, 49, 32, 32][57344, 32, 1792, 1]cuda:0" = torch.ops.aten.permute.default(view_619, [0, 2, 1, 3]);  view_619 = None
        clone_250: "bf16[128, 49, 32, 32][50176, 1024, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_232, memory_format = torch.contiguous_format);  permute_232 = None
        view_620: "bf16[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_250, [128, 49, 1024]);  clone_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        convert_element_type_750: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_341, torch.bfloat16);  primals_341 = None
        convert_element_type_751: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_340, torch.bfloat16);  primals_340 = None
        view_621: "bf16[6272, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_620, [6272, 1024]);  view_620 = None
        permute_233: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_751, [1, 0]);  convert_element_type_751 = None
        addmm_89: "bf16[6272, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_750, view_621, permute_233);  convert_element_type_750 = None
        view_622: "bf16[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_89, [128, 49, 1024]);  addmm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_623: "bf16[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_622, [-1, 7, 7, 1024]);  view_622 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_624: "bf16[128, 1, 1, 7, 7, 1024][50176, 50176, 50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_623, [-1, 1, 1, 7, 7, 1024]);  view_623 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_234: "bf16[128, 1, 7, 1, 7, 1024][50176, 50176, 7168, 50176, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_624, [0, 1, 3, 2, 4, 5]);  view_624 = None
        view_625: "bf16[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_234, [-1, 7, 7, 1024]);  permute_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_42: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 42)
        inductor_random_default_3: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_42, 'rand');  inductor_lookup_seed_default_42 = None
        lt_42: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_3, 0.9043478220701218);  inductor_random_default_3 = None
        convert_element_type_755: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_42, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_65: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_755, 0.9043478220701218);  convert_element_type_755 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_229: "bf16[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_625, div_65);  view_625 = div_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_242: "bf16[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_605, mul_229);  view_605 = mul_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_626: "bf16[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(add_242, [128, -1, 1024]);  add_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        convert_element_type_756: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_626, torch.float32)
        var_mean_49 = torch.ops.aten.var_mean.correction(convert_element_type_756, [2], correction = 0, keepdim = True)
        getitem_167: "f32[128, 49, 1][49, 1, 1]cuda:0" = var_mean_49[0]
        getitem_168: "f32[128, 49, 1][49, 1, 1]cuda:0" = var_mean_49[1];  var_mean_49 = None
        add_243: "f32[128, 49, 1][49, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_167, 1e-05);  getitem_167 = None
        rsqrt_49: "f32[128, 49, 1][49, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_243);  add_243 = None
        sub_72: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_756, getitem_168);  convert_element_type_756 = None
        mul_230: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_49);  sub_72 = None
        mul_231: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_230, primals_342);  mul_230 = None
        add_244: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_231, primals_343);  mul_231 = primals_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_757: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_345, torch.bfloat16);  primals_345 = None
        convert_element_type_758: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_344, torch.bfloat16);  primals_344 = None
        convert_element_type_759: "bf16[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_244, torch.bfloat16);  add_244 = None
        view_627: "bf16[6272, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_759, [6272, 1024]);  convert_element_type_759 = None
        permute_235: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_758, [1, 0]);  convert_element_type_758 = None
        addmm_90: "bf16[6272, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_757, view_627, permute_235);  convert_element_type_757 = None
        view_628: "bf16[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_90, [128, 49, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_763: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_628, torch.float32);  view_628 = None
        mul_232: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_763, 0.5)
        mul_233: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_763, 0.7071067811865476);  convert_element_type_763 = None
        erf_22: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_233);  mul_233 = None
        add_245: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_234: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_232, add_245);  mul_232 = add_245 = None
        convert_element_type_764: "bf16[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_234, torch.bfloat16);  mul_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_765: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_347, torch.bfloat16);  primals_347 = None
        convert_element_type_766: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_346, torch.bfloat16);  primals_346 = None
        view_629: "bf16[6272, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_764, [6272, 4096]);  convert_element_type_764 = None
        permute_236: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_766, [1, 0]);  convert_element_type_766 = None
        addmm_91: "bf16[6272, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_765, view_629, permute_236);  convert_element_type_765 = None
        view_630: "bf16[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_91, [128, 49, 1024]);  addmm_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_43: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 43)
        inductor_random_default_2: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_43, 'rand');  inductor_lookup_seed_default_43 = None
        lt_43: "b8[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_2, 0.9043478220701218);  inductor_random_default_2 = None
        convert_element_type_770: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_43, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_66: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_770, 0.9043478220701218);  convert_element_type_770 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_235: "bf16[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_630, div_66);  view_630 = div_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_246: "bf16[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_626, mul_235);  mul_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_631: "bf16[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(add_246, [128, 7, 7, 1024]);  add_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        convert_element_type_771: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_631, torch.float32)
        var_mean_50 = torch.ops.aten.var_mean.correction(convert_element_type_771, [3], correction = 0, keepdim = True)
        getitem_169: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = var_mean_50[0]
        getitem_170: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = var_mean_50[1];  var_mean_50 = None
        add_247: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_169, 1e-05);  getitem_169 = None
        rsqrt_50: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_247);  add_247 = None
        sub_73: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_771, getitem_170);  convert_element_type_771 = None
        mul_236: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_50);  sub_73 = None
        mul_237: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_236, primals_348);  mul_236 = None
        add_248: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_237, primals_349);  mul_237 = primals_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_632: "f32[128, 1, 7, 1, 7, 1024][50176, 50176, 7168, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(add_248, [128, 1, 7, 1, 7, 1024]);  add_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_237: "f32[128, 1, 1, 7, 7, 1024][50176, 50176, 7168, 7168, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_632, [0, 1, 3, 2, 4, 5]);  view_632 = None
        view_633: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_237, [-1, 7, 7, 1024]);  permute_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_634: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_633, [-1, 49, 1024]);  view_633 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        convert_element_type_772: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_351, torch.bfloat16);  primals_351 = None
        convert_element_type_773: "bf16[3072, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_350, torch.bfloat16);  primals_350 = None
        convert_element_type_774: "bf16[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_634, torch.bfloat16);  view_634 = None
        view_635: "bf16[6272, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_774, [6272, 1024]);  convert_element_type_774 = None
        permute_238: "bf16[1024, 3072][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_773, [1, 0]);  convert_element_type_773 = None
        addmm_92: "bf16[6272, 3072][3072, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_772, view_635, permute_238);  convert_element_type_772 = None
        view_636: "bf16[128, 49, 3072][150528, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_92, [128, 49, 3072]);  addmm_92 = None
        view_637: "bf16[128, 49, 3, 32, 32][150528, 3072, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_636, [128, 49, 3, 32, -1]);  view_636 = None
        permute_239: "bf16[3, 128, 32, 49, 32][1024, 150528, 32, 3072, 1]cuda:0" = torch.ops.aten.permute.default(view_637, [2, 0, 3, 1, 4]);  view_637 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_23 = torch.ops.aten.unbind.int(permute_239);  permute_239 = None
        getitem_171: "bf16[128, 32, 49, 32][150528, 32, 3072, 1]cuda:0" = unbind_23[0]
        getitem_172: "bf16[128, 32, 49, 32][150528, 32, 3072, 1]cuda:0" = unbind_23[1]
        getitem_173: "bf16[128, 32, 49, 32][150528, 32, 3072, 1]cuda:0" = unbind_23[2];  unbind_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_238: "bf16[128, 32, 49, 32][50176, 32, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_171, 0.1767766952966369);  getitem_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_240: "bf16[128, 32, 32, 49][150528, 32, 1, 3072]cuda:0" = torch.ops.aten.permute.default(getitem_172, [0, 1, 3, 2]);  getitem_172 = None
        expand_92: "bf16[128, 32, 49, 32][50176, 32, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_238, [128, 32, 49, 32]);  mul_238 = None
        clone_254: "bf16[128, 32, 49, 32][50176, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
        view_638: "bf16[4096, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_254, [4096, 49, 32]);  clone_254 = None
        expand_93: "bf16[128, 32, 32, 49][150528, 32, 1, 3072]cuda:0" = torch.ops.aten.expand.default(permute_240, [128, 32, 32, 49]);  permute_240 = None
        clone_255: "bf16[128, 32, 32, 49][50176, 1568, 49, 1]cuda:0" = torch.ops.aten.clone.default(expand_93, memory_format = torch.contiguous_format);  expand_93 = None
        view_639: "bf16[4096, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(clone_255, [4096, 32, 49]);  clone_255 = None
        bmm_46: "bf16[4096, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_638, view_639)
        view_640: "bf16[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_46, [128, 32, 49, 49])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_641: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_353, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_67: "f32[2401, 32][32, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_352, [view_641]);  view_641 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_642: "f32[49, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(index_67, [49, 49, -1]);  index_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_241: "f32[32, 49, 49][1, 1568, 32]cuda:0" = torch.ops.aten.permute.default(view_642, [2, 0, 1]);  view_642 = None
        clone_256: "f32[32, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_241, memory_format = torch.contiguous_format);  permute_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_45: "f32[1, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_256, 0);  clone_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_249: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_640, unsqueeze_45);  view_640 = unsqueeze_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_23: "f32[128, 32, 49, 1][1568, 49, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_249, [-1], True)
        sub_74: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_249, amax_23);  add_249 = None
        exp_23: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_74);  sub_74 = None
        sum_24: "f32[128, 32, 49, 1][1568, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_67: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        convert_element_type_780: "bf16[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_67, torch.bfloat16);  div_67 = None
        expand_94: "bf16[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_780, [128, 32, 49, 49]);  convert_element_type_780 = None
        view_643: "bf16[4096, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(expand_94, [4096, 49, 49]);  expand_94 = None
        expand_95: "bf16[128, 32, 49, 32][150528, 32, 3072, 1]cuda:0" = torch.ops.aten.expand.default(getitem_173, [128, 32, 49, 32]);  getitem_173 = None
        clone_258: "bf16[128, 32, 49, 32][50176, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(expand_95, memory_format = torch.contiguous_format);  expand_95 = None
        view_644: "bf16[4096, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_258, [4096, 49, 32]);  clone_258 = None
        constant_pad_nd_default: "bf16[4096, 56, 56][3136, 56, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_643, [0, 7, 0, 7, 0, 0])
        constant_pad_nd_default_1: "bf16[4096, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_644, [0, 0, 0, 7, 0, 0])
        bmm_default: "bf16[4096, 56, 32][1792, 32, 1]cuda:0" = torch.ops.aten.bmm.default(constant_pad_nd_default, constant_pad_nd_default_1);  constant_pad_nd_default = constant_pad_nd_default_1 = None
        slice_tensor: "bf16[4096, 49, 32][1792, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default, 1, 0, -7);  bmm_default = None
        view_645: "bf16[128, 32, 49, 32][57344, 1792, 32, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor, [128, 32, 49, 32]);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_242: "bf16[128, 49, 32, 32][57344, 32, 1792, 1]cuda:0" = torch.ops.aten.permute.default(view_645, [0, 2, 1, 3]);  view_645 = None
        clone_259: "bf16[128, 49, 32, 32][50176, 1024, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_242, memory_format = torch.contiguous_format);  permute_242 = None
        view_646: "bf16[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_259, [128, 49, 1024]);  clone_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        convert_element_type_783: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_355, torch.bfloat16);  primals_355 = None
        convert_element_type_784: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_354, torch.bfloat16);  primals_354 = None
        view_647: "bf16[6272, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_646, [6272, 1024]);  view_646 = None
        permute_243: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_784, [1, 0]);  convert_element_type_784 = None
        addmm_93: "bf16[6272, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_783, view_647, permute_243);  convert_element_type_783 = None
        view_648: "bf16[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_93, [128, 49, 1024]);  addmm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_649: "bf16[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_648, [-1, 7, 7, 1024]);  view_648 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_650: "bf16[128, 1, 1, 7, 7, 1024][50176, 50176, 50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_649, [-1, 1, 1, 7, 7, 1024]);  view_649 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_244: "bf16[128, 1, 7, 1, 7, 1024][50176, 50176, 7168, 50176, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_650, [0, 1, 3, 2, 4, 5]);  view_650 = None
        view_651: "bf16[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_244, [-1, 7, 7, 1024]);  permute_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_44: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 44)
        inductor_random_default_1: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_44, 'rand');  inductor_lookup_seed_default_44 = None
        lt_44: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default_1, 0.8999999985098839);  inductor_random_default_1 = None
        convert_element_type_788: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_44, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_68: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_788, 0.8999999985098839);  convert_element_type_788 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_239: "bf16[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_651, div_68);  view_651 = div_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_250: "bf16[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_631, mul_239);  mul_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_652: "bf16[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(add_250, [128, -1, 1024]);  add_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        convert_element_type_789: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_652, torch.float32)
        var_mean_51 = torch.ops.aten.var_mean.correction(convert_element_type_789, [2], correction = 0, keepdim = True)
        getitem_174: "f32[128, 49, 1][49, 1, 1]cuda:0" = var_mean_51[0]
        getitem_175: "f32[128, 49, 1][49, 1, 1]cuda:0" = var_mean_51[1];  var_mean_51 = None
        add_251: "f32[128, 49, 1][49, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_174, 1e-05);  getitem_174 = None
        rsqrt_51: "f32[128, 49, 1][49, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_251);  add_251 = None
        sub_75: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_789, getitem_175);  convert_element_type_789 = None
        mul_240: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_51);  sub_75 = None
        mul_241: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_240, primals_356);  mul_240 = None
        add_252: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_241, primals_357);  mul_241 = primals_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        convert_element_type_790: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_359, torch.bfloat16);  primals_359 = None
        convert_element_type_791: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_358, torch.bfloat16);  primals_358 = None
        convert_element_type_792: "bf16[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_252, torch.bfloat16);  add_252 = None
        view_653: "bf16[6272, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_792, [6272, 1024]);  convert_element_type_792 = None
        permute_245: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_791, [1, 0]);  convert_element_type_791 = None
        addmm_94: "bf16[6272, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_790, view_653, permute_245);  convert_element_type_790 = None
        view_654: "bf16[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_94, [128, 49, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_796: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_654, torch.float32);  view_654 = None
        mul_242: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_796, 0.5)
        mul_243: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_796, 0.7071067811865476);  convert_element_type_796 = None
        erf_23: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_243);  mul_243 = None
        add_253: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_244: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_242, add_253);  mul_242 = add_253 = None
        convert_element_type_797: "bf16[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_244, torch.bfloat16);  mul_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        convert_element_type_798: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_361, torch.bfloat16);  primals_361 = None
        convert_element_type_799: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_360, torch.bfloat16);  primals_360 = None
        view_655: "bf16[6272, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_797, [6272, 4096]);  convert_element_type_797 = None
        permute_246: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_799, [1, 0]);  convert_element_type_799 = None
        addmm_95: "bf16[6272, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_798, view_655, permute_246);  convert_element_type_798 = None
        view_656: "bf16[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_95, [128, 49, 1024]);  addmm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_45: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 45);  inductor_seeds_default = None
        inductor_random_default: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_45, 'rand');  inductor_lookup_seed_default_45 = None
        lt_45: "b8[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.lt.Scalar(inductor_random_default, 0.8999999985098839);  inductor_random_default = None
        convert_element_type_803: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_45, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_69: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_803, 0.8999999985098839);  convert_element_type_803 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_245: "bf16[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_656, div_69);  view_656 = div_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_254: "bf16[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_652, mul_245);  mul_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_657: "bf16[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(add_254, [128, 7, 7, 1024]);  add_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:981 in forward_features, code: x = self.norm(x)
        convert_element_type_804: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_657, torch.float32)
        var_mean_52 = torch.ops.aten.var_mean.correction(convert_element_type_804, [3], correction = 0, keepdim = True)
        getitem_176: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = var_mean_52[0]
        getitem_177: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = var_mean_52[1];  var_mean_52 = None
        add_255: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_176, 1e-05);  getitem_176 = None
        rsqrt_52: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_255);  add_255 = None
        sub_76: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_804, getitem_177);  convert_element_type_804 = None
        mul_246: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_52);  sub_76 = None
        mul_247: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_246, primals_362);  mul_246 = None
        add_256: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_247, primals_363);  mul_247 = primals_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:65 in forward, code: return x.mean(self.dim, keepdim=not self.flatten)
        mean: "f32[128, 1024][1024, 1]cuda:0" = torch.ops.aten.mean.dim(add_256, [1, 2]);  add_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:141 in forward, code: x = self.fc(x)
        convert_element_type_805: "bf16[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_365, torch.bfloat16);  primals_365 = None
        convert_element_type_806: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_364, torch.bfloat16);  primals_364 = None
        convert_element_type_807: "bf16[128, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mean, torch.bfloat16);  mean = None
        permute_247: "bf16[1024, 1000][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_806, [1, 0]);  convert_element_type_806 = None
        addmm_96: "bf16[128, 1000][1000, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_805, convert_element_type_807, permute_247);  convert_element_type_805 = None
        permute_248: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_247, [1, 0]);  permute_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_252: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_246, [1, 0]);  permute_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_256: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_245, [1, 0]);  permute_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_261: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_266: "bf16[4096, 49, 49][2401, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_643, [0, 2, 1]);  view_643 = None
        permute_267: "bf16[4096, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_644, [0, 2, 1]);  view_644 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_269: "bf16[4096, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_638, [0, 2, 1]);  view_638 = None
        permute_270: "bf16[4096, 49, 32][1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_639, [0, 2, 1]);  view_639 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_273: "bf16[3072, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_238, [1, 0]);  permute_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_278: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_236, [1, 0]);  permute_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_282: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_235, [1, 0]);  permute_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_287: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_292: "bf16[4096, 49, 49][2401, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_617, [0, 2, 1]);  view_617 = None
        permute_293: "bf16[4096, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_618, [0, 2, 1]);  view_618 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_295: "bf16[4096, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_612, [0, 2, 1]);  view_612 = None
        permute_296: "bf16[4096, 49, 32][1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_613, [0, 2, 1]);  view_613 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_299: "bf16[3072, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:541 in forward, code: x = self.reduction(x)
        permute_306: "bf16[1024, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_226, [1, 0]);  permute_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_309: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_224, [1, 0]);  permute_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_313: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_223, [1, 0]);  permute_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_318: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_221, [1, 0]);  permute_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_323: "bf16[8192, 49, 49][2401, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_587, [0, 2, 1]);  view_587 = None
        permute_324: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_588, [0, 2, 1]);  view_588 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_326: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_580, [0, 2, 1]);  view_580 = None
        permute_327: "bf16[8192, 49, 32][1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_581, [0, 2, 1]);  view_581 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_330: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_216, [1, 0]);  permute_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_335: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_214, [1, 0]);  permute_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_339: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_213, [1, 0]);  permute_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_344: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_349: "bf16[8192, 49, 49][2401, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_559, [0, 2, 1]);  view_559 = None
        permute_350: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_560, [0, 2, 1]);  view_560 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_352: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_554, [0, 2, 1]);  view_554 = None
        permute_353: "bf16[8192, 49, 32][1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_555, [0, 2, 1]);  view_555 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_356: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_361: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_204, [1, 0]);  permute_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_365: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_203, [1, 0]);  permute_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_370: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_375: "bf16[8192, 49, 49][2401, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_533, [0, 2, 1]);  view_533 = None
        permute_376: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_534, [0, 2, 1]);  view_534 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_378: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_526, [0, 2, 1]);  view_526 = None
        permute_379: "bf16[8192, 49, 32][1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_527, [0, 2, 1]);  view_527 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_382: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_387: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_194, [1, 0]);  permute_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_391: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_193, [1, 0]);  permute_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_396: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_191, [1, 0]);  permute_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_401: "bf16[8192, 49, 49][2401, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_505, [0, 2, 1]);  view_505 = None
        permute_402: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_506, [0, 2, 1]);  view_506 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_404: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_500, [0, 2, 1]);  view_500 = None
        permute_405: "bf16[8192, 49, 32][1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_501, [0, 2, 1]);  view_501 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_408: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_413: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_417: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_183, [1, 0]);  permute_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_422: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_427: "bf16[8192, 49, 49][2401, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_479, [0, 2, 1]);  view_479 = None
        permute_428: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_480, [0, 2, 1]);  view_480 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_430: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_472, [0, 2, 1]);  view_472 = None
        permute_431: "bf16[8192, 49, 32][1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_473, [0, 2, 1]);  view_473 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_434: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_439: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_443: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_448: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_171, [1, 0]);  permute_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_453: "bf16[8192, 49, 49][2401, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_451, [0, 2, 1]);  view_451 = None
        permute_454: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_452, [0, 2, 1]);  view_452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_456: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_446, [0, 2, 1]);  view_446 = None
        permute_457: "bf16[8192, 49, 32][1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_447, [0, 2, 1]);  view_447 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_460: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_465: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_469: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_474: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_479: "bf16[8192, 49, 49][2401, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_425, [0, 2, 1]);  view_425 = None
        permute_480: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_426, [0, 2, 1]);  view_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_482: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_418, [0, 2, 1]);  view_418 = None
        permute_483: "bf16[8192, 49, 32][1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_419, [0, 2, 1]);  view_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_486: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_491: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_495: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_500: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_505: "bf16[8192, 49, 49][2401, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_397, [0, 2, 1]);  view_397 = None
        permute_506: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_398, [0, 2, 1]);  view_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_508: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_392, [0, 2, 1]);  view_392 = None
        permute_509: "bf16[8192, 49, 32][1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_393, [0, 2, 1]);  view_393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_512: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_146, [1, 0]);  permute_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_517: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_521: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_526: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_531: "bf16[8192, 49, 49][2401, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_371, [0, 2, 1]);  view_371 = None
        permute_532: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_372, [0, 2, 1]);  view_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_534: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_364, [0, 2, 1]);  view_364 = None
        permute_535: "bf16[8192, 49, 32][1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_365, [0, 2, 1]);  view_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_538: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_543: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_547: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_552: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_557: "bf16[8192, 49, 49][2401, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_343, [0, 2, 1]);  view_343 = None
        permute_558: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_344, [0, 2, 1]);  view_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_560: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_338, [0, 2, 1]);  view_338 = None
        permute_561: "bf16[8192, 49, 32][1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_339, [0, 2, 1]);  view_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_564: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_126, [1, 0]);  permute_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_569: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_573: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_578: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_583: "bf16[8192, 49, 49][2401, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_317, [0, 2, 1]);  view_317 = None
        permute_584: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_318, [0, 2, 1]);  view_318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_586: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_310, [0, 2, 1]);  view_310 = None
        permute_587: "bf16[8192, 49, 32][1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_311, [0, 2, 1]);  view_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_590: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_595: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_599: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_604: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_609: "bf16[8192, 49, 49][2401, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_289, [0, 2, 1]);  view_289 = None
        permute_610: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_290, [0, 2, 1]);  view_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_612: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_284, [0, 2, 1]);  view_284 = None
        permute_613: "bf16[8192, 49, 32][1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_285, [0, 2, 1]);  view_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_616: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_621: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_104, [1, 0]);  permute_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_625: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_630: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_635: "bf16[8192, 49, 49][2401, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_263, [0, 2, 1]);  view_263 = None
        permute_636: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_264, [0, 2, 1]);  view_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_638: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_256, [0, 2, 1]);  view_256 = None
        permute_639: "bf16[8192, 49, 32][1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_257, [0, 2, 1]);  view_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_642: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_647: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_651: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_656: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_661: "bf16[8192, 49, 49][2401, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_235, [0, 2, 1]);  view_235 = None
        permute_662: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_236, [0, 2, 1]);  view_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_664: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_230, [0, 2, 1]);  view_230 = None
        permute_665: "bf16[8192, 49, 32][1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_231, [0, 2, 1]);  view_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_668: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_673: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_84, [1, 0]);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_677: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_682: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_687: "bf16[8192, 49, 49][2401, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_209, [0, 2, 1]);  view_209 = None
        permute_688: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_690: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_202, [0, 2, 1]);  view_202 = None
        permute_691: "bf16[8192, 49, 32][1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_203, [0, 2, 1]);  view_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_694: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_699: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_703: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_708: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_713: "bf16[8192, 49, 49][2401, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_181, [0, 2, 1]);  view_181 = None
        permute_714: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_182, [0, 2, 1]);  view_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_716: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_176, [0, 2, 1]);  view_176 = None
        permute_717: "bf16[8192, 49, 32][1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_177, [0, 2, 1]);  view_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_720: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_725: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_729: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_734: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_739: "bf16[8192, 49, 49][2401, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_155, [0, 2, 1]);  view_155 = None
        permute_740: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_156, [0, 2, 1]);  view_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_742: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_148, [0, 2, 1]);  view_148 = None
        permute_743: "bf16[8192, 49, 32][1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_149, [0, 2, 1]);  view_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_746: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_751: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_755: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_760: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_765: "bf16[8192, 49, 49][2401, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_127, [0, 2, 1]);  view_127 = None
        permute_766: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_128, [0, 2, 1]);  view_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_768: "bf16[8192, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
        permute_769: "bf16[8192, 49, 32][1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_772: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:541 in forward, code: x = self.reduction(x)
        permute_779: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_782: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_786: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_791: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_796: "bf16[16384, 49, 49][2401, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
        permute_797: "bf16[16384, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_799: "bf16[16384, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_90, [0, 2, 1]);  view_90 = None
        permute_800: "bf16[16384, 49, 32][1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_91, [0, 2, 1]);  view_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_803: "bf16[768, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_808: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_812: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_817: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_822: "bf16[16384, 49, 49][2401, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_69, [0, 2, 1]);  view_69 = None
        permute_823: "bf16[16384, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_70, [0, 2, 1]);  view_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_825: "bf16[16384, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_64, [0, 2, 1]);  view_64 = None
        permute_826: "bf16[16384, 49, 32][1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_65, [0, 2, 1]);  view_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_829: "bf16[768, 256][256, 1]cuda:0" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:541 in forward, code: x = self.reduction(x)
        permute_836: "bf16[256, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:540 in forward, code: x = self.norm(x)
        div_118: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_5, 512);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_839: "bf16[128, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_843: "bf16[512, 128][128, 1]cuda:0" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_119: "f32[128, 3136, 1][3136, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_4, 128);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_848: "bf16[128, 128][128, 1]cuda:0" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_853: "bf16[32768, 49, 49][2401, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_39, [0, 2, 1]);  view_39 = None
        permute_854: "bf16[32768, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_40, [0, 2, 1]);  view_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_856: "bf16[32768, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None
        permute_857: "bf16[32768, 49, 32][1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_860: "bf16[384, 128][128, 1]cuda:0" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_120: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_3, 128);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_865: "bf16[128, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_869: "bf16[512, 128][128, 1]cuda:0" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_121: "f32[128, 3136, 1][3136, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_2, 128);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_874: "bf16[128, 128][128, 1]cuda:0" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_879: "bf16[32768, 49, 49][2401, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
        permute_880: "bf16[32768, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_882: "bf16[32768, 32, 49][1568, 1, 32]cuda:0" = torch.ops.aten.permute.default(view_6, [0, 2, 1]);  view_6 = None
        permute_883: "bf16[32768, 49, 32][1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_886: "bf16[384, 128][128, 1]cuda:0" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        return (addmm_96, primals_4, primals_5, primals_6, primals_10, primals_11, primals_14, primals_20, primals_22, primals_25, primals_26, primals_29, primals_35, primals_38, primals_42, primals_43, primals_46, primals_52, primals_54, primals_57, primals_58, primals_61, primals_67, primals_70, primals_74, primals_75, primals_78, primals_84, primals_86, primals_89, primals_90, primals_93, primals_99, primals_103, primals_104, primals_107, primals_113, primals_115, primals_118, primals_119, primals_122, primals_128, primals_132, primals_133, primals_136, primals_142, primals_144, primals_147, primals_148, primals_151, primals_157, primals_161, primals_162, primals_165, primals_171, primals_173, primals_176, primals_177, primals_180, primals_186, primals_190, primals_191, primals_194, primals_200, primals_202, primals_205, primals_206, primals_209, primals_215, primals_219, primals_220, primals_223, primals_229, primals_231, primals_234, primals_235, primals_238, primals_244, primals_248, primals_249, primals_252, primals_258, primals_260, primals_263, primals_264, primals_267, primals_273, primals_277, primals_278, primals_281, primals_287, primals_289, primals_292, primals_293, primals_296, primals_302, primals_306, primals_307, primals_310, primals_316, primals_318, primals_321, primals_322, primals_325, primals_331, primals_334, primals_338, primals_339, primals_342, primals_348, primals_352, primals_353, primals_356, primals_362, convert_element_type_1, convert_element_type_2, convolution, getitem_1, rsqrt, getitem_3, rsqrt_1, view_3, bmm_default_45, amax, sum_1, view_15, mul_5, view_21, addmm_2, view_23, mul_10, view_29, bmm_default_43, amax_1, sum_2, view_43, fmod_2, lt, mul_14, view_49, addmm_6, view_51, lt_1, mul_20, view_56, mm, getitem_19, rsqrt_6, view_61, bmm_default_41, amax_2, sum_3, view_73, lt_2, view_78, getitem_24, rsqrt_7, view_79, addmm_10, view_81, lt_3, view_83, getitem_26, rsqrt_8, view_87, bmm_default_39, amax_3, sum_4, view_101, fmod_6, lt_4, view_106, getitem_31, rsqrt_9, view_107, addmm_14, view_109, lt_5, view_113, getitem_33, rsqrt_10, view_114, mm_1, getitem_35, rsqrt_11, view_119, bmm_default_37, amax_4, sum_5, view_131, lt_6, view_136, getitem_40, rsqrt_12, view_137, addmm_18, view_139, lt_7, view_141, getitem_42, rsqrt_13, view_145, bmm_default_35, amax_5, sum_6, view_159, fmod_10, lt_8, view_164, getitem_47, rsqrt_14, view_165, addmm_22, view_167, lt_9, view_169, getitem_49, rsqrt_15, view_173, bmm_default_33, amax_6, sum_7, view_185, lt_10, view_190, getitem_54, rsqrt_16, view_191, addmm_26, view_193, lt_11, view_195, getitem_56, rsqrt_17, view_199, bmm_default_31, amax_7, sum_8, view_213, lt_12, view_218, getitem_61, rsqrt_18, view_219, addmm_30, view_221, lt_13, view_223, getitem_63, rsqrt_19, view_227, bmm_default_29, amax_8, sum_9, view_239, lt_14, view_244, getitem_68, rsqrt_20, view_245, addmm_34, view_247, lt_15, view_249, getitem_70, rsqrt_21, view_253, bmm_default_27, amax_9, sum_10, view_267, lt_16, view_272, getitem_75, rsqrt_22, view_273, addmm_38, view_275, lt_17, view_277, getitem_77, rsqrt_23, view_281, bmm_default_25, amax_10, sum_11, view_293, lt_18, view_298, getitem_82, rsqrt_24, view_299, addmm_42, view_301, lt_19, view_303, getitem_84, rsqrt_25, view_307, bmm_default_23, amax_11, sum_12, view_321, lt_20, view_326, getitem_89, rsqrt_26, view_327, addmm_46, view_329, lt_21, view_331, getitem_91, rsqrt_27, view_335, bmm_default_21, amax_12, sum_13, view_347, lt_22, view_352, getitem_96, rsqrt_28, view_353, addmm_50, view_355, lt_23, view_357, getitem_98, rsqrt_29, view_361, bmm_default_19, amax_13, sum_14, view_375, lt_24, view_380, getitem_103, rsqrt_30, view_381, addmm_54, view_383, lt_25, view_385, getitem_105, rsqrt_31, view_389, bmm_default_17, amax_14, sum_15, view_401, lt_26, view_406, getitem_110, rsqrt_32, view_407, addmm_58, view_409, lt_27, view_411, getitem_112, rsqrt_33, view_415, bmm_default_15, amax_15, sum_16, view_429, lt_28, view_434, getitem_117, rsqrt_34, view_435, addmm_62, view_437, lt_29, view_439, getitem_119, rsqrt_35, view_443, bmm_default_13, amax_16, sum_17, view_455, lt_30, view_460, getitem_124, rsqrt_36, view_461, addmm_66, view_463, lt_31, view_465, getitem_126, rsqrt_37, view_469, bmm_default_11, amax_17, sum_18, view_483, lt_32, view_488, getitem_131, rsqrt_38, view_489, addmm_70, view_491, lt_33, view_493, getitem_133, rsqrt_39, view_497, bmm_default_9, amax_18, sum_19, view_509, lt_34, view_514, getitem_138, rsqrt_40, view_515, addmm_74, view_517, lt_35, view_519, getitem_140, rsqrt_41, view_523, bmm_default_7, amax_19, sum_20, view_537, lt_36, view_542, getitem_145, rsqrt_42, view_543, addmm_78, view_545, lt_37, view_547, getitem_147, rsqrt_43, view_551, bmm_default_5, amax_20, sum_21, view_563, lt_38, view_568, getitem_152, rsqrt_44, view_569, addmm_82, view_571, lt_39, view_573, getitem_154, rsqrt_45, view_577, bmm_default_3, amax_21, sum_22, view_591, lt_40, view_596, getitem_159, rsqrt_46, view_597, addmm_86, view_599, lt_41, view_603, getitem_161, rsqrt_47, view_604, mm_2, getitem_163, rsqrt_48, view_609, bmm_44, amax_22, sum_23, view_621, lt_42, view_626, getitem_168, rsqrt_49, view_627, addmm_90, view_629, lt_43, view_631, getitem_170, rsqrt_50, view_635, bmm_46, amax_23, sum_24, view_647, lt_44, view_652, getitem_175, rsqrt_51, view_653, addmm_94, view_655, lt_45, view_657, getitem_177, rsqrt_52, convert_element_type_807, permute_248, permute_252, permute_256, permute_261, permute_266, permute_267, permute_269, permute_270, permute_273, permute_278, permute_282, permute_287, permute_292, permute_293, permute_295, permute_296, permute_299, permute_306, permute_309, permute_313, permute_318, permute_323, permute_324, permute_326, permute_327, permute_330, permute_335, permute_339, permute_344, permute_349, permute_350, permute_352, permute_353, permute_356, permute_361, permute_365, permute_370, permute_375, permute_376, permute_378, permute_379, permute_382, permute_387, permute_391, permute_396, permute_401, permute_402, permute_404, permute_405, permute_408, permute_413, permute_417, permute_422, permute_427, permute_428, permute_430, permute_431, permute_434, permute_439, permute_443, permute_448, permute_453, permute_454, permute_456, permute_457, permute_460, permute_465, permute_469, permute_474, permute_479, permute_480, permute_482, permute_483, permute_486, permute_491, permute_495, permute_500, permute_505, permute_506, permute_508, permute_509, permute_512, permute_517, permute_521, permute_526, permute_531, permute_532, permute_534, permute_535, permute_538, permute_543, permute_547, permute_552, permute_557, permute_558, permute_560, permute_561, permute_564, permute_569, permute_573, permute_578, permute_583, permute_584, permute_586, permute_587, permute_590, permute_595, permute_599, permute_604, permute_609, permute_610, permute_612, permute_613, permute_616, permute_621, permute_625, permute_630, permute_635, permute_636, permute_638, permute_639, permute_642, permute_647, permute_651, permute_656, permute_661, permute_662, permute_664, permute_665, permute_668, permute_673, permute_677, permute_682, permute_687, permute_688, permute_690, permute_691, permute_694, permute_699, permute_703, permute_708, permute_713, permute_714, permute_716, permute_717, permute_720, permute_725, permute_729, permute_734, permute_739, permute_740, permute_742, permute_743, permute_746, permute_751, permute_755, permute_760, permute_765, permute_766, permute_768, permute_769, permute_772, permute_779, permute_782, permute_786, permute_791, permute_796, permute_797, permute_799, permute_800, permute_803, permute_808, permute_812, permute_817, permute_822, permute_823, permute_825, permute_826, permute_829, permute_836, div_118, permute_839, permute_843, div_119, permute_848, permute_853, permute_854, permute_856, permute_857, permute_860, div_120, permute_865, permute_869, div_121, permute_874, permute_879, permute_880, permute_882, permute_883, permute_886)
