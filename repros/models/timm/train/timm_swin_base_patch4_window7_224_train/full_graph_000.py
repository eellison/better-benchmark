class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[128, 3, 224, 224]", primals_2: "f32[128, 3, 4, 4]", primals_3: "f32[128]", primals_4: "f32[128]", primals_5: "f32[128]", primals_6: "f32[128]", primals_7: "f32[128]", primals_8: "f32[384, 128]", primals_9: "f32[384]", primals_10: "f32[169, 4]", primals_11: "i64[49, 49]", primals_12: "f32[128, 128]", primals_13: "f32[128]", primals_14: "f32[128]", primals_15: "f32[128]", primals_16: "f32[512, 128]", primals_17: "f32[512]", primals_18: "f32[128, 512]", primals_19: "f32[128]", primals_20: "f32[128]", primals_21: "f32[128]", primals_22: "f32[64, 49, 49]", primals_23: "f32[384, 128]", primals_24: "f32[384]", primals_25: "f32[169, 4]", primals_26: "i64[49, 49]", primals_27: "f32[128, 128]", primals_28: "f32[128]", primals_29: "f32[128]", primals_30: "f32[128]", primals_31: "f32[512, 128]", primals_32: "f32[512]", primals_33: "f32[128, 512]", primals_34: "f32[128]", primals_35: "f32[512]", primals_36: "f32[512]", primals_37: "f32[256, 512]", primals_38: "f32[256]", primals_39: "f32[256]", primals_40: "f32[768, 256]", primals_41: "f32[768]", primals_42: "f32[169, 8]", primals_43: "i64[49, 49]", primals_44: "f32[256, 256]", primals_45: "f32[256]", primals_46: "f32[256]", primals_47: "f32[256]", primals_48: "f32[1024, 256]", primals_49: "f32[1024]", primals_50: "f32[256, 1024]", primals_51: "f32[256]", primals_52: "f32[256]", primals_53: "f32[256]", primals_54: "f32[16, 49, 49]", primals_55: "f32[768, 256]", primals_56: "f32[768]", primals_57: "f32[169, 8]", primals_58: "i64[49, 49]", primals_59: "f32[256, 256]", primals_60: "f32[256]", primals_61: "f32[256]", primals_62: "f32[256]", primals_63: "f32[1024, 256]", primals_64: "f32[1024]", primals_65: "f32[256, 1024]", primals_66: "f32[256]", primals_67: "f32[1024]", primals_68: "f32[1024]", primals_69: "f32[512, 1024]", primals_70: "f32[512]", primals_71: "f32[512]", primals_72: "f32[1536, 512]", primals_73: "f32[1536]", primals_74: "f32[169, 16]", primals_75: "i64[49, 49]", primals_76: "f32[512, 512]", primals_77: "f32[512]", primals_78: "f32[512]", primals_79: "f32[512]", primals_80: "f32[2048, 512]", primals_81: "f32[2048]", primals_82: "f32[512, 2048]", primals_83: "f32[512]", primals_84: "f32[512]", primals_85: "f32[512]", primals_86: "f32[4, 49, 49]", primals_87: "f32[1536, 512]", primals_88: "f32[1536]", primals_89: "f32[169, 16]", primals_90: "i64[49, 49]", primals_91: "f32[512, 512]", primals_92: "f32[512]", primals_93: "f32[512]", primals_94: "f32[512]", primals_95: "f32[2048, 512]", primals_96: "f32[2048]", primals_97: "f32[512, 2048]", primals_98: "f32[512]", primals_99: "f32[512]", primals_100: "f32[512]", primals_101: "f32[1536, 512]", primals_102: "f32[1536]", primals_103: "f32[169, 16]", primals_104: "i64[49, 49]", primals_105: "f32[512, 512]", primals_106: "f32[512]", primals_107: "f32[512]", primals_108: "f32[512]", primals_109: "f32[2048, 512]", primals_110: "f32[2048]", primals_111: "f32[512, 2048]", primals_112: "f32[512]", primals_113: "f32[512]", primals_114: "f32[512]", primals_115: "f32[4, 49, 49]", primals_116: "f32[1536, 512]", primals_117: "f32[1536]", primals_118: "f32[169, 16]", primals_119: "i64[49, 49]", primals_120: "f32[512, 512]", primals_121: "f32[512]", primals_122: "f32[512]", primals_123: "f32[512]", primals_124: "f32[2048, 512]", primals_125: "f32[2048]", primals_126: "f32[512, 2048]", primals_127: "f32[512]", primals_128: "f32[512]", primals_129: "f32[512]", primals_130: "f32[1536, 512]", primals_131: "f32[1536]", primals_132: "f32[169, 16]", primals_133: "i64[49, 49]", primals_134: "f32[512, 512]", primals_135: "f32[512]", primals_136: "f32[512]", primals_137: "f32[512]", primals_138: "f32[2048, 512]", primals_139: "f32[2048]", primals_140: "f32[512, 2048]", primals_141: "f32[512]", primals_142: "f32[512]", primals_143: "f32[512]", primals_144: "f32[4, 49, 49]", primals_145: "f32[1536, 512]", primals_146: "f32[1536]", primals_147: "f32[169, 16]", primals_148: "i64[49, 49]", primals_149: "f32[512, 512]", primals_150: "f32[512]", primals_151: "f32[512]", primals_152: "f32[512]", primals_153: "f32[2048, 512]", primals_154: "f32[2048]", primals_155: "f32[512, 2048]", primals_156: "f32[512]", primals_157: "f32[512]", primals_158: "f32[512]", primals_159: "f32[1536, 512]", primals_160: "f32[1536]", primals_161: "f32[169, 16]", primals_162: "i64[49, 49]", primals_163: "f32[512, 512]", primals_164: "f32[512]", primals_165: "f32[512]", primals_166: "f32[512]", primals_167: "f32[2048, 512]", primals_168: "f32[2048]", primals_169: "f32[512, 2048]", primals_170: "f32[512]", primals_171: "f32[512]", primals_172: "f32[512]", primals_173: "f32[4, 49, 49]", primals_174: "f32[1536, 512]", primals_175: "f32[1536]", primals_176: "f32[169, 16]", primals_177: "i64[49, 49]", primals_178: "f32[512, 512]", primals_179: "f32[512]", primals_180: "f32[512]", primals_181: "f32[512]", primals_182: "f32[2048, 512]", primals_183: "f32[2048]", primals_184: "f32[512, 2048]", primals_185: "f32[512]", primals_186: "f32[512]", primals_187: "f32[512]", primals_188: "f32[1536, 512]", primals_189: "f32[1536]", primals_190: "f32[169, 16]", primals_191: "i64[49, 49]", primals_192: "f32[512, 512]", primals_193: "f32[512]", primals_194: "f32[512]", primals_195: "f32[512]", primals_196: "f32[2048, 512]", primals_197: "f32[2048]", primals_198: "f32[512, 2048]", primals_199: "f32[512]", primals_200: "f32[512]", primals_201: "f32[512]", primals_202: "f32[4, 49, 49]", primals_203: "f32[1536, 512]", primals_204: "f32[1536]", primals_205: "f32[169, 16]", primals_206: "i64[49, 49]", primals_207: "f32[512, 512]", primals_208: "f32[512]", primals_209: "f32[512]", primals_210: "f32[512]", primals_211: "f32[2048, 512]", primals_212: "f32[2048]", primals_213: "f32[512, 2048]", primals_214: "f32[512]", primals_215: "f32[512]", primals_216: "f32[512]", primals_217: "f32[1536, 512]", primals_218: "f32[1536]", primals_219: "f32[169, 16]", primals_220: "i64[49, 49]", primals_221: "f32[512, 512]", primals_222: "f32[512]", primals_223: "f32[512]", primals_224: "f32[512]", primals_225: "f32[2048, 512]", primals_226: "f32[2048]", primals_227: "f32[512, 2048]", primals_228: "f32[512]", primals_229: "f32[512]", primals_230: "f32[512]", primals_231: "f32[4, 49, 49]", primals_232: "f32[1536, 512]", primals_233: "f32[1536]", primals_234: "f32[169, 16]", primals_235: "i64[49, 49]", primals_236: "f32[512, 512]", primals_237: "f32[512]", primals_238: "f32[512]", primals_239: "f32[512]", primals_240: "f32[2048, 512]", primals_241: "f32[2048]", primals_242: "f32[512, 2048]", primals_243: "f32[512]", primals_244: "f32[512]", primals_245: "f32[512]", primals_246: "f32[1536, 512]", primals_247: "f32[1536]", primals_248: "f32[169, 16]", primals_249: "i64[49, 49]", primals_250: "f32[512, 512]", primals_251: "f32[512]", primals_252: "f32[512]", primals_253: "f32[512]", primals_254: "f32[2048, 512]", primals_255: "f32[2048]", primals_256: "f32[512, 2048]", primals_257: "f32[512]", primals_258: "f32[512]", primals_259: "f32[512]", primals_260: "f32[4, 49, 49]", primals_261: "f32[1536, 512]", primals_262: "f32[1536]", primals_263: "f32[169, 16]", primals_264: "i64[49, 49]", primals_265: "f32[512, 512]", primals_266: "f32[512]", primals_267: "f32[512]", primals_268: "f32[512]", primals_269: "f32[2048, 512]", primals_270: "f32[2048]", primals_271: "f32[512, 2048]", primals_272: "f32[512]", primals_273: "f32[512]", primals_274: "f32[512]", primals_275: "f32[1536, 512]", primals_276: "f32[1536]", primals_277: "f32[169, 16]", primals_278: "i64[49, 49]", primals_279: "f32[512, 512]", primals_280: "f32[512]", primals_281: "f32[512]", primals_282: "f32[512]", primals_283: "f32[2048, 512]", primals_284: "f32[2048]", primals_285: "f32[512, 2048]", primals_286: "f32[512]", primals_287: "f32[512]", primals_288: "f32[512]", primals_289: "f32[4, 49, 49]", primals_290: "f32[1536, 512]", primals_291: "f32[1536]", primals_292: "f32[169, 16]", primals_293: "i64[49, 49]", primals_294: "f32[512, 512]", primals_295: "f32[512]", primals_296: "f32[512]", primals_297: "f32[512]", primals_298: "f32[2048, 512]", primals_299: "f32[2048]", primals_300: "f32[512, 2048]", primals_301: "f32[512]", primals_302: "f32[512]", primals_303: "f32[512]", primals_304: "f32[1536, 512]", primals_305: "f32[1536]", primals_306: "f32[169, 16]", primals_307: "i64[49, 49]", primals_308: "f32[512, 512]", primals_309: "f32[512]", primals_310: "f32[512]", primals_311: "f32[512]", primals_312: "f32[2048, 512]", primals_313: "f32[2048]", primals_314: "f32[512, 2048]", primals_315: "f32[512]", primals_316: "f32[512]", primals_317: "f32[512]", primals_318: "f32[4, 49, 49]", primals_319: "f32[1536, 512]", primals_320: "f32[1536]", primals_321: "f32[169, 16]", primals_322: "i64[49, 49]", primals_323: "f32[512, 512]", primals_324: "f32[512]", primals_325: "f32[512]", primals_326: "f32[512]", primals_327: "f32[2048, 512]", primals_328: "f32[2048]", primals_329: "f32[512, 2048]", primals_330: "f32[512]", primals_331: "f32[2048]", primals_332: "f32[2048]", primals_333: "f32[1024, 2048]", primals_334: "f32[1024]", primals_335: "f32[1024]", primals_336: "f32[3072, 1024]", primals_337: "f32[3072]", primals_338: "f32[169, 32]", primals_339: "i64[49, 49]", primals_340: "f32[1024, 1024]", primals_341: "f32[1024]", primals_342: "f32[1024]", primals_343: "f32[1024]", primals_344: "f32[4096, 1024]", primals_345: "f32[4096]", primals_346: "f32[1024, 4096]", primals_347: "f32[1024]", primals_348: "f32[1024]", primals_349: "f32[1024]", primals_350: "f32[3072, 1024]", primals_351: "f32[3072]", primals_352: "f32[169, 32]", primals_353: "i64[49, 49]", primals_354: "f32[1024, 1024]", primals_355: "f32[1024]", primals_356: "f32[1024]", primals_357: "f32[1024]", primals_358: "f32[4096, 1024]", primals_359: "f32[4096]", primals_360: "f32[1024, 4096]", primals_361: "f32[1024]", primals_362: "f32[1024]", primals_363: "f32[1024]", primals_364: "f32[1000, 1024]", primals_365: "f32[1000]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.proj(x)
        convolution: "f32[128, 128, 56, 56]" = torch.ops.aten.convolution.default(primals_1, primals_2, primals_3, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  primals_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/format.py:68 in nchw_to, code: x = x.permute(0, 2, 3, 1)
        permute: "f32[128, 56, 56, 128]" = torch.ops.aten.permute.default(convolution, [0, 2, 3, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:141 in forward, code: x = self.norm(x)
        var_mean = torch.ops.aten.var_mean.correction(permute, [3], correction = 0, keepdim = True)
        getitem: "f32[128, 56, 56, 1]" = var_mean[0]
        getitem_1: "f32[128, 56, 56, 1]" = var_mean[1];  var_mean = None
        add: "f32[128, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[128, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(permute, getitem_1);  permute = None
        mul: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul, primals_4);  mul = None
        add_1: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(mul_1, primals_5);  mul_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_1 = torch.ops.aten.var_mean.correction(add_1, [3], correction = 0, keepdim = True)
        getitem_2: "f32[128, 56, 56, 1]" = var_mean_1[0]
        getitem_3: "f32[128, 56, 56, 1]" = var_mean_1[1];  var_mean_1 = None
        add_2: "f32[128, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[128, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub_1: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(add_1, getitem_3)
        mul_2: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_3: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_2, primals_6);  mul_2 = None
        add_3: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(mul_3, primals_7);  mul_3 = primals_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.reshape.default(add_3, [128, 8, 7, 8, 7, 128]);  add_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_1: "f32[128, 8, 8, 7, 7, 128]" = torch.ops.aten.permute.default(view, [0, 1, 3, 2, 4, 5]);  view = None
        clone: "f32[128, 8, 8, 7, 7, 128]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_1: "f32[8192, 7, 7, 128]" = torch.ops.aten.reshape.default(clone, [-1, 7, 7, 128]);  clone = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_2: "f32[8192, 49, 128]" = torch.ops.aten.reshape.default(view_1, [-1, 49, 128]);  view_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_3: "f32[401408, 128]" = torch.ops.aten.reshape.default(view_2, [401408, 128]);  view_2 = None
        permute_2: "f32[128, 384]" = torch.ops.aten.permute.default(primals_8, [1, 0])
        addmm: "f32[401408, 384]" = torch.ops.aten.addmm.default(primals_9, view_3, permute_2);  primals_9 = permute_2 = None
        view_4: "f32[8192, 49, 384]" = torch.ops.aten.reshape.default(addmm, [8192, 49, 384]);  addmm = None
        view_5: "f32[8192, 49, 3, 4, 32]" = torch.ops.aten.reshape.default(view_4, [8192, 49, 3, 4, -1]);  view_4 = None
        permute_3: "f32[3, 8192, 4, 49, 32]" = torch.ops.aten.permute.default(view_5, [2, 0, 3, 1, 4]);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind = torch.ops.aten.unbind.int(permute_3);  permute_3 = None
        getitem_4: "f32[8192, 4, 49, 32]" = unbind[0]
        getitem_5: "f32[8192, 4, 49, 32]" = unbind[1]
        getitem_6: "f32[8192, 4, 49, 32]" = unbind[2];  unbind = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_4: "f32[8192, 4, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_4, 0.1767766952966369);  getitem_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_4: "f32[8192, 4, 32, 49]" = torch.ops.aten.permute.default(getitem_5, [0, 1, 3, 2]);  getitem_5 = None
        expand: "f32[8192, 4, 49, 32]" = torch.ops.aten.expand.default(mul_4, [8192, 4, 49, 32]);  mul_4 = None
        clone_1: "f32[8192, 4, 49, 32]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_6: "f32[32768, 49, 32]" = torch.ops.aten.reshape.default(clone_1, [32768, 49, 32]);  clone_1 = None
        expand_1: "f32[8192, 4, 32, 49]" = torch.ops.aten.expand.default(permute_4, [8192, 4, 32, 49]);  permute_4 = None
        clone_2: "f32[8192, 4, 32, 49]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_7: "f32[32768, 32, 49]" = torch.ops.aten.reshape.default(clone_2, [32768, 32, 49]);  clone_2 = None
        bmm: "f32[32768, 49, 49]" = torch.ops.aten.bmm.default(view_6, view_7)
        view_8: "f32[8192, 4, 49, 49]" = torch.ops.aten.reshape.default(bmm, [8192, 4, 49, 49]);  bmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_9: "i64[2401]" = torch.ops.aten.reshape.default(primals_11, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index: "f32[2401, 4]" = torch.ops.aten.index.Tensor(primals_10, [view_9]);  primals_10 = view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_10: "f32[49, 49, 4]" = torch.ops.aten.reshape.default(index, [49, 49, -1]);  index = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_5: "f32[4, 49, 49]" = torch.ops.aten.permute.default(view_10, [2, 0, 1]);  view_10 = None
        clone_3: "f32[4, 49, 49]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze: "f32[1, 4, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_3, 0);  clone_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_4: "f32[8192, 4, 49, 49]" = torch.ops.aten.add.Tensor(view_8, unsqueeze);  view_8 = unsqueeze = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax: "f32[8192, 4, 49, 1]" = torch.ops.aten.amax.default(add_4, [-1], True)
        sub_2: "f32[8192, 4, 49, 49]" = torch.ops.aten.sub.Tensor(add_4, amax);  add_4 = amax = None
        exp: "f32[8192, 4, 49, 49]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_1: "f32[8192, 4, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[8192, 4, 49, 49]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_2: "f32[8192, 4, 49, 49]" = torch.ops.aten.expand.default(div, [8192, 4, 49, 49])
        view_11: "f32[32768, 49, 49]" = torch.ops.aten.reshape.default(expand_2, [32768, 49, 49]);  expand_2 = None
        expand_3: "f32[8192, 4, 49, 32]" = torch.ops.aten.expand.default(getitem_6, [8192, 4, 49, 32]);  getitem_6 = None
        clone_5: "f32[8192, 4, 49, 32]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_12: "f32[32768, 49, 32]" = torch.ops.aten.reshape.default(clone_5, [32768, 49, 32]);  clone_5 = None
        bmm_1: "f32[32768, 49, 32]" = torch.ops.aten.bmm.default(view_11, view_12);  view_11 = None
        view_13: "f32[8192, 4, 49, 32]" = torch.ops.aten.reshape.default(bmm_1, [8192, 4, 49, 32]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_6: "f32[8192, 49, 4, 32]" = torch.ops.aten.permute.default(view_13, [0, 2, 1, 3]);  view_13 = None
        clone_6: "f32[8192, 49, 4, 32]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
        view_14: "f32[8192, 49, 128]" = torch.ops.aten.reshape.default(clone_6, [8192, 49, 128]);  clone_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_15: "f32[401408, 128]" = torch.ops.aten.reshape.default(view_14, [401408, 128]);  view_14 = None
        permute_7: "f32[128, 128]" = torch.ops.aten.permute.default(primals_12, [1, 0])
        addmm_1: "f32[401408, 128]" = torch.ops.aten.addmm.default(primals_13, view_15, permute_7);  primals_13 = permute_7 = None
        view_16: "f32[8192, 49, 128]" = torch.ops.aten.reshape.default(addmm_1, [8192, 49, 128]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_17: "f32[8192, 7, 7, 128]" = torch.ops.aten.reshape.default(view_16, [-1, 7, 7, 128]);  view_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_18: "f32[128, 8, 8, 7, 7, 128]" = torch.ops.aten.reshape.default(view_17, [-1, 8, 8, 7, 7, 128]);  view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_8: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.permute.default(view_18, [0, 1, 3, 2, 4, 5]);  view_18 = None
        clone_8: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
        view_19: "f32[128, 56, 56, 128]" = torch.ops.aten.reshape.default(clone_8, [-1, 56, 56, 128]);  clone_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_5: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(add_1, view_19);  add_1 = view_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_20: "f32[128, 3136, 128]" = torch.ops.aten.reshape.default(add_5, [128, -1, 128]);  add_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_2 = torch.ops.aten.var_mean.correction(view_20, [2], correction = 0, keepdim = True)
        getitem_7: "f32[128, 3136, 1]" = var_mean_2[0]
        getitem_8: "f32[128, 3136, 1]" = var_mean_2[1];  var_mean_2 = None
        add_6: "f32[128, 3136, 1]" = torch.ops.aten.add.Tensor(getitem_7, 1e-05);  getitem_7 = None
        rsqrt_2: "f32[128, 3136, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_3: "f32[128, 3136, 128]" = torch.ops.aten.sub.Tensor(view_20, getitem_8);  getitem_8 = None
        mul_5: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_6: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(mul_5, primals_14)
        add_7: "f32[128, 3136, 128]" = torch.ops.aten.add.Tensor(mul_6, primals_15);  mul_6 = primals_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_21: "f32[401408, 128]" = torch.ops.aten.reshape.default(add_7, [401408, 128]);  add_7 = None
        permute_9: "f32[128, 512]" = torch.ops.aten.permute.default(primals_16, [1, 0])
        addmm_2: "f32[401408, 512]" = torch.ops.aten.addmm.default(primals_17, view_21, permute_9);  primals_17 = permute_9 = None
        view_22: "f32[128, 3136, 512]" = torch.ops.aten.reshape.default(addmm_2, [128, 3136, 512])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_7: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(view_22, 0.5)
        mul_8: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(view_22, 0.7071067811865476);  view_22 = None
        erf: "f32[128, 3136, 512]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
        add_8: "f32[128, 3136, 512]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_9: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(mul_7, add_8);  mul_7 = add_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_23: "f32[401408, 512]" = torch.ops.aten.reshape.default(mul_9, [401408, 512]);  mul_9 = None
        permute_10: "f32[512, 128]" = torch.ops.aten.permute.default(primals_18, [1, 0])
        addmm_3: "f32[401408, 128]" = torch.ops.aten.addmm.default(primals_19, view_23, permute_10);  primals_19 = permute_10 = None
        view_24: "f32[128, 3136, 128]" = torch.ops.aten.reshape.default(addmm_3, [128, 3136, 128]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_9: "f32[128, 3136, 128]" = torch.ops.aten.add.Tensor(view_20, view_24);  view_20 = view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_25: "f32[128, 56, 56, 128]" = torch.ops.aten.reshape.default(add_9, [128, 56, 56, 128]);  add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_3 = torch.ops.aten.var_mean.correction(view_25, [3], correction = 0, keepdim = True)
        getitem_9: "f32[128, 56, 56, 1]" = var_mean_3[0]
        getitem_10: "f32[128, 56, 56, 1]" = var_mean_3[1];  var_mean_3 = None
        add_10: "f32[128, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem_9, 1e-05);  getitem_9 = None
        rsqrt_3: "f32[128, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_4: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(view_25, getitem_10);  getitem_10 = None
        mul_10: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = None
        mul_11: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_10, primals_20)
        add_11: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(mul_11, primals_21);  mul_11 = primals_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota: "i64[56]" = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_12: "i64[56]" = torch.ops.aten.add.Tensor(iota, 3)
        fmod: "i64[56]" = torch.ops.aten.fmod.Scalar(add_12, 56);  add_12 = None
        index_1: "f32[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(add_11, [None, fmod]);  add_11 = None
        index_2: "f32[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(index_1, [None, None, fmod]);  index_1 = fmod = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_26: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.reshape.default(index_2, [128, 8, 7, 8, 7, 128]);  index_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_11: "f32[128, 8, 8, 7, 7, 128]" = torch.ops.aten.permute.default(view_26, [0, 1, 3, 2, 4, 5]);  view_26 = None
        clone_11: "f32[128, 8, 8, 7, 7, 128]" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
        view_27: "f32[8192, 7, 7, 128]" = torch.ops.aten.reshape.default(clone_11, [-1, 7, 7, 128]);  clone_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_28: "f32[8192, 49, 128]" = torch.ops.aten.reshape.default(view_27, [-1, 49, 128]);  view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_29: "f32[401408, 128]" = torch.ops.aten.reshape.default(view_28, [401408, 128]);  view_28 = None
        permute_12: "f32[128, 384]" = torch.ops.aten.permute.default(primals_23, [1, 0])
        addmm_4: "f32[401408, 384]" = torch.ops.aten.addmm.default(primals_24, view_29, permute_12);  primals_24 = permute_12 = None
        view_30: "f32[8192, 49, 384]" = torch.ops.aten.reshape.default(addmm_4, [8192, 49, 384]);  addmm_4 = None
        view_31: "f32[8192, 49, 3, 4, 32]" = torch.ops.aten.reshape.default(view_30, [8192, 49, 3, 4, -1]);  view_30 = None
        permute_13: "f32[3, 8192, 4, 49, 32]" = torch.ops.aten.permute.default(view_31, [2, 0, 3, 1, 4]);  view_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_1 = torch.ops.aten.unbind.int(permute_13);  permute_13 = None
        getitem_11: "f32[8192, 4, 49, 32]" = unbind_1[0]
        getitem_12: "f32[8192, 4, 49, 32]" = unbind_1[1]
        getitem_13: "f32[8192, 4, 49, 32]" = unbind_1[2];  unbind_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_12: "f32[8192, 4, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_11, 0.1767766952966369);  getitem_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_14: "f32[8192, 4, 32, 49]" = torch.ops.aten.permute.default(getitem_12, [0, 1, 3, 2]);  getitem_12 = None
        expand_4: "f32[8192, 4, 49, 32]" = torch.ops.aten.expand.default(mul_12, [8192, 4, 49, 32]);  mul_12 = None
        clone_12: "f32[8192, 4, 49, 32]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_32: "f32[32768, 49, 32]" = torch.ops.aten.reshape.default(clone_12, [32768, 49, 32]);  clone_12 = None
        expand_5: "f32[8192, 4, 32, 49]" = torch.ops.aten.expand.default(permute_14, [8192, 4, 32, 49]);  permute_14 = None
        clone_13: "f32[8192, 4, 32, 49]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_33: "f32[32768, 32, 49]" = torch.ops.aten.reshape.default(clone_13, [32768, 32, 49]);  clone_13 = None
        bmm_2: "f32[32768, 49, 49]" = torch.ops.aten.bmm.default(view_32, view_33)
        view_34: "f32[8192, 4, 49, 49]" = torch.ops.aten.reshape.default(bmm_2, [8192, 4, 49, 49]);  bmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_35: "i64[2401]" = torch.ops.aten.reshape.default(primals_26, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_3: "f32[2401, 4]" = torch.ops.aten.index.Tensor(primals_25, [view_35]);  primals_25 = view_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_36: "f32[49, 49, 4]" = torch.ops.aten.reshape.default(index_3, [49, 49, -1]);  index_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_15: "f32[4, 49, 49]" = torch.ops.aten.permute.default(view_36, [2, 0, 1]);  view_36 = None
        clone_14: "f32[4, 49, 49]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_1: "f32[1, 4, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_14, 0);  clone_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_14: "f32[8192, 4, 49, 49]" = torch.ops.aten.add.Tensor(view_34, unsqueeze_1);  view_34 = unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_37: "f32[128, 64, 4, 49, 49]" = torch.ops.aten.reshape.default(add_14, [-1, 64, 4, 49, 49]);  add_14 = None
        unsqueeze_2: "f32[64, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(primals_22, 1);  primals_22 = None
        unsqueeze_3: "f32[1, 64, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 0);  unsqueeze_2 = None
        add_15: "f32[128, 64, 4, 49, 49]" = torch.ops.aten.add.Tensor(view_37, unsqueeze_3);  view_37 = unsqueeze_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_38: "f32[8192, 4, 49, 49]" = torch.ops.aten.reshape.default(add_15, [-1, 4, 49, 49]);  add_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_1: "f32[8192, 4, 49, 1]" = torch.ops.aten.amax.default(view_38, [-1], True)
        sub_5: "f32[8192, 4, 49, 49]" = torch.ops.aten.sub.Tensor(view_38, amax_1);  view_38 = amax_1 = None
        exp_1: "f32[8192, 4, 49, 49]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        sum_2: "f32[8192, 4, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[8192, 4, 49, 49]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_6: "f32[8192, 4, 49, 49]" = torch.ops.aten.expand.default(div_1, [8192, 4, 49, 49])
        view_39: "f32[32768, 49, 49]" = torch.ops.aten.reshape.default(expand_6, [32768, 49, 49]);  expand_6 = None
        expand_7: "f32[8192, 4, 49, 32]" = torch.ops.aten.expand.default(getitem_13, [8192, 4, 49, 32]);  getitem_13 = None
        clone_16: "f32[8192, 4, 49, 32]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_40: "f32[32768, 49, 32]" = torch.ops.aten.reshape.default(clone_16, [32768, 49, 32]);  clone_16 = None
        bmm_3: "f32[32768, 49, 32]" = torch.ops.aten.bmm.default(view_39, view_40);  view_39 = None
        view_41: "f32[8192, 4, 49, 32]" = torch.ops.aten.reshape.default(bmm_3, [8192, 4, 49, 32]);  bmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_16: "f32[8192, 49, 4, 32]" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
        clone_17: "f32[8192, 49, 4, 32]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        view_42: "f32[8192, 49, 128]" = torch.ops.aten.reshape.default(clone_17, [8192, 49, 128]);  clone_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_43: "f32[401408, 128]" = torch.ops.aten.reshape.default(view_42, [401408, 128]);  view_42 = None
        permute_17: "f32[128, 128]" = torch.ops.aten.permute.default(primals_27, [1, 0])
        addmm_5: "f32[401408, 128]" = torch.ops.aten.addmm.default(primals_28, view_43, permute_17);  primals_28 = permute_17 = None
        view_44: "f32[8192, 49, 128]" = torch.ops.aten.reshape.default(addmm_5, [8192, 49, 128]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_45: "f32[8192, 7, 7, 128]" = torch.ops.aten.reshape.default(view_44, [-1, 7, 7, 128]);  view_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_46: "f32[128, 8, 8, 7, 7, 128]" = torch.ops.aten.reshape.default(view_45, [-1, 8, 8, 7, 7, 128]);  view_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_18: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.permute.default(view_46, [0, 1, 3, 2, 4, 5]);  view_46 = None
        clone_19: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_47: "f32[128, 56, 56, 128]" = torch.ops.aten.reshape.default(clone_19, [-1, 56, 56, 128]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        add_16: "i64[56]" = torch.ops.aten.add.Tensor(iota, 53);  iota = None
        fmod_2: "i64[56]" = torch.ops.aten.fmod.Scalar(add_16, 56);  add_16 = None
        index_4: "f32[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(view_47, [None, fmod_2]);  view_47 = None
        index_5: "f32[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(index_4, [None, None, fmod_2]);  index_4 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[46]" = torch.ops.prims.inductor_seeds.default(46, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_45: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        lt: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_45, 0.9956521736457944);  inductor_random_default_45 = None
        convert_element_type: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_2: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type, 0.9956521736457944);  convert_element_type = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_13: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(index_5, div_2);  index_5 = div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_18: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(view_25, mul_13);  view_25 = mul_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_48: "f32[128, 3136, 128]" = torch.ops.aten.reshape.default(add_18, [128, -1, 128]);  add_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_4 = torch.ops.aten.var_mean.correction(view_48, [2], correction = 0, keepdim = True)
        getitem_14: "f32[128, 3136, 1]" = var_mean_4[0]
        getitem_15: "f32[128, 3136, 1]" = var_mean_4[1];  var_mean_4 = None
        add_19: "f32[128, 3136, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_4: "f32[128, 3136, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        sub_6: "f32[128, 3136, 128]" = torch.ops.aten.sub.Tensor(view_48, getitem_15);  getitem_15 = None
        mul_14: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
        mul_15: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(mul_14, primals_29)
        add_20: "f32[128, 3136, 128]" = torch.ops.aten.add.Tensor(mul_15, primals_30);  mul_15 = primals_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_49: "f32[401408, 128]" = torch.ops.aten.reshape.default(add_20, [401408, 128]);  add_20 = None
        permute_19: "f32[128, 512]" = torch.ops.aten.permute.default(primals_31, [1, 0])
        addmm_6: "f32[401408, 512]" = torch.ops.aten.addmm.default(primals_32, view_49, permute_19);  primals_32 = permute_19 = None
        view_50: "f32[128, 3136, 512]" = torch.ops.aten.reshape.default(addmm_6, [128, 3136, 512])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_16: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(view_50, 0.5)
        mul_17: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(view_50, 0.7071067811865476);  view_50 = None
        erf_1: "f32[128, 3136, 512]" = torch.ops.aten.erf.default(mul_17);  mul_17 = None
        add_21: "f32[128, 3136, 512]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_18: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(mul_16, add_21);  mul_16 = add_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_51: "f32[401408, 512]" = torch.ops.aten.reshape.default(mul_18, [401408, 512]);  mul_18 = None
        permute_20: "f32[512, 128]" = torch.ops.aten.permute.default(primals_33, [1, 0])
        addmm_7: "f32[401408, 128]" = torch.ops.aten.addmm.default(primals_34, view_51, permute_20);  primals_34 = permute_20 = None
        view_52: "f32[128, 3136, 128]" = torch.ops.aten.reshape.default(addmm_7, [128, 3136, 128]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_44: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        lt_1: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_44, 0.9956521736457944);  inductor_random_default_44 = None
        convert_element_type_1: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_1, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_3: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_1, 0.9956521736457944);  convert_element_type_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_19: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(view_52, div_3);  view_52 = div_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_22: "f32[128, 3136, 128]" = torch.ops.aten.add.Tensor(view_48, mul_19);  view_48 = mul_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_53: "f32[128, 56, 56, 128]" = torch.ops.aten.reshape.default(add_22, [128, 56, 56, 128]);  add_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:539 in forward, code: x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        view_54: "f32[128, 28, 2, 28, 2, 128]" = torch.ops.aten.reshape.default(view_53, [128, 28, 2, 28, 2, 128]);  view_53 = None
        permute_21: "f32[128, 28, 28, 2, 2, 128]" = torch.ops.aten.permute.default(view_54, [0, 1, 3, 4, 2, 5]);  view_54 = None
        clone_22: "f32[128, 28, 28, 2, 2, 128]" = torch.ops.aten.clone.default(permute_21, memory_format = torch.contiguous_format);  permute_21 = None
        view_55: "f32[128, 28, 28, 512]" = torch.ops.aten.reshape.default(clone_22, [128, 28, 28, 512]);  clone_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:540 in forward, code: x = self.norm(x)
        var_mean_5 = torch.ops.aten.var_mean.correction(view_55, [3], correction = 0, keepdim = True)
        getitem_16: "f32[128, 28, 28, 1]" = var_mean_5[0]
        getitem_17: "f32[128, 28, 28, 1]" = var_mean_5[1];  var_mean_5 = None
        add_23: "f32[128, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_5: "f32[128, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        sub_7: "f32[128, 28, 28, 512]" = torch.ops.aten.sub.Tensor(view_55, getitem_17);  view_55 = getitem_17 = None
        mul_20: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_5);  sub_7 = None
        mul_21: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(mul_20, primals_35)
        add_24: "f32[128, 28, 28, 512]" = torch.ops.aten.add.Tensor(mul_21, primals_36);  mul_21 = primals_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:541 in forward, code: x = self.reduction(x)
        permute_22: "f32[512, 256]" = torch.ops.aten.permute.default(primals_37, [1, 0])
        view_56: "f32[100352, 512]" = torch.ops.aten.reshape.default(add_24, [100352, 512]);  add_24 = None
        mm: "f32[100352, 256]" = torch.ops.aten.mm.default(view_56, permute_22);  permute_22 = None
        view_57: "f32[128, 28, 28, 256]" = torch.ops.aten.reshape.default(mm, [128, 28, 28, 256])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_6 = torch.ops.aten.var_mean.correction(view_57, [3], correction = 0, keepdim = True)
        getitem_18: "f32[128, 28, 28, 1]" = var_mean_6[0]
        getitem_19: "f32[128, 28, 28, 1]" = var_mean_6[1];  var_mean_6 = None
        add_25: "f32[128, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_6: "f32[128, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        sub_8: "f32[128, 28, 28, 256]" = torch.ops.aten.sub.Tensor(view_57, getitem_19)
        mul_22: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_6);  sub_8 = None
        mul_23: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_22, primals_38);  mul_22 = None
        add_26: "f32[128, 28, 28, 256]" = torch.ops.aten.add.Tensor(mul_23, primals_39);  mul_23 = primals_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_58: "f32[128, 4, 7, 4, 7, 256]" = torch.ops.aten.reshape.default(add_26, [128, 4, 7, 4, 7, 256]);  add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_23: "f32[128, 4, 4, 7, 7, 256]" = torch.ops.aten.permute.default(view_58, [0, 1, 3, 2, 4, 5]);  view_58 = None
        clone_23: "f32[128, 4, 4, 7, 7, 256]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
        view_59: "f32[2048, 7, 7, 256]" = torch.ops.aten.reshape.default(clone_23, [-1, 7, 7, 256]);  clone_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_60: "f32[2048, 49, 256]" = torch.ops.aten.reshape.default(view_59, [-1, 49, 256]);  view_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_61: "f32[100352, 256]" = torch.ops.aten.reshape.default(view_60, [100352, 256]);  view_60 = None
        permute_24: "f32[256, 768]" = torch.ops.aten.permute.default(primals_40, [1, 0])
        addmm_8: "f32[100352, 768]" = torch.ops.aten.addmm.default(primals_41, view_61, permute_24);  primals_41 = permute_24 = None
        view_62: "f32[2048, 49, 768]" = torch.ops.aten.reshape.default(addmm_8, [2048, 49, 768]);  addmm_8 = None
        view_63: "f32[2048, 49, 3, 8, 32]" = torch.ops.aten.reshape.default(view_62, [2048, 49, 3, 8, -1]);  view_62 = None
        permute_25: "f32[3, 2048, 8, 49, 32]" = torch.ops.aten.permute.default(view_63, [2, 0, 3, 1, 4]);  view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_2 = torch.ops.aten.unbind.int(permute_25);  permute_25 = None
        getitem_20: "f32[2048, 8, 49, 32]" = unbind_2[0]
        getitem_21: "f32[2048, 8, 49, 32]" = unbind_2[1]
        getitem_22: "f32[2048, 8, 49, 32]" = unbind_2[2];  unbind_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_24: "f32[2048, 8, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_20, 0.1767766952966369);  getitem_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_26: "f32[2048, 8, 32, 49]" = torch.ops.aten.permute.default(getitem_21, [0, 1, 3, 2]);  getitem_21 = None
        expand_8: "f32[2048, 8, 49, 32]" = torch.ops.aten.expand.default(mul_24, [2048, 8, 49, 32]);  mul_24 = None
        clone_24: "f32[2048, 8, 49, 32]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_64: "f32[16384, 49, 32]" = torch.ops.aten.reshape.default(clone_24, [16384, 49, 32]);  clone_24 = None
        expand_9: "f32[2048, 8, 32, 49]" = torch.ops.aten.expand.default(permute_26, [2048, 8, 32, 49]);  permute_26 = None
        clone_25: "f32[2048, 8, 32, 49]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_65: "f32[16384, 32, 49]" = torch.ops.aten.reshape.default(clone_25, [16384, 32, 49]);  clone_25 = None
        bmm_4: "f32[16384, 49, 49]" = torch.ops.aten.bmm.default(view_64, view_65)
        view_66: "f32[2048, 8, 49, 49]" = torch.ops.aten.reshape.default(bmm_4, [2048, 8, 49, 49]);  bmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_67: "i64[2401]" = torch.ops.aten.reshape.default(primals_43, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_6: "f32[2401, 8]" = torch.ops.aten.index.Tensor(primals_42, [view_67]);  primals_42 = view_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_68: "f32[49, 49, 8]" = torch.ops.aten.reshape.default(index_6, [49, 49, -1]);  index_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_27: "f32[8, 49, 49]" = torch.ops.aten.permute.default(view_68, [2, 0, 1]);  view_68 = None
        clone_26: "f32[8, 49, 49]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_4: "f32[1, 8, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_26, 0);  clone_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_27: "f32[2048, 8, 49, 49]" = torch.ops.aten.add.Tensor(view_66, unsqueeze_4);  view_66 = unsqueeze_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_2: "f32[2048, 8, 49, 1]" = torch.ops.aten.amax.default(add_27, [-1], True)
        sub_9: "f32[2048, 8, 49, 49]" = torch.ops.aten.sub.Tensor(add_27, amax_2);  add_27 = amax_2 = None
        exp_2: "f32[2048, 8, 49, 49]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        sum_3: "f32[2048, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_4: "f32[2048, 8, 49, 49]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_10: "f32[2048, 8, 49, 49]" = torch.ops.aten.expand.default(div_4, [2048, 8, 49, 49])
        view_69: "f32[16384, 49, 49]" = torch.ops.aten.reshape.default(expand_10, [16384, 49, 49]);  expand_10 = None
        expand_11: "f32[2048, 8, 49, 32]" = torch.ops.aten.expand.default(getitem_22, [2048, 8, 49, 32]);  getitem_22 = None
        clone_28: "f32[2048, 8, 49, 32]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_70: "f32[16384, 49, 32]" = torch.ops.aten.reshape.default(clone_28, [16384, 49, 32]);  clone_28 = None
        bmm_5: "f32[16384, 49, 32]" = torch.ops.aten.bmm.default(view_69, view_70);  view_69 = None
        view_71: "f32[2048, 8, 49, 32]" = torch.ops.aten.reshape.default(bmm_5, [2048, 8, 49, 32]);  bmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_28: "f32[2048, 49, 8, 32]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
        clone_29: "f32[2048, 49, 8, 32]" = torch.ops.aten.clone.default(permute_28, memory_format = torch.contiguous_format);  permute_28 = None
        view_72: "f32[2048, 49, 256]" = torch.ops.aten.reshape.default(clone_29, [2048, 49, 256]);  clone_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_73: "f32[100352, 256]" = torch.ops.aten.reshape.default(view_72, [100352, 256]);  view_72 = None
        permute_29: "f32[256, 256]" = torch.ops.aten.permute.default(primals_44, [1, 0])
        addmm_9: "f32[100352, 256]" = torch.ops.aten.addmm.default(primals_45, view_73, permute_29);  primals_45 = permute_29 = None
        view_74: "f32[2048, 49, 256]" = torch.ops.aten.reshape.default(addmm_9, [2048, 49, 256]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_75: "f32[2048, 7, 7, 256]" = torch.ops.aten.reshape.default(view_74, [-1, 7, 7, 256]);  view_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_76: "f32[128, 4, 4, 7, 7, 256]" = torch.ops.aten.reshape.default(view_75, [-1, 4, 4, 7, 7, 256]);  view_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_30: "f32[128, 4, 7, 4, 7, 256]" = torch.ops.aten.permute.default(view_76, [0, 1, 3, 2, 4, 5]);  view_76 = None
        clone_31: "f32[128, 4, 7, 4, 7, 256]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
        view_77: "f32[128, 28, 28, 256]" = torch.ops.aten.reshape.default(clone_31, [-1, 28, 28, 256]);  clone_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_43: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        lt_2: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_43, 0.9913043472915888);  inductor_random_default_43 = None
        convert_element_type_2: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_2, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_5: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_2, 0.9913043472915888);  convert_element_type_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_25: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(view_77, div_5);  view_77 = div_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_28: "f32[128, 28, 28, 256]" = torch.ops.aten.add.Tensor(view_57, mul_25);  view_57 = mul_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_78: "f32[128, 784, 256]" = torch.ops.aten.reshape.default(add_28, [128, -1, 256]);  add_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_7 = torch.ops.aten.var_mean.correction(view_78, [2], correction = 0, keepdim = True)
        getitem_23: "f32[128, 784, 1]" = var_mean_7[0]
        getitem_24: "f32[128, 784, 1]" = var_mean_7[1];  var_mean_7 = None
        add_29: "f32[128, 784, 1]" = torch.ops.aten.add.Tensor(getitem_23, 1e-05);  getitem_23 = None
        rsqrt_7: "f32[128, 784, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        sub_10: "f32[128, 784, 256]" = torch.ops.aten.sub.Tensor(view_78, getitem_24);  getitem_24 = None
        mul_26: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_7);  sub_10 = None
        mul_27: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(mul_26, primals_46)
        add_30: "f32[128, 784, 256]" = torch.ops.aten.add.Tensor(mul_27, primals_47);  mul_27 = primals_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_79: "f32[100352, 256]" = torch.ops.aten.reshape.default(add_30, [100352, 256]);  add_30 = None
        permute_31: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_48, [1, 0])
        addmm_10: "f32[100352, 1024]" = torch.ops.aten.addmm.default(primals_49, view_79, permute_31);  primals_49 = permute_31 = None
        view_80: "f32[128, 784, 1024]" = torch.ops.aten.reshape.default(addmm_10, [128, 784, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_28: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(view_80, 0.5)
        mul_29: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(view_80, 0.7071067811865476);  view_80 = None
        erf_2: "f32[128, 784, 1024]" = torch.ops.aten.erf.default(mul_29);  mul_29 = None
        add_31: "f32[128, 784, 1024]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_30: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(mul_28, add_31);  mul_28 = add_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_81: "f32[100352, 1024]" = torch.ops.aten.reshape.default(mul_30, [100352, 1024]);  mul_30 = None
        permute_32: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_50, [1, 0])
        addmm_11: "f32[100352, 256]" = torch.ops.aten.addmm.default(primals_51, view_81, permute_32);  primals_51 = permute_32 = None
        view_82: "f32[128, 784, 256]" = torch.ops.aten.reshape.default(addmm_11, [128, 784, 256]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_3: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_42: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        lt_3: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_42, 0.9913043472915888);  inductor_random_default_42 = None
        convert_element_type_3: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_3, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_6: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_3, 0.9913043472915888);  convert_element_type_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_31: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(view_82, div_6);  view_82 = div_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_32: "f32[128, 784, 256]" = torch.ops.aten.add.Tensor(view_78, mul_31);  view_78 = mul_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_83: "f32[128, 28, 28, 256]" = torch.ops.aten.reshape.default(add_32, [128, 28, 28, 256]);  add_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_8 = torch.ops.aten.var_mean.correction(view_83, [3], correction = 0, keepdim = True)
        getitem_25: "f32[128, 28, 28, 1]" = var_mean_8[0]
        getitem_26: "f32[128, 28, 28, 1]" = var_mean_8[1];  var_mean_8 = None
        add_33: "f32[128, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_25, 1e-05);  getitem_25 = None
        rsqrt_8: "f32[128, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        sub_11: "f32[128, 28, 28, 256]" = torch.ops.aten.sub.Tensor(view_83, getitem_26);  getitem_26 = None
        mul_32: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_8);  sub_11 = None
        mul_33: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_32, primals_52)
        add_34: "f32[128, 28, 28, 256]" = torch.ops.aten.add.Tensor(mul_33, primals_53);  mul_33 = primals_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota_4: "i64[28]" = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_35: "i64[28]" = torch.ops.aten.add.Tensor(iota_4, 3)
        fmod_4: "i64[28]" = torch.ops.aten.fmod.Scalar(add_35, 28);  add_35 = None
        index_7: "f32[128, 28, 28, 256]" = torch.ops.aten.index.Tensor(add_34, [None, fmod_4]);  add_34 = None
        index_8: "f32[128, 28, 28, 256]" = torch.ops.aten.index.Tensor(index_7, [None, None, fmod_4]);  index_7 = fmod_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_84: "f32[128, 4, 7, 4, 7, 256]" = torch.ops.aten.reshape.default(index_8, [128, 4, 7, 4, 7, 256]);  index_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_33: "f32[128, 4, 4, 7, 7, 256]" = torch.ops.aten.permute.default(view_84, [0, 1, 3, 2, 4, 5]);  view_84 = None
        clone_34: "f32[128, 4, 4, 7, 7, 256]" = torch.ops.aten.clone.default(permute_33, memory_format = torch.contiguous_format);  permute_33 = None
        view_85: "f32[2048, 7, 7, 256]" = torch.ops.aten.reshape.default(clone_34, [-1, 7, 7, 256]);  clone_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_86: "f32[2048, 49, 256]" = torch.ops.aten.reshape.default(view_85, [-1, 49, 256]);  view_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_87: "f32[100352, 256]" = torch.ops.aten.reshape.default(view_86, [100352, 256]);  view_86 = None
        permute_34: "f32[256, 768]" = torch.ops.aten.permute.default(primals_55, [1, 0])
        addmm_12: "f32[100352, 768]" = torch.ops.aten.addmm.default(primals_56, view_87, permute_34);  primals_56 = permute_34 = None
        view_88: "f32[2048, 49, 768]" = torch.ops.aten.reshape.default(addmm_12, [2048, 49, 768]);  addmm_12 = None
        view_89: "f32[2048, 49, 3, 8, 32]" = torch.ops.aten.reshape.default(view_88, [2048, 49, 3, 8, -1]);  view_88 = None
        permute_35: "f32[3, 2048, 8, 49, 32]" = torch.ops.aten.permute.default(view_89, [2, 0, 3, 1, 4]);  view_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_3 = torch.ops.aten.unbind.int(permute_35);  permute_35 = None
        getitem_27: "f32[2048, 8, 49, 32]" = unbind_3[0]
        getitem_28: "f32[2048, 8, 49, 32]" = unbind_3[1]
        getitem_29: "f32[2048, 8, 49, 32]" = unbind_3[2];  unbind_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_34: "f32[2048, 8, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_27, 0.1767766952966369);  getitem_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_36: "f32[2048, 8, 32, 49]" = torch.ops.aten.permute.default(getitem_28, [0, 1, 3, 2]);  getitem_28 = None
        expand_12: "f32[2048, 8, 49, 32]" = torch.ops.aten.expand.default(mul_34, [2048, 8, 49, 32]);  mul_34 = None
        clone_35: "f32[2048, 8, 49, 32]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_90: "f32[16384, 49, 32]" = torch.ops.aten.reshape.default(clone_35, [16384, 49, 32]);  clone_35 = None
        expand_13: "f32[2048, 8, 32, 49]" = torch.ops.aten.expand.default(permute_36, [2048, 8, 32, 49]);  permute_36 = None
        clone_36: "f32[2048, 8, 32, 49]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_91: "f32[16384, 32, 49]" = torch.ops.aten.reshape.default(clone_36, [16384, 32, 49]);  clone_36 = None
        bmm_6: "f32[16384, 49, 49]" = torch.ops.aten.bmm.default(view_90, view_91)
        view_92: "f32[2048, 8, 49, 49]" = torch.ops.aten.reshape.default(bmm_6, [2048, 8, 49, 49]);  bmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_93: "i64[2401]" = torch.ops.aten.reshape.default(primals_58, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_9: "f32[2401, 8]" = torch.ops.aten.index.Tensor(primals_57, [view_93]);  primals_57 = view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_94: "f32[49, 49, 8]" = torch.ops.aten.reshape.default(index_9, [49, 49, -1]);  index_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_37: "f32[8, 49, 49]" = torch.ops.aten.permute.default(view_94, [2, 0, 1]);  view_94 = None
        clone_37: "f32[8, 49, 49]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_5: "f32[1, 8, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_37, 0);  clone_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_37: "f32[2048, 8, 49, 49]" = torch.ops.aten.add.Tensor(view_92, unsqueeze_5);  view_92 = unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_95: "f32[128, 16, 8, 49, 49]" = torch.ops.aten.reshape.default(add_37, [-1, 16, 8, 49, 49]);  add_37 = None
        unsqueeze_6: "f32[16, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(primals_54, 1);  primals_54 = None
        unsqueeze_7: "f32[1, 16, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 0);  unsqueeze_6 = None
        add_38: "f32[128, 16, 8, 49, 49]" = torch.ops.aten.add.Tensor(view_95, unsqueeze_7);  view_95 = unsqueeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_96: "f32[2048, 8, 49, 49]" = torch.ops.aten.reshape.default(add_38, [-1, 8, 49, 49]);  add_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_3: "f32[2048, 8, 49, 1]" = torch.ops.aten.amax.default(view_96, [-1], True)
        sub_12: "f32[2048, 8, 49, 49]" = torch.ops.aten.sub.Tensor(view_96, amax_3);  view_96 = amax_3 = None
        exp_3: "f32[2048, 8, 49, 49]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_4: "f32[2048, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_7: "f32[2048, 8, 49, 49]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_14: "f32[2048, 8, 49, 49]" = torch.ops.aten.expand.default(div_7, [2048, 8, 49, 49])
        view_97: "f32[16384, 49, 49]" = torch.ops.aten.reshape.default(expand_14, [16384, 49, 49]);  expand_14 = None
        expand_15: "f32[2048, 8, 49, 32]" = torch.ops.aten.expand.default(getitem_29, [2048, 8, 49, 32]);  getitem_29 = None
        clone_39: "f32[2048, 8, 49, 32]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_98: "f32[16384, 49, 32]" = torch.ops.aten.reshape.default(clone_39, [16384, 49, 32]);  clone_39 = None
        bmm_7: "f32[16384, 49, 32]" = torch.ops.aten.bmm.default(view_97, view_98);  view_97 = None
        view_99: "f32[2048, 8, 49, 32]" = torch.ops.aten.reshape.default(bmm_7, [2048, 8, 49, 32]);  bmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_38: "f32[2048, 49, 8, 32]" = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None
        clone_40: "f32[2048, 49, 8, 32]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
        view_100: "f32[2048, 49, 256]" = torch.ops.aten.reshape.default(clone_40, [2048, 49, 256]);  clone_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_101: "f32[100352, 256]" = torch.ops.aten.reshape.default(view_100, [100352, 256]);  view_100 = None
        permute_39: "f32[256, 256]" = torch.ops.aten.permute.default(primals_59, [1, 0])
        addmm_13: "f32[100352, 256]" = torch.ops.aten.addmm.default(primals_60, view_101, permute_39);  primals_60 = permute_39 = None
        view_102: "f32[2048, 49, 256]" = torch.ops.aten.reshape.default(addmm_13, [2048, 49, 256]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_103: "f32[2048, 7, 7, 256]" = torch.ops.aten.reshape.default(view_102, [-1, 7, 7, 256]);  view_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_104: "f32[128, 4, 4, 7, 7, 256]" = torch.ops.aten.reshape.default(view_103, [-1, 4, 4, 7, 7, 256]);  view_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_40: "f32[128, 4, 7, 4, 7, 256]" = torch.ops.aten.permute.default(view_104, [0, 1, 3, 2, 4, 5]);  view_104 = None
        clone_42: "f32[128, 4, 7, 4, 7, 256]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_105: "f32[128, 28, 28, 256]" = torch.ops.aten.reshape.default(clone_42, [-1, 28, 28, 256]);  clone_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        add_39: "i64[28]" = torch.ops.aten.add.Tensor(iota_4, 25);  iota_4 = None
        fmod_6: "i64[28]" = torch.ops.aten.fmod.Scalar(add_39, 28);  add_39 = None
        index_10: "f32[128, 28, 28, 256]" = torch.ops.aten.index.Tensor(view_105, [None, fmod_6]);  view_105 = None
        index_11: "f32[128, 28, 28, 256]" = torch.ops.aten.index.Tensor(index_10, [None, None, fmod_6]);  index_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_4: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_random_default_41: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        lt_4: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_41, 0.9869565209373832);  inductor_random_default_41 = None
        convert_element_type_4: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_4, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_8: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_4, 0.9869565209373832);  convert_element_type_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_35: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(index_11, div_8);  index_11 = div_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_41: "f32[128, 28, 28, 256]" = torch.ops.aten.add.Tensor(view_83, mul_35);  view_83 = mul_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_106: "f32[128, 784, 256]" = torch.ops.aten.reshape.default(add_41, [128, -1, 256]);  add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_9 = torch.ops.aten.var_mean.correction(view_106, [2], correction = 0, keepdim = True)
        getitem_30: "f32[128, 784, 1]" = var_mean_9[0]
        getitem_31: "f32[128, 784, 1]" = var_mean_9[1];  var_mean_9 = None
        add_42: "f32[128, 784, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_9: "f32[128, 784, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_13: "f32[128, 784, 256]" = torch.ops.aten.sub.Tensor(view_106, getitem_31);  getitem_31 = None
        mul_36: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_9);  sub_13 = None
        mul_37: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(mul_36, primals_61)
        add_43: "f32[128, 784, 256]" = torch.ops.aten.add.Tensor(mul_37, primals_62);  mul_37 = primals_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_107: "f32[100352, 256]" = torch.ops.aten.reshape.default(add_43, [100352, 256]);  add_43 = None
        permute_41: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_63, [1, 0])
        addmm_14: "f32[100352, 1024]" = torch.ops.aten.addmm.default(primals_64, view_107, permute_41);  primals_64 = permute_41 = None
        view_108: "f32[128, 784, 1024]" = torch.ops.aten.reshape.default(addmm_14, [128, 784, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_38: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(view_108, 0.5)
        mul_39: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(view_108, 0.7071067811865476);  view_108 = None
        erf_3: "f32[128, 784, 1024]" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
        add_44: "f32[128, 784, 1024]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_40: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(mul_38, add_44);  mul_38 = add_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_109: "f32[100352, 1024]" = torch.ops.aten.reshape.default(mul_40, [100352, 1024]);  mul_40 = None
        permute_42: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_65, [1, 0])
        addmm_15: "f32[100352, 256]" = torch.ops.aten.addmm.default(primals_66, view_109, permute_42);  primals_66 = permute_42 = None
        view_110: "f32[128, 784, 256]" = torch.ops.aten.reshape.default(addmm_15, [128, 784, 256]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_5: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_random_default_40: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_5, 'rand');  inductor_lookup_seed_default_5 = None
        lt_5: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_40, 0.9869565209373832);  inductor_random_default_40 = None
        convert_element_type_5: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_5, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_9: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_5, 0.9869565209373832);  convert_element_type_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_41: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(view_110, div_9);  view_110 = div_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_45: "f32[128, 784, 256]" = torch.ops.aten.add.Tensor(view_106, mul_41);  view_106 = mul_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_111: "f32[128, 28, 28, 256]" = torch.ops.aten.reshape.default(add_45, [128, 28, 28, 256]);  add_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:539 in forward, code: x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        view_112: "f32[128, 14, 2, 14, 2, 256]" = torch.ops.aten.reshape.default(view_111, [128, 14, 2, 14, 2, 256]);  view_111 = None
        permute_43: "f32[128, 14, 14, 2, 2, 256]" = torch.ops.aten.permute.default(view_112, [0, 1, 3, 4, 2, 5]);  view_112 = None
        clone_45: "f32[128, 14, 14, 2, 2, 256]" = torch.ops.aten.clone.default(permute_43, memory_format = torch.contiguous_format);  permute_43 = None
        view_113: "f32[128, 14, 14, 1024]" = torch.ops.aten.reshape.default(clone_45, [128, 14, 14, 1024]);  clone_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:540 in forward, code: x = self.norm(x)
        var_mean_10 = torch.ops.aten.var_mean.correction(view_113, [3], correction = 0, keepdim = True)
        getitem_32: "f32[128, 14, 14, 1]" = var_mean_10[0]
        getitem_33: "f32[128, 14, 14, 1]" = var_mean_10[1];  var_mean_10 = None
        add_46: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_10: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_14: "f32[128, 14, 14, 1024]" = torch.ops.aten.sub.Tensor(view_113, getitem_33);  view_113 = getitem_33 = None
        mul_42: "f32[128, 14, 14, 1024]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_10);  sub_14 = None
        mul_43: "f32[128, 14, 14, 1024]" = torch.ops.aten.mul.Tensor(mul_42, primals_67)
        add_47: "f32[128, 14, 14, 1024]" = torch.ops.aten.add.Tensor(mul_43, primals_68);  mul_43 = primals_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:541 in forward, code: x = self.reduction(x)
        permute_44: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_69, [1, 0])
        view_114: "f32[25088, 1024]" = torch.ops.aten.reshape.default(add_47, [25088, 1024]);  add_47 = None
        mm_1: "f32[25088, 512]" = torch.ops.aten.mm.default(view_114, permute_44);  permute_44 = None
        view_115: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(mm_1, [128, 14, 14, 512])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_11 = torch.ops.aten.var_mean.correction(view_115, [3], correction = 0, keepdim = True)
        getitem_34: "f32[128, 14, 14, 1]" = var_mean_11[0]
        getitem_35: "f32[128, 14, 14, 1]" = var_mean_11[1];  var_mean_11 = None
        add_48: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_11: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        sub_15: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_115, getitem_35)
        mul_44: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_11);  sub_15 = None
        mul_45: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_44, primals_70);  mul_44 = None
        add_49: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_45, primals_71);  mul_45 = primals_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_116: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(add_49, [128, 2, 7, 2, 7, 512]);  add_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_45: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_116, [0, 1, 3, 2, 4, 5]);  view_116 = None
        clone_46: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
        view_117: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_46, [-1, 7, 7, 512]);  clone_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_118: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_117, [-1, 49, 512]);  view_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_119: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_118, [25088, 512]);  view_118 = None
        permute_46: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_72, [1, 0])
        addmm_16: "f32[25088, 1536]" = torch.ops.aten.addmm.default(primals_73, view_119, permute_46);  primals_73 = permute_46 = None
        view_120: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(addmm_16, [512, 49, 1536]);  addmm_16 = None
        view_121: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.reshape.default(view_120, [512, 49, 3, 16, -1]);  view_120 = None
        permute_47: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_121, [2, 0, 3, 1, 4]);  view_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_4 = torch.ops.aten.unbind.int(permute_47);  permute_47 = None
        getitem_36: "f32[512, 16, 49, 32]" = unbind_4[0]
        getitem_37: "f32[512, 16, 49, 32]" = unbind_4[1]
        getitem_38: "f32[512, 16, 49, 32]" = unbind_4[2];  unbind_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_46: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_36, 0.1767766952966369);  getitem_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_48: "f32[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_37, [0, 1, 3, 2]);  getitem_37 = None
        expand_16: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_46, [512, 16, 49, 32]);  mul_46 = None
        clone_47: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_122: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_47, [8192, 49, 32]);  clone_47 = None
        expand_17: "f32[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_48, [512, 16, 32, 49]);  permute_48 = None
        clone_48: "f32[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_123: "f32[8192, 32, 49]" = torch.ops.aten.reshape.default(clone_48, [8192, 32, 49]);  clone_48 = None
        bmm_8: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_122, view_123)
        view_124: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_8, [512, 16, 49, 49]);  bmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_125: "i64[2401]" = torch.ops.aten.reshape.default(primals_75, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_12: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_74, [view_125]);  primals_74 = view_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_126: "f32[49, 49, 16]" = torch.ops.aten.reshape.default(index_12, [49, 49, -1]);  index_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_49: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_126, [2, 0, 1]);  view_126 = None
        clone_49: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_8: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_49, 0);  clone_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_50: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_124, unsqueeze_8);  view_124 = unsqueeze_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_4: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(add_50, [-1], True)
        sub_16: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_50, amax_4);  add_50 = amax_4 = None
        exp_4: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_5: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_10: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_18: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_10, [512, 16, 49, 49])
        view_127: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_18, [8192, 49, 49]);  expand_18 = None
        expand_19: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_38, [512, 16, 49, 32]);  getitem_38 = None
        clone_51: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_128: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_51, [8192, 49, 32]);  clone_51 = None
        bmm_9: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_127, view_128);  view_127 = None
        view_129: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_9, [512, 16, 49, 32]);  bmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_50: "f32[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None
        clone_52: "f32[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_50, memory_format = torch.contiguous_format);  permute_50 = None
        view_130: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(clone_52, [512, 49, 512]);  clone_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_131: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_130, [25088, 512]);  view_130 = None
        permute_51: "f32[512, 512]" = torch.ops.aten.permute.default(primals_76, [1, 0])
        addmm_17: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_77, view_131, permute_51);  primals_77 = permute_51 = None
        view_132: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(addmm_17, [512, 49, 512]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_133: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_132, [-1, 7, 7, 512]);  view_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_134: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_133, [-1, 2, 2, 7, 7, 512]);  view_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_52: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_134, [0, 1, 3, 2, 4, 5]);  view_134 = None
        clone_54: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
        view_135: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_54, [-1, 14, 14, 512]);  clone_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_6: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6)
        inductor_random_default_39: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_6, 'rand');  inductor_lookup_seed_default_6 = None
        lt_6: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_39, 0.9826086945831776);  inductor_random_default_39 = None
        convert_element_type_6: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_6, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_11: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_6, 0.9826086945831776);  convert_element_type_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_47: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_135, div_11);  view_135 = div_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_51: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_115, mul_47);  view_115 = mul_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_136: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_51, [128, -1, 512]);  add_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_12 = torch.ops.aten.var_mean.correction(view_136, [2], correction = 0, keepdim = True)
        getitem_39: "f32[128, 196, 1]" = var_mean_12[0]
        getitem_40: "f32[128, 196, 1]" = var_mean_12[1];  var_mean_12 = None
        add_52: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_39, 1e-05);  getitem_39 = None
        rsqrt_12: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        sub_17: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(view_136, getitem_40);  getitem_40 = None
        mul_48: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_12);  sub_17 = None
        mul_49: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_48, primals_78)
        add_53: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_49, primals_79);  mul_49 = primals_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_137: "f32[25088, 512]" = torch.ops.aten.reshape.default(add_53, [25088, 512]);  add_53 = None
        permute_53: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_80, [1, 0])
        addmm_18: "f32[25088, 2048]" = torch.ops.aten.addmm.default(primals_81, view_137, permute_53);  primals_81 = permute_53 = None
        view_138: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_18, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_50: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_138, 0.5)
        mul_51: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_138, 0.7071067811865476);  view_138 = None
        erf_4: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
        add_54: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_52: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_50, add_54);  mul_50 = add_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_139: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_52, [25088, 2048]);  mul_52 = None
        permute_54: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_82, [1, 0])
        addmm_19: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_83, view_139, permute_54);  primals_83 = permute_54 = None
        view_140: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(addmm_19, [128, 196, 512]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_7: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 7)
        inductor_random_default_38: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_7, 'rand');  inductor_lookup_seed_default_7 = None
        lt_7: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_38, 0.9826086945831776);  inductor_random_default_38 = None
        convert_element_type_7: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_7, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_12: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_7, 0.9826086945831776);  convert_element_type_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_53: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_140, div_12);  view_140 = div_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_55: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_136, mul_53);  view_136 = mul_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_141: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_55, [128, 14, 14, 512]);  add_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_13 = torch.ops.aten.var_mean.correction(view_141, [3], correction = 0, keepdim = True)
        getitem_41: "f32[128, 14, 14, 1]" = var_mean_13[0]
        getitem_42: "f32[128, 14, 14, 1]" = var_mean_13[1];  var_mean_13 = None
        add_56: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_41, 1e-05);  getitem_41 = None
        rsqrt_13: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        sub_18: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_141, getitem_42);  getitem_42 = None
        mul_54: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_13);  sub_18 = None
        mul_55: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_54, primals_84)
        add_57: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_55, primals_85);  mul_55 = primals_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota_8: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_58: "i64[14]" = torch.ops.aten.add.Tensor(iota_8, 3)
        fmod_8: "i64[14]" = torch.ops.aten.fmod.Scalar(add_58, 14);  add_58 = None
        index_13: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(add_57, [None, fmod_8]);  add_57 = None
        index_14: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_13, [None, None, fmod_8]);  index_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_142: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(index_14, [128, 2, 7, 2, 7, 512]);  index_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_55: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_142, [0, 1, 3, 2, 4, 5]);  view_142 = None
        clone_57: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_55, memory_format = torch.contiguous_format);  permute_55 = None
        view_143: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_57, [-1, 7, 7, 512]);  clone_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_144: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_143, [-1, 49, 512]);  view_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_145: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_144, [25088, 512]);  view_144 = None
        permute_56: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_87, [1, 0])
        addmm_20: "f32[25088, 1536]" = torch.ops.aten.addmm.default(primals_88, view_145, permute_56);  primals_88 = permute_56 = None
        view_146: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(addmm_20, [512, 49, 1536]);  addmm_20 = None
        view_147: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.reshape.default(view_146, [512, 49, 3, 16, -1]);  view_146 = None
        permute_57: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_147, [2, 0, 3, 1, 4]);  view_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_5 = torch.ops.aten.unbind.int(permute_57);  permute_57 = None
        getitem_43: "f32[512, 16, 49, 32]" = unbind_5[0]
        getitem_44: "f32[512, 16, 49, 32]" = unbind_5[1]
        getitem_45: "f32[512, 16, 49, 32]" = unbind_5[2];  unbind_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_56: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_43, 0.1767766952966369);  getitem_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_58: "f32[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_44, [0, 1, 3, 2]);  getitem_44 = None
        expand_20: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_56, [512, 16, 49, 32]);  mul_56 = None
        clone_58: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_148: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_58, [8192, 49, 32]);  clone_58 = None
        expand_21: "f32[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_58, [512, 16, 32, 49]);  permute_58 = None
        clone_59: "f32[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_149: "f32[8192, 32, 49]" = torch.ops.aten.reshape.default(clone_59, [8192, 32, 49]);  clone_59 = None
        bmm_10: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_148, view_149)
        view_150: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_10, [512, 16, 49, 49]);  bmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_151: "i64[2401]" = torch.ops.aten.reshape.default(primals_90, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_15: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_89, [view_151]);  primals_89 = view_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_152: "f32[49, 49, 16]" = torch.ops.aten.reshape.default(index_15, [49, 49, -1]);  index_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_59: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_152, [2, 0, 1]);  view_152 = None
        clone_60: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_9: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_60, 0);  clone_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_60: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_150, unsqueeze_9);  view_150 = unsqueeze_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_153: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.reshape.default(add_60, [-1, 4, 16, 49, 49]);  add_60 = None
        unsqueeze_10: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(primals_86, 1);  primals_86 = None
        unsqueeze_11: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 0);  unsqueeze_10 = None
        add_61: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_153, unsqueeze_11);  view_153 = unsqueeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_154: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(add_61, [-1, 16, 49, 49]);  add_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_5: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(view_154, [-1], True)
        sub_19: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_154, amax_5);  view_154 = amax_5 = None
        exp_5: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_6: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_13: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_22: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_13, [512, 16, 49, 49])
        view_155: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_22, [8192, 49, 49]);  expand_22 = None
        expand_23: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_45, [512, 16, 49, 32]);  getitem_45 = None
        clone_62: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_156: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_62, [8192, 49, 32]);  clone_62 = None
        bmm_11: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_155, view_156);  view_155 = None
        view_157: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_11, [512, 16, 49, 32]);  bmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_60: "f32[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_157, [0, 2, 1, 3]);  view_157 = None
        clone_63: "f32[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
        view_158: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(clone_63, [512, 49, 512]);  clone_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_159: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_158, [25088, 512]);  view_158 = None
        permute_61: "f32[512, 512]" = torch.ops.aten.permute.default(primals_91, [1, 0])
        addmm_21: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_92, view_159, permute_61);  primals_92 = permute_61 = None
        view_160: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(addmm_21, [512, 49, 512]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_161: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_160, [-1, 7, 7, 512]);  view_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_162: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_161, [-1, 2, 2, 7, 7, 512]);  view_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_62: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_162, [0, 1, 3, 2, 4, 5]);  view_162 = None
        clone_65: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_163: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_65, [-1, 14, 14, 512]);  clone_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        add_62: "i64[14]" = torch.ops.aten.add.Tensor(iota_8, 11);  iota_8 = None
        fmod_10: "i64[14]" = torch.ops.aten.fmod.Scalar(add_62, 14);  add_62 = None
        index_16: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_163, [None, fmod_10]);  view_163 = None
        index_17: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_16, [None, None, fmod_10]);  index_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_8: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 8)
        inductor_random_default_37: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_8, 'rand');  inductor_lookup_seed_default_8 = None
        lt_8: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_37, 0.9782608672976494);  inductor_random_default_37 = None
        convert_element_type_8: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_8, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_14: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_8, 0.9782608672976494);  convert_element_type_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_57: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_17, div_14);  index_17 = div_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_64: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_141, mul_57);  view_141 = mul_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_164: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_64, [128, -1, 512]);  add_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_14 = torch.ops.aten.var_mean.correction(view_164, [2], correction = 0, keepdim = True)
        getitem_46: "f32[128, 196, 1]" = var_mean_14[0]
        getitem_47: "f32[128, 196, 1]" = var_mean_14[1];  var_mean_14 = None
        add_65: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_14: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        sub_20: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(view_164, getitem_47);  getitem_47 = None
        mul_58: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_14);  sub_20 = None
        mul_59: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_58, primals_93)
        add_66: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_59, primals_94);  mul_59 = primals_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_165: "f32[25088, 512]" = torch.ops.aten.reshape.default(add_66, [25088, 512]);  add_66 = None
        permute_63: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_95, [1, 0])
        addmm_22: "f32[25088, 2048]" = torch.ops.aten.addmm.default(primals_96, view_165, permute_63);  primals_96 = permute_63 = None
        view_166: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_22, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_60: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_166, 0.5)
        mul_61: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_166, 0.7071067811865476);  view_166 = None
        erf_5: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
        add_67: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_62: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_60, add_67);  mul_60 = add_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_167: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_62, [25088, 2048]);  mul_62 = None
        permute_64: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_97, [1, 0])
        addmm_23: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_98, view_167, permute_64);  primals_98 = permute_64 = None
        view_168: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(addmm_23, [128, 196, 512]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_9: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 9)
        inductor_random_default_36: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_9, 'rand');  inductor_lookup_seed_default_9 = None
        lt_9: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_36, 0.9782608672976494);  inductor_random_default_36 = None
        convert_element_type_9: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_9, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_15: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_9, 0.9782608672976494);  convert_element_type_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_63: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_168, div_15);  view_168 = div_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_68: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_164, mul_63);  view_164 = mul_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_169: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_68, [128, 14, 14, 512]);  add_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_15 = torch.ops.aten.var_mean.correction(view_169, [3], correction = 0, keepdim = True)
        getitem_48: "f32[128, 14, 14, 1]" = var_mean_15[0]
        getitem_49: "f32[128, 14, 14, 1]" = var_mean_15[1];  var_mean_15 = None
        add_69: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_15: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_21: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_169, getitem_49);  getitem_49 = None
        mul_64: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_15);  sub_21 = None
        mul_65: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_64, primals_99)
        add_70: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_65, primals_100);  mul_65 = primals_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_170: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(add_70, [128, 2, 7, 2, 7, 512]);  add_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_65: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_170, [0, 1, 3, 2, 4, 5]);  view_170 = None
        clone_68: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_65, memory_format = torch.contiguous_format);  permute_65 = None
        view_171: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_68, [-1, 7, 7, 512]);  clone_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_172: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_171, [-1, 49, 512]);  view_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_173: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_172, [25088, 512]);  view_172 = None
        permute_66: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_101, [1, 0])
        addmm_24: "f32[25088, 1536]" = torch.ops.aten.addmm.default(primals_102, view_173, permute_66);  primals_102 = permute_66 = None
        view_174: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(addmm_24, [512, 49, 1536]);  addmm_24 = None
        view_175: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.reshape.default(view_174, [512, 49, 3, 16, -1]);  view_174 = None
        permute_67: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_175, [2, 0, 3, 1, 4]);  view_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_6 = torch.ops.aten.unbind.int(permute_67);  permute_67 = None
        getitem_50: "f32[512, 16, 49, 32]" = unbind_6[0]
        getitem_51: "f32[512, 16, 49, 32]" = unbind_6[1]
        getitem_52: "f32[512, 16, 49, 32]" = unbind_6[2];  unbind_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_66: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_50, 0.1767766952966369);  getitem_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_68: "f32[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_51, [0, 1, 3, 2]);  getitem_51 = None
        expand_24: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_66, [512, 16, 49, 32]);  mul_66 = None
        clone_69: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_176: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_69, [8192, 49, 32]);  clone_69 = None
        expand_25: "f32[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_68, [512, 16, 32, 49]);  permute_68 = None
        clone_70: "f32[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_177: "f32[8192, 32, 49]" = torch.ops.aten.reshape.default(clone_70, [8192, 32, 49]);  clone_70 = None
        bmm_12: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_176, view_177)
        view_178: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_12, [512, 16, 49, 49]);  bmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_179: "i64[2401]" = torch.ops.aten.reshape.default(primals_104, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_18: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_103, [view_179]);  primals_103 = view_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_180: "f32[49, 49, 16]" = torch.ops.aten.reshape.default(index_18, [49, 49, -1]);  index_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_69: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_180, [2, 0, 1]);  view_180 = None
        clone_71: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_69, memory_format = torch.contiguous_format);  permute_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_12: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_71, 0);  clone_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_71: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_178, unsqueeze_12);  view_178 = unsqueeze_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_6: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(add_71, [-1], True)
        sub_22: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_71, amax_6);  add_71 = amax_6 = None
        exp_6: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_7: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_16: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_26: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_16, [512, 16, 49, 49])
        view_181: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_26, [8192, 49, 49]);  expand_26 = None
        expand_27: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_52, [512, 16, 49, 32]);  getitem_52 = None
        clone_73: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_182: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_73, [8192, 49, 32]);  clone_73 = None
        bmm_13: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_181, view_182);  view_181 = None
        view_183: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_13, [512, 16, 49, 32]);  bmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_70: "f32[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None
        clone_74: "f32[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
        view_184: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(clone_74, [512, 49, 512]);  clone_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_185: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_184, [25088, 512]);  view_184 = None
        permute_71: "f32[512, 512]" = torch.ops.aten.permute.default(primals_105, [1, 0])
        addmm_25: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_106, view_185, permute_71);  primals_106 = permute_71 = None
        view_186: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(addmm_25, [512, 49, 512]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_187: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_186, [-1, 7, 7, 512]);  view_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_188: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_187, [-1, 2, 2, 7, 7, 512]);  view_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_72: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_188, [0, 1, 3, 2, 4, 5]);  view_188 = None
        clone_76: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format);  permute_72 = None
        view_189: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_76, [-1, 14, 14, 512]);  clone_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_10: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 10)
        inductor_random_default_35: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_10, 'rand');  inductor_lookup_seed_default_10 = None
        lt_10: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_35, 0.9739130418747663);  inductor_random_default_35 = None
        convert_element_type_10: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_10, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_17: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_10, 0.9739130418747663);  convert_element_type_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_67: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_189, div_17);  view_189 = div_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_72: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_169, mul_67);  view_169 = mul_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_190: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_72, [128, -1, 512]);  add_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_16 = torch.ops.aten.var_mean.correction(view_190, [2], correction = 0, keepdim = True)
        getitem_53: "f32[128, 196, 1]" = var_mean_16[0]
        getitem_54: "f32[128, 196, 1]" = var_mean_16[1];  var_mean_16 = None
        add_73: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_53, 1e-05);  getitem_53 = None
        rsqrt_16: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        sub_23: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(view_190, getitem_54);  getitem_54 = None
        mul_68: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_16);  sub_23 = None
        mul_69: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_68, primals_107)
        add_74: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_69, primals_108);  mul_69 = primals_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_191: "f32[25088, 512]" = torch.ops.aten.reshape.default(add_74, [25088, 512]);  add_74 = None
        permute_73: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_109, [1, 0])
        addmm_26: "f32[25088, 2048]" = torch.ops.aten.addmm.default(primals_110, view_191, permute_73);  primals_110 = permute_73 = None
        view_192: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_26, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_70: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_192, 0.5)
        mul_71: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_192, 0.7071067811865476);  view_192 = None
        erf_6: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_71);  mul_71 = None
        add_75: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_72: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_70, add_75);  mul_70 = add_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_193: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_72, [25088, 2048]);  mul_72 = None
        permute_74: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_111, [1, 0])
        addmm_27: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_112, view_193, permute_74);  primals_112 = permute_74 = None
        view_194: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(addmm_27, [128, 196, 512]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_11: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 11)
        inductor_random_default_34: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_11, 'rand');  inductor_lookup_seed_default_11 = None
        lt_11: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_34, 0.9739130418747663);  inductor_random_default_34 = None
        convert_element_type_11: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_11, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_18: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_11, 0.9739130418747663);  convert_element_type_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_73: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_194, div_18);  view_194 = div_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_76: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_190, mul_73);  view_190 = mul_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_195: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_76, [128, 14, 14, 512]);  add_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_17 = torch.ops.aten.var_mean.correction(view_195, [3], correction = 0, keepdim = True)
        getitem_55: "f32[128, 14, 14, 1]" = var_mean_17[0]
        getitem_56: "f32[128, 14, 14, 1]" = var_mean_17[1];  var_mean_17 = None
        add_77: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_55, 1e-05);  getitem_55 = None
        rsqrt_17: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        sub_24: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_195, getitem_56);  getitem_56 = None
        mul_74: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_17);  sub_24 = None
        mul_75: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_74, primals_113)
        add_78: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_75, primals_114);  mul_75 = primals_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_19: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(add_78, [None, fmod_8]);  add_78 = None
        index_20: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_19, [None, None, fmod_8]);  index_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_196: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(index_20, [128, 2, 7, 2, 7, 512]);  index_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_75: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_196, [0, 1, 3, 2, 4, 5]);  view_196 = None
        clone_79: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_75, memory_format = torch.contiguous_format);  permute_75 = None
        view_197: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_79, [-1, 7, 7, 512]);  clone_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_198: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_197, [-1, 49, 512]);  view_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_199: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_198, [25088, 512]);  view_198 = None
        permute_76: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_116, [1, 0])
        addmm_28: "f32[25088, 1536]" = torch.ops.aten.addmm.default(primals_117, view_199, permute_76);  primals_117 = permute_76 = None
        view_200: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(addmm_28, [512, 49, 1536]);  addmm_28 = None
        view_201: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.reshape.default(view_200, [512, 49, 3, 16, -1]);  view_200 = None
        permute_77: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_201, [2, 0, 3, 1, 4]);  view_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_7 = torch.ops.aten.unbind.int(permute_77);  permute_77 = None
        getitem_57: "f32[512, 16, 49, 32]" = unbind_7[0]
        getitem_58: "f32[512, 16, 49, 32]" = unbind_7[1]
        getitem_59: "f32[512, 16, 49, 32]" = unbind_7[2];  unbind_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_76: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_57, 0.1767766952966369);  getitem_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_78: "f32[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_58, [0, 1, 3, 2]);  getitem_58 = None
        expand_28: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_76, [512, 16, 49, 32]);  mul_76 = None
        clone_80: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_202: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_80, [8192, 49, 32]);  clone_80 = None
        expand_29: "f32[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_78, [512, 16, 32, 49]);  permute_78 = None
        clone_81: "f32[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_203: "f32[8192, 32, 49]" = torch.ops.aten.reshape.default(clone_81, [8192, 32, 49]);  clone_81 = None
        bmm_14: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_202, view_203)
        view_204: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_14, [512, 16, 49, 49]);  bmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_205: "i64[2401]" = torch.ops.aten.reshape.default(primals_119, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_21: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_118, [view_205]);  primals_118 = view_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_206: "f32[49, 49, 16]" = torch.ops.aten.reshape.default(index_21, [49, 49, -1]);  index_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_79: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_206, [2, 0, 1]);  view_206 = None
        clone_82: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_13: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_82, 0);  clone_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_81: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_204, unsqueeze_13);  view_204 = unsqueeze_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_207: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.reshape.default(add_81, [-1, 4, 16, 49, 49]);  add_81 = None
        unsqueeze_14: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(primals_115, 1);  primals_115 = None
        unsqueeze_15: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, 0);  unsqueeze_14 = None
        add_82: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_207, unsqueeze_15);  view_207 = unsqueeze_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_208: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(add_82, [-1, 16, 49, 49]);  add_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_7: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(view_208, [-1], True)
        sub_25: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_208, amax_7);  view_208 = amax_7 = None
        exp_7: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_8: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_19: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_30: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_19, [512, 16, 49, 49])
        view_209: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_30, [8192, 49, 49]);  expand_30 = None
        expand_31: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_59, [512, 16, 49, 32]);  getitem_59 = None
        clone_84: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_210: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_84, [8192, 49, 32]);  clone_84 = None
        bmm_15: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_209, view_210);  view_209 = None
        view_211: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_15, [512, 16, 49, 32]);  bmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_80: "f32[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_211, [0, 2, 1, 3]);  view_211 = None
        clone_85: "f32[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
        view_212: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(clone_85, [512, 49, 512]);  clone_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_213: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_212, [25088, 512]);  view_212 = None
        permute_81: "f32[512, 512]" = torch.ops.aten.permute.default(primals_120, [1, 0])
        addmm_29: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_121, view_213, permute_81);  primals_121 = permute_81 = None
        view_214: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(addmm_29, [512, 49, 512]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_215: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_214, [-1, 7, 7, 512]);  view_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_216: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_215, [-1, 2, 2, 7, 7, 512]);  view_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_82: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_216, [0, 1, 3, 2, 4, 5]);  view_216 = None
        clone_87: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        view_217: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_87, [-1, 14, 14, 512]);  clone_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_22: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_217, [None, fmod_10]);  view_217 = None
        index_23: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_22, [None, None, fmod_10]);  index_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_12: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 12)
        inductor_random_default_33: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_12, 'rand');  inductor_lookup_seed_default_12 = None
        lt_12: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_33, 0.9695652164518833);  inductor_random_default_33 = None
        convert_element_type_12: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_12, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_20: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_12, 0.9695652164518833);  convert_element_type_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_77: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_23, div_20);  index_23 = div_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_85: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_195, mul_77);  view_195 = mul_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_218: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_85, [128, -1, 512]);  add_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_18 = torch.ops.aten.var_mean.correction(view_218, [2], correction = 0, keepdim = True)
        getitem_60: "f32[128, 196, 1]" = var_mean_18[0]
        getitem_61: "f32[128, 196, 1]" = var_mean_18[1];  var_mean_18 = None
        add_86: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
        rsqrt_18: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        sub_26: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(view_218, getitem_61);  getitem_61 = None
        mul_78: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_18);  sub_26 = None
        mul_79: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_78, primals_122)
        add_87: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_79, primals_123);  mul_79 = primals_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_219: "f32[25088, 512]" = torch.ops.aten.reshape.default(add_87, [25088, 512]);  add_87 = None
        permute_83: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_124, [1, 0])
        addmm_30: "f32[25088, 2048]" = torch.ops.aten.addmm.default(primals_125, view_219, permute_83);  primals_125 = permute_83 = None
        view_220: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_30, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_80: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_220, 0.5)
        mul_81: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_220, 0.7071067811865476);  view_220 = None
        erf_7: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_81);  mul_81 = None
        add_88: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_82: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_80, add_88);  mul_80 = add_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_221: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_82, [25088, 2048]);  mul_82 = None
        permute_84: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_126, [1, 0])
        addmm_31: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_127, view_221, permute_84);  primals_127 = permute_84 = None
        view_222: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(addmm_31, [128, 196, 512]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_13: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 13)
        inductor_random_default_32: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_13, 'rand');  inductor_lookup_seed_default_13 = None
        lt_13: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_32, 0.9695652164518833);  inductor_random_default_32 = None
        convert_element_type_13: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_13, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_21: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_13, 0.9695652164518833);  convert_element_type_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_83: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_222, div_21);  view_222 = div_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_89: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_218, mul_83);  view_218 = mul_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_223: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_89, [128, 14, 14, 512]);  add_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_19 = torch.ops.aten.var_mean.correction(view_223, [3], correction = 0, keepdim = True)
        getitem_62: "f32[128, 14, 14, 1]" = var_mean_19[0]
        getitem_63: "f32[128, 14, 14, 1]" = var_mean_19[1];  var_mean_19 = None
        add_90: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
        rsqrt_19: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        sub_27: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_223, getitem_63);  getitem_63 = None
        mul_84: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_19);  sub_27 = None
        mul_85: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_84, primals_128)
        add_91: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_85, primals_129);  mul_85 = primals_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_224: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(add_91, [128, 2, 7, 2, 7, 512]);  add_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_85: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_224, [0, 1, 3, 2, 4, 5]);  view_224 = None
        clone_90: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None
        view_225: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_90, [-1, 7, 7, 512]);  clone_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_226: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_225, [-1, 49, 512]);  view_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_227: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_226, [25088, 512]);  view_226 = None
        permute_86: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_130, [1, 0])
        addmm_32: "f32[25088, 1536]" = torch.ops.aten.addmm.default(primals_131, view_227, permute_86);  primals_131 = permute_86 = None
        view_228: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(addmm_32, [512, 49, 1536]);  addmm_32 = None
        view_229: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.reshape.default(view_228, [512, 49, 3, 16, -1]);  view_228 = None
        permute_87: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_229, [2, 0, 3, 1, 4]);  view_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_8 = torch.ops.aten.unbind.int(permute_87);  permute_87 = None
        getitem_64: "f32[512, 16, 49, 32]" = unbind_8[0]
        getitem_65: "f32[512, 16, 49, 32]" = unbind_8[1]
        getitem_66: "f32[512, 16, 49, 32]" = unbind_8[2];  unbind_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_86: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_64, 0.1767766952966369);  getitem_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_88: "f32[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_65, [0, 1, 3, 2]);  getitem_65 = None
        expand_32: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_86, [512, 16, 49, 32]);  mul_86 = None
        clone_91: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_230: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_91, [8192, 49, 32]);  clone_91 = None
        expand_33: "f32[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_88, [512, 16, 32, 49]);  permute_88 = None
        clone_92: "f32[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_231: "f32[8192, 32, 49]" = torch.ops.aten.reshape.default(clone_92, [8192, 32, 49]);  clone_92 = None
        bmm_16: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_230, view_231)
        view_232: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_16, [512, 16, 49, 49]);  bmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_233: "i64[2401]" = torch.ops.aten.reshape.default(primals_133, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_24: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_132, [view_233]);  primals_132 = view_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_234: "f32[49, 49, 16]" = torch.ops.aten.reshape.default(index_24, [49, 49, -1]);  index_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_89: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_234, [2, 0, 1]);  view_234 = None
        clone_93: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_89, memory_format = torch.contiguous_format);  permute_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_16: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_93, 0);  clone_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_92: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_232, unsqueeze_16);  view_232 = unsqueeze_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_8: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(add_92, [-1], True)
        sub_28: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_92, amax_8);  add_92 = amax_8 = None
        exp_8: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_9: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_22: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_34: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_22, [512, 16, 49, 49])
        view_235: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_34, [8192, 49, 49]);  expand_34 = None
        expand_35: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_66, [512, 16, 49, 32]);  getitem_66 = None
        clone_95: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_236: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_95, [8192, 49, 32]);  clone_95 = None
        bmm_17: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_235, view_236);  view_235 = None
        view_237: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_17, [512, 16, 49, 32]);  bmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_90: "f32[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_237, [0, 2, 1, 3]);  view_237 = None
        clone_96: "f32[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
        view_238: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(clone_96, [512, 49, 512]);  clone_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_239: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_238, [25088, 512]);  view_238 = None
        permute_91: "f32[512, 512]" = torch.ops.aten.permute.default(primals_134, [1, 0])
        addmm_33: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_135, view_239, permute_91);  primals_135 = permute_91 = None
        view_240: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(addmm_33, [512, 49, 512]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_241: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_240, [-1, 7, 7, 512]);  view_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_242: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_241, [-1, 2, 2, 7, 7, 512]);  view_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_92: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_242, [0, 1, 3, 2, 4, 5]);  view_242 = None
        clone_98: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
        view_243: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_98, [-1, 14, 14, 512]);  clone_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_14: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 14)
        inductor_random_default_31: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_14, 'rand');  inductor_lookup_seed_default_14 = None
        lt_14: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_31, 0.9652173891663551);  inductor_random_default_31 = None
        convert_element_type_14: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_14, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_23: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_14, 0.9652173891663551);  convert_element_type_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_87: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_243, div_23);  view_243 = div_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_93: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_223, mul_87);  view_223 = mul_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_244: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_93, [128, -1, 512]);  add_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_20 = torch.ops.aten.var_mean.correction(view_244, [2], correction = 0, keepdim = True)
        getitem_67: "f32[128, 196, 1]" = var_mean_20[0]
        getitem_68: "f32[128, 196, 1]" = var_mean_20[1];  var_mean_20 = None
        add_94: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_67, 1e-05);  getitem_67 = None
        rsqrt_20: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        sub_29: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(view_244, getitem_68);  getitem_68 = None
        mul_88: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_20);  sub_29 = None
        mul_89: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_88, primals_136)
        add_95: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_89, primals_137);  mul_89 = primals_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_245: "f32[25088, 512]" = torch.ops.aten.reshape.default(add_95, [25088, 512]);  add_95 = None
        permute_93: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_138, [1, 0])
        addmm_34: "f32[25088, 2048]" = torch.ops.aten.addmm.default(primals_139, view_245, permute_93);  primals_139 = permute_93 = None
        view_246: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_34, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_90: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_246, 0.5)
        mul_91: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_246, 0.7071067811865476);  view_246 = None
        erf_8: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_91);  mul_91 = None
        add_96: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_92: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_90, add_96);  mul_90 = add_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_247: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_92, [25088, 2048]);  mul_92 = None
        permute_94: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_140, [1, 0])
        addmm_35: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_141, view_247, permute_94);  primals_141 = permute_94 = None
        view_248: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(addmm_35, [128, 196, 512]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_15: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 15)
        inductor_random_default_30: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_15, 'rand');  inductor_lookup_seed_default_15 = None
        lt_15: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_30, 0.9652173891663551);  inductor_random_default_30 = None
        convert_element_type_15: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_15, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_24: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_15, 0.9652173891663551);  convert_element_type_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_93: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_248, div_24);  view_248 = div_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_97: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_244, mul_93);  view_244 = mul_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_249: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_97, [128, 14, 14, 512]);  add_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_21 = torch.ops.aten.var_mean.correction(view_249, [3], correction = 0, keepdim = True)
        getitem_69: "f32[128, 14, 14, 1]" = var_mean_21[0]
        getitem_70: "f32[128, 14, 14, 1]" = var_mean_21[1];  var_mean_21 = None
        add_98: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_69, 1e-05);  getitem_69 = None
        rsqrt_21: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        sub_30: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_249, getitem_70);  getitem_70 = None
        mul_94: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_21);  sub_30 = None
        mul_95: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_94, primals_142)
        add_99: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_95, primals_143);  mul_95 = primals_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_25: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(add_99, [None, fmod_8]);  add_99 = None
        index_26: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_25, [None, None, fmod_8]);  index_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_250: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(index_26, [128, 2, 7, 2, 7, 512]);  index_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_95: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_250, [0, 1, 3, 2, 4, 5]);  view_250 = None
        clone_101: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_251: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_101, [-1, 7, 7, 512]);  clone_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_252: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_251, [-1, 49, 512]);  view_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_253: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_252, [25088, 512]);  view_252 = None
        permute_96: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_145, [1, 0])
        addmm_36: "f32[25088, 1536]" = torch.ops.aten.addmm.default(primals_146, view_253, permute_96);  primals_146 = permute_96 = None
        view_254: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(addmm_36, [512, 49, 1536]);  addmm_36 = None
        view_255: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.reshape.default(view_254, [512, 49, 3, 16, -1]);  view_254 = None
        permute_97: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_255, [2, 0, 3, 1, 4]);  view_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_9 = torch.ops.aten.unbind.int(permute_97);  permute_97 = None
        getitem_71: "f32[512, 16, 49, 32]" = unbind_9[0]
        getitem_72: "f32[512, 16, 49, 32]" = unbind_9[1]
        getitem_73: "f32[512, 16, 49, 32]" = unbind_9[2];  unbind_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_96: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_71, 0.1767766952966369);  getitem_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_98: "f32[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_72, [0, 1, 3, 2]);  getitem_72 = None
        expand_36: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_96, [512, 16, 49, 32]);  mul_96 = None
        clone_102: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_256: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_102, [8192, 49, 32]);  clone_102 = None
        expand_37: "f32[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_98, [512, 16, 32, 49]);  permute_98 = None
        clone_103: "f32[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_257: "f32[8192, 32, 49]" = torch.ops.aten.reshape.default(clone_103, [8192, 32, 49]);  clone_103 = None
        bmm_18: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_256, view_257)
        view_258: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_18, [512, 16, 49, 49]);  bmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_259: "i64[2401]" = torch.ops.aten.reshape.default(primals_148, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_27: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_147, [view_259]);  primals_147 = view_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_260: "f32[49, 49, 16]" = torch.ops.aten.reshape.default(index_27, [49, 49, -1]);  index_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_99: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_260, [2, 0, 1]);  view_260 = None
        clone_104: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_99, memory_format = torch.contiguous_format);  permute_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_17: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_104, 0);  clone_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_102: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_258, unsqueeze_17);  view_258 = unsqueeze_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_261: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.reshape.default(add_102, [-1, 4, 16, 49, 49]);  add_102 = None
        unsqueeze_18: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(primals_144, 1);  primals_144 = None
        unsqueeze_19: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, 0);  unsqueeze_18 = None
        add_103: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_261, unsqueeze_19);  view_261 = unsqueeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_262: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(add_103, [-1, 16, 49, 49]);  add_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_9: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(view_262, [-1], True)
        sub_31: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_262, amax_9);  view_262 = amax_9 = None
        exp_9: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_10: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_25: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_38: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_25, [512, 16, 49, 49])
        view_263: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_38, [8192, 49, 49]);  expand_38 = None
        expand_39: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_73, [512, 16, 49, 32]);  getitem_73 = None
        clone_106: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_264: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_106, [8192, 49, 32]);  clone_106 = None
        bmm_19: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_263, view_264);  view_263 = None
        view_265: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_19, [512, 16, 49, 32]);  bmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_100: "f32[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_265, [0, 2, 1, 3]);  view_265 = None
        clone_107: "f32[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_100, memory_format = torch.contiguous_format);  permute_100 = None
        view_266: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(clone_107, [512, 49, 512]);  clone_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_267: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_266, [25088, 512]);  view_266 = None
        permute_101: "f32[512, 512]" = torch.ops.aten.permute.default(primals_149, [1, 0])
        addmm_37: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_150, view_267, permute_101);  primals_150 = permute_101 = None
        view_268: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(addmm_37, [512, 49, 512]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_269: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_268, [-1, 7, 7, 512]);  view_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_270: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_269, [-1, 2, 2, 7, 7, 512]);  view_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_102: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_270, [0, 1, 3, 2, 4, 5]);  view_270 = None
        clone_109: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
        view_271: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_109, [-1, 14, 14, 512]);  clone_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_28: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_271, [None, fmod_10]);  view_271 = None
        index_29: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_28, [None, None, fmod_10]);  index_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_16: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 16)
        inductor_random_default_29: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_16, 'rand');  inductor_lookup_seed_default_16 = None
        lt_16: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_29, 0.960869561880827);  inductor_random_default_29 = None
        convert_element_type_16: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_16, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_26: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_16, 0.960869561880827);  convert_element_type_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_97: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_29, div_26);  index_29 = div_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_106: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_249, mul_97);  view_249 = mul_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_272: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_106, [128, -1, 512]);  add_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_22 = torch.ops.aten.var_mean.correction(view_272, [2], correction = 0, keepdim = True)
        getitem_74: "f32[128, 196, 1]" = var_mean_22[0]
        getitem_75: "f32[128, 196, 1]" = var_mean_22[1];  var_mean_22 = None
        add_107: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
        rsqrt_22: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_107);  add_107 = None
        sub_32: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(view_272, getitem_75);  getitem_75 = None
        mul_98: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_22);  sub_32 = None
        mul_99: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_98, primals_151)
        add_108: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_99, primals_152);  mul_99 = primals_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_273: "f32[25088, 512]" = torch.ops.aten.reshape.default(add_108, [25088, 512]);  add_108 = None
        permute_103: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_153, [1, 0])
        addmm_38: "f32[25088, 2048]" = torch.ops.aten.addmm.default(primals_154, view_273, permute_103);  primals_154 = permute_103 = None
        view_274: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_38, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_100: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_274, 0.5)
        mul_101: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_274, 0.7071067811865476);  view_274 = None
        erf_9: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_101);  mul_101 = None
        add_109: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_102: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_100, add_109);  mul_100 = add_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_275: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_102, [25088, 2048]);  mul_102 = None
        permute_104: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_155, [1, 0])
        addmm_39: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_156, view_275, permute_104);  primals_156 = permute_104 = None
        view_276: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(addmm_39, [128, 196, 512]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_17: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 17)
        inductor_random_default_28: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_17, 'rand');  inductor_lookup_seed_default_17 = None
        lt_17: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_28, 0.960869561880827);  inductor_random_default_28 = None
        convert_element_type_17: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_17, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_27: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_17, 0.960869561880827);  convert_element_type_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_103: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_276, div_27);  view_276 = div_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_110: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_272, mul_103);  view_272 = mul_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_277: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_110, [128, 14, 14, 512]);  add_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_23 = torch.ops.aten.var_mean.correction(view_277, [3], correction = 0, keepdim = True)
        getitem_76: "f32[128, 14, 14, 1]" = var_mean_23[0]
        getitem_77: "f32[128, 14, 14, 1]" = var_mean_23[1];  var_mean_23 = None
        add_111: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
        rsqrt_23: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
        sub_33: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_277, getitem_77);  getitem_77 = None
        mul_104: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_23);  sub_33 = None
        mul_105: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_104, primals_157)
        add_112: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_105, primals_158);  mul_105 = primals_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_278: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(add_112, [128, 2, 7, 2, 7, 512]);  add_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_105: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_278, [0, 1, 3, 2, 4, 5]);  view_278 = None
        clone_112: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_105, memory_format = torch.contiguous_format);  permute_105 = None
        view_279: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_112, [-1, 7, 7, 512]);  clone_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_280: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_279, [-1, 49, 512]);  view_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_281: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_280, [25088, 512]);  view_280 = None
        permute_106: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_159, [1, 0])
        addmm_40: "f32[25088, 1536]" = torch.ops.aten.addmm.default(primals_160, view_281, permute_106);  primals_160 = permute_106 = None
        view_282: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(addmm_40, [512, 49, 1536]);  addmm_40 = None
        view_283: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.reshape.default(view_282, [512, 49, 3, 16, -1]);  view_282 = None
        permute_107: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_283, [2, 0, 3, 1, 4]);  view_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_10 = torch.ops.aten.unbind.int(permute_107);  permute_107 = None
        getitem_78: "f32[512, 16, 49, 32]" = unbind_10[0]
        getitem_79: "f32[512, 16, 49, 32]" = unbind_10[1]
        getitem_80: "f32[512, 16, 49, 32]" = unbind_10[2];  unbind_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_106: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_78, 0.1767766952966369);  getitem_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_108: "f32[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_79, [0, 1, 3, 2]);  getitem_79 = None
        expand_40: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_106, [512, 16, 49, 32]);  mul_106 = None
        clone_113: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_284: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_113, [8192, 49, 32]);  clone_113 = None
        expand_41: "f32[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_108, [512, 16, 32, 49]);  permute_108 = None
        clone_114: "f32[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_285: "f32[8192, 32, 49]" = torch.ops.aten.reshape.default(clone_114, [8192, 32, 49]);  clone_114 = None
        bmm_20: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_284, view_285)
        view_286: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_20, [512, 16, 49, 49]);  bmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_287: "i64[2401]" = torch.ops.aten.reshape.default(primals_162, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_30: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_161, [view_287]);  primals_161 = view_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_288: "f32[49, 49, 16]" = torch.ops.aten.reshape.default(index_30, [49, 49, -1]);  index_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_109: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_288, [2, 0, 1]);  view_288 = None
        clone_115: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_109, memory_format = torch.contiguous_format);  permute_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_20: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_115, 0);  clone_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_113: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_286, unsqueeze_20);  view_286 = unsqueeze_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_10: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(add_113, [-1], True)
        sub_34: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_113, amax_10);  add_113 = amax_10 = None
        exp_10: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_11: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_28: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_42: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_28, [512, 16, 49, 49])
        view_289: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_42, [8192, 49, 49]);  expand_42 = None
        expand_43: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_80, [512, 16, 49, 32]);  getitem_80 = None
        clone_117: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_290: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_117, [8192, 49, 32]);  clone_117 = None
        bmm_21: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_289, view_290);  view_289 = None
        view_291: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_21, [512, 16, 49, 32]);  bmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_110: "f32[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_291, [0, 2, 1, 3]);  view_291 = None
        clone_118: "f32[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None
        view_292: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(clone_118, [512, 49, 512]);  clone_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_293: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_292, [25088, 512]);  view_292 = None
        permute_111: "f32[512, 512]" = torch.ops.aten.permute.default(primals_163, [1, 0])
        addmm_41: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_164, view_293, permute_111);  primals_164 = permute_111 = None
        view_294: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(addmm_41, [512, 49, 512]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_295: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_294, [-1, 7, 7, 512]);  view_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_296: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_295, [-1, 2, 2, 7, 7, 512]);  view_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_112: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_296, [0, 1, 3, 2, 4, 5]);  view_296 = None
        clone_120: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
        view_297: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_120, [-1, 14, 14, 512]);  clone_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_18: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 18)
        inductor_random_default_27: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_18, 'rand');  inductor_lookup_seed_default_18 = None
        lt_18: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_27, 0.9565217345952988);  inductor_random_default_27 = None
        convert_element_type_18: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_18, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_29: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_18, 0.9565217345952988);  convert_element_type_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_107: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_297, div_29);  view_297 = div_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_114: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_277, mul_107);  view_277 = mul_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_298: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_114, [128, -1, 512]);  add_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_24 = torch.ops.aten.var_mean.correction(view_298, [2], correction = 0, keepdim = True)
        getitem_81: "f32[128, 196, 1]" = var_mean_24[0]
        getitem_82: "f32[128, 196, 1]" = var_mean_24[1];  var_mean_24 = None
        add_115: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_81, 1e-05);  getitem_81 = None
        rsqrt_24: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
        sub_35: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(view_298, getitem_82);  getitem_82 = None
        mul_108: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_24);  sub_35 = None
        mul_109: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_108, primals_165)
        add_116: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_109, primals_166);  mul_109 = primals_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_299: "f32[25088, 512]" = torch.ops.aten.reshape.default(add_116, [25088, 512]);  add_116 = None
        permute_113: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_167, [1, 0])
        addmm_42: "f32[25088, 2048]" = torch.ops.aten.addmm.default(primals_168, view_299, permute_113);  primals_168 = permute_113 = None
        view_300: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_42, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_110: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_300, 0.5)
        mul_111: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_300, 0.7071067811865476);  view_300 = None
        erf_10: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_111);  mul_111 = None
        add_117: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_112: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_110, add_117);  mul_110 = add_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_301: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_112, [25088, 2048]);  mul_112 = None
        permute_114: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_169, [1, 0])
        addmm_43: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_170, view_301, permute_114);  primals_170 = permute_114 = None
        view_302: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(addmm_43, [128, 196, 512]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_19: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 19)
        inductor_random_default_26: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_19, 'rand');  inductor_lookup_seed_default_19 = None
        lt_19: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_26, 0.9565217345952988);  inductor_random_default_26 = None
        convert_element_type_19: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_19, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_30: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_19, 0.9565217345952988);  convert_element_type_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_113: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_302, div_30);  view_302 = div_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_118: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_298, mul_113);  view_298 = mul_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_303: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_118, [128, 14, 14, 512]);  add_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_25 = torch.ops.aten.var_mean.correction(view_303, [3], correction = 0, keepdim = True)
        getitem_83: "f32[128, 14, 14, 1]" = var_mean_25[0]
        getitem_84: "f32[128, 14, 14, 1]" = var_mean_25[1];  var_mean_25 = None
        add_119: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_83, 1e-05);  getitem_83 = None
        rsqrt_25: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
        sub_36: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_303, getitem_84);  getitem_84 = None
        mul_114: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_25);  sub_36 = None
        mul_115: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_114, primals_171)
        add_120: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_115, primals_172);  mul_115 = primals_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_31: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(add_120, [None, fmod_8]);  add_120 = None
        index_32: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_31, [None, None, fmod_8]);  index_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_304: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(index_32, [128, 2, 7, 2, 7, 512]);  index_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_115: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_304, [0, 1, 3, 2, 4, 5]);  view_304 = None
        clone_123: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
        view_305: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_123, [-1, 7, 7, 512]);  clone_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_306: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_305, [-1, 49, 512]);  view_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_307: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_306, [25088, 512]);  view_306 = None
        permute_116: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_174, [1, 0])
        addmm_44: "f32[25088, 1536]" = torch.ops.aten.addmm.default(primals_175, view_307, permute_116);  primals_175 = permute_116 = None
        view_308: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(addmm_44, [512, 49, 1536]);  addmm_44 = None
        view_309: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.reshape.default(view_308, [512, 49, 3, 16, -1]);  view_308 = None
        permute_117: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_309, [2, 0, 3, 1, 4]);  view_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_11 = torch.ops.aten.unbind.int(permute_117);  permute_117 = None
        getitem_85: "f32[512, 16, 49, 32]" = unbind_11[0]
        getitem_86: "f32[512, 16, 49, 32]" = unbind_11[1]
        getitem_87: "f32[512, 16, 49, 32]" = unbind_11[2];  unbind_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_116: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_85, 0.1767766952966369);  getitem_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_118: "f32[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_86, [0, 1, 3, 2]);  getitem_86 = None
        expand_44: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_116, [512, 16, 49, 32]);  mul_116 = None
        clone_124: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_310: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_124, [8192, 49, 32]);  clone_124 = None
        expand_45: "f32[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_118, [512, 16, 32, 49]);  permute_118 = None
        clone_125: "f32[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_311: "f32[8192, 32, 49]" = torch.ops.aten.reshape.default(clone_125, [8192, 32, 49]);  clone_125 = None
        bmm_22: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_310, view_311)
        view_312: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_22, [512, 16, 49, 49]);  bmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_313: "i64[2401]" = torch.ops.aten.reshape.default(primals_177, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_33: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_176, [view_313]);  primals_176 = view_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_314: "f32[49, 49, 16]" = torch.ops.aten.reshape.default(index_33, [49, 49, -1]);  index_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_119: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_314, [2, 0, 1]);  view_314 = None
        clone_126: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_119, memory_format = torch.contiguous_format);  permute_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_21: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_126, 0);  clone_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_123: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_312, unsqueeze_21);  view_312 = unsqueeze_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_315: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.reshape.default(add_123, [-1, 4, 16, 49, 49]);  add_123 = None
        unsqueeze_22: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(primals_173, 1);  primals_173 = None
        unsqueeze_23: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, 0);  unsqueeze_22 = None
        add_124: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_315, unsqueeze_23);  view_315 = unsqueeze_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_316: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(add_124, [-1, 16, 49, 49]);  add_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_11: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(view_316, [-1], True)
        sub_37: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_316, amax_11);  view_316 = amax_11 = None
        exp_11: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
        sum_12: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_31: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_46: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_31, [512, 16, 49, 49])
        view_317: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_46, [8192, 49, 49]);  expand_46 = None
        expand_47: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_87, [512, 16, 49, 32]);  getitem_87 = None
        clone_128: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_318: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_128, [8192, 49, 32]);  clone_128 = None
        bmm_23: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_317, view_318);  view_317 = None
        view_319: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_23, [512, 16, 49, 32]);  bmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_120: "f32[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_319, [0, 2, 1, 3]);  view_319 = None
        clone_129: "f32[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_120, memory_format = torch.contiguous_format);  permute_120 = None
        view_320: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(clone_129, [512, 49, 512]);  clone_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_321: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_320, [25088, 512]);  view_320 = None
        permute_121: "f32[512, 512]" = torch.ops.aten.permute.default(primals_178, [1, 0])
        addmm_45: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_179, view_321, permute_121);  primals_179 = permute_121 = None
        view_322: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(addmm_45, [512, 49, 512]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_323: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_322, [-1, 7, 7, 512]);  view_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_324: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_323, [-1, 2, 2, 7, 7, 512]);  view_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_122: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_324, [0, 1, 3, 2, 4, 5]);  view_324 = None
        clone_131: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
        view_325: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_131, [-1, 14, 14, 512]);  clone_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_34: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_325, [None, fmod_10]);  view_325 = None
        index_35: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_34, [None, None, fmod_10]);  index_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_20: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 20)
        inductor_random_default_25: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_20, 'rand');  inductor_lookup_seed_default_20 = None
        lt_20: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_25, 0.9521739110350609);  inductor_random_default_25 = None
        convert_element_type_20: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_20, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_32: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_20, 0.9521739110350609);  convert_element_type_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_117: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_35, div_32);  index_35 = div_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_127: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_303, mul_117);  view_303 = mul_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_326: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_127, [128, -1, 512]);  add_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_26 = torch.ops.aten.var_mean.correction(view_326, [2], correction = 0, keepdim = True)
        getitem_88: "f32[128, 196, 1]" = var_mean_26[0]
        getitem_89: "f32[128, 196, 1]" = var_mean_26[1];  var_mean_26 = None
        add_128: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_26: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
        sub_38: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(view_326, getitem_89);  getitem_89 = None
        mul_118: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_26);  sub_38 = None
        mul_119: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_118, primals_180)
        add_129: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_119, primals_181);  mul_119 = primals_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_327: "f32[25088, 512]" = torch.ops.aten.reshape.default(add_129, [25088, 512]);  add_129 = None
        permute_123: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_182, [1, 0])
        addmm_46: "f32[25088, 2048]" = torch.ops.aten.addmm.default(primals_183, view_327, permute_123);  primals_183 = permute_123 = None
        view_328: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_46, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_120: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_328, 0.5)
        mul_121: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_328, 0.7071067811865476);  view_328 = None
        erf_11: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_121);  mul_121 = None
        add_130: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_122: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_120, add_130);  mul_120 = add_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_329: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_122, [25088, 2048]);  mul_122 = None
        permute_124: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_184, [1, 0])
        addmm_47: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_185, view_329, permute_124);  primals_185 = permute_124 = None
        view_330: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(addmm_47, [128, 196, 512]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_21: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 21)
        inductor_random_default_24: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_21, 'rand');  inductor_lookup_seed_default_21 = None
        lt_21: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_24, 0.9521739110350609);  inductor_random_default_24 = None
        convert_element_type_21: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_21, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_33: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_21, 0.9521739110350609);  convert_element_type_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_123: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_330, div_33);  view_330 = div_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_131: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_326, mul_123);  view_326 = mul_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_331: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_131, [128, 14, 14, 512]);  add_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_27 = torch.ops.aten.var_mean.correction(view_331, [3], correction = 0, keepdim = True)
        getitem_90: "f32[128, 14, 14, 1]" = var_mean_27[0]
        getitem_91: "f32[128, 14, 14, 1]" = var_mean_27[1];  var_mean_27 = None
        add_132: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
        rsqrt_27: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
        sub_39: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_331, getitem_91);  getitem_91 = None
        mul_124: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_27);  sub_39 = None
        mul_125: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_124, primals_186)
        add_133: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_125, primals_187);  mul_125 = primals_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_332: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(add_133, [128, 2, 7, 2, 7, 512]);  add_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_125: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_332, [0, 1, 3, 2, 4, 5]);  view_332 = None
        clone_134: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
        view_333: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_134, [-1, 7, 7, 512]);  clone_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_334: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_333, [-1, 49, 512]);  view_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_335: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_334, [25088, 512]);  view_334 = None
        permute_126: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_188, [1, 0])
        addmm_48: "f32[25088, 1536]" = torch.ops.aten.addmm.default(primals_189, view_335, permute_126);  primals_189 = permute_126 = None
        view_336: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(addmm_48, [512, 49, 1536]);  addmm_48 = None
        view_337: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.reshape.default(view_336, [512, 49, 3, 16, -1]);  view_336 = None
        permute_127: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_337, [2, 0, 3, 1, 4]);  view_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_12 = torch.ops.aten.unbind.int(permute_127);  permute_127 = None
        getitem_92: "f32[512, 16, 49, 32]" = unbind_12[0]
        getitem_93: "f32[512, 16, 49, 32]" = unbind_12[1]
        getitem_94: "f32[512, 16, 49, 32]" = unbind_12[2];  unbind_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_126: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_92, 0.1767766952966369);  getitem_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_128: "f32[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_93, [0, 1, 3, 2]);  getitem_93 = None
        expand_48: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_126, [512, 16, 49, 32]);  mul_126 = None
        clone_135: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_338: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_135, [8192, 49, 32]);  clone_135 = None
        expand_49: "f32[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_128, [512, 16, 32, 49]);  permute_128 = None
        clone_136: "f32[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_339: "f32[8192, 32, 49]" = torch.ops.aten.reshape.default(clone_136, [8192, 32, 49]);  clone_136 = None
        bmm_24: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_338, view_339)
        view_340: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_24, [512, 16, 49, 49]);  bmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_341: "i64[2401]" = torch.ops.aten.reshape.default(primals_191, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_36: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_190, [view_341]);  primals_190 = view_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_342: "f32[49, 49, 16]" = torch.ops.aten.reshape.default(index_36, [49, 49, -1]);  index_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_129: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_342, [2, 0, 1]);  view_342 = None
        clone_137: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_24: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_137, 0);  clone_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_134: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_340, unsqueeze_24);  view_340 = unsqueeze_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_12: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(add_134, [-1], True)
        sub_40: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_134, amax_12);  add_134 = amax_12 = None
        exp_12: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_40);  sub_40 = None
        sum_13: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_34: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_50: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_34, [512, 16, 49, 49])
        view_343: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_50, [8192, 49, 49]);  expand_50 = None
        expand_51: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_94, [512, 16, 49, 32]);  getitem_94 = None
        clone_139: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        view_344: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_139, [8192, 49, 32]);  clone_139 = None
        bmm_25: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_343, view_344);  view_343 = None
        view_345: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_25, [512, 16, 49, 32]);  bmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_130: "f32[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_345, [0, 2, 1, 3]);  view_345 = None
        clone_140: "f32[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_130, memory_format = torch.contiguous_format);  permute_130 = None
        view_346: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(clone_140, [512, 49, 512]);  clone_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_347: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_346, [25088, 512]);  view_346 = None
        permute_131: "f32[512, 512]" = torch.ops.aten.permute.default(primals_192, [1, 0])
        addmm_49: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_193, view_347, permute_131);  primals_193 = permute_131 = None
        view_348: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(addmm_49, [512, 49, 512]);  addmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_349: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_348, [-1, 7, 7, 512]);  view_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_350: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_349, [-1, 2, 2, 7, 7, 512]);  view_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_132: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_350, [0, 1, 3, 2, 4, 5]);  view_350 = None
        clone_142: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_132, memory_format = torch.contiguous_format);  permute_132 = None
        view_351: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_142, [-1, 14, 14, 512]);  clone_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_22: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 22)
        inductor_random_default_23: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_22, 'rand');  inductor_lookup_seed_default_22 = None
        lt_22: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_23, 0.947826087474823);  inductor_random_default_23 = None
        convert_element_type_22: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_22, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_35: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_22, 0.947826087474823);  convert_element_type_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_127: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_351, div_35);  view_351 = div_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_135: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_331, mul_127);  view_331 = mul_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_352: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_135, [128, -1, 512]);  add_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_28 = torch.ops.aten.var_mean.correction(view_352, [2], correction = 0, keepdim = True)
        getitem_95: "f32[128, 196, 1]" = var_mean_28[0]
        getitem_96: "f32[128, 196, 1]" = var_mean_28[1];  var_mean_28 = None
        add_136: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_95, 1e-05);  getitem_95 = None
        rsqrt_28: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
        sub_41: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(view_352, getitem_96);  getitem_96 = None
        mul_128: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_28);  sub_41 = None
        mul_129: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_128, primals_194)
        add_137: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_129, primals_195);  mul_129 = primals_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_353: "f32[25088, 512]" = torch.ops.aten.reshape.default(add_137, [25088, 512]);  add_137 = None
        permute_133: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_196, [1, 0])
        addmm_50: "f32[25088, 2048]" = torch.ops.aten.addmm.default(primals_197, view_353, permute_133);  primals_197 = permute_133 = None
        view_354: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_50, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_130: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_354, 0.5)
        mul_131: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_354, 0.7071067811865476);  view_354 = None
        erf_12: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_131);  mul_131 = None
        add_138: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_132: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_130, add_138);  mul_130 = add_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_355: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_132, [25088, 2048]);  mul_132 = None
        permute_134: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_198, [1, 0])
        addmm_51: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_199, view_355, permute_134);  primals_199 = permute_134 = None
        view_356: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(addmm_51, [128, 196, 512]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_23: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 23)
        inductor_random_default_22: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_23, 'rand');  inductor_lookup_seed_default_23 = None
        lt_23: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_22, 0.947826087474823);  inductor_random_default_22 = None
        convert_element_type_23: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_23, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_36: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_23, 0.947826087474823);  convert_element_type_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_133: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_356, div_36);  view_356 = div_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_139: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_352, mul_133);  view_352 = mul_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_357: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_139, [128, 14, 14, 512]);  add_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_29 = torch.ops.aten.var_mean.correction(view_357, [3], correction = 0, keepdim = True)
        getitem_97: "f32[128, 14, 14, 1]" = var_mean_29[0]
        getitem_98: "f32[128, 14, 14, 1]" = var_mean_29[1];  var_mean_29 = None
        add_140: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_97, 1e-05);  getitem_97 = None
        rsqrt_29: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
        sub_42: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_357, getitem_98);  getitem_98 = None
        mul_134: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_29);  sub_42 = None
        mul_135: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_134, primals_200)
        add_141: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_135, primals_201);  mul_135 = primals_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_37: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(add_141, [None, fmod_8]);  add_141 = None
        index_38: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_37, [None, None, fmod_8]);  index_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_358: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(index_38, [128, 2, 7, 2, 7, 512]);  index_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_135: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_358, [0, 1, 3, 2, 4, 5]);  view_358 = None
        clone_145: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None
        view_359: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_145, [-1, 7, 7, 512]);  clone_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_360: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_359, [-1, 49, 512]);  view_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_361: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_360, [25088, 512]);  view_360 = None
        permute_136: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_203, [1, 0])
        addmm_52: "f32[25088, 1536]" = torch.ops.aten.addmm.default(primals_204, view_361, permute_136);  primals_204 = permute_136 = None
        view_362: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(addmm_52, [512, 49, 1536]);  addmm_52 = None
        view_363: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.reshape.default(view_362, [512, 49, 3, 16, -1]);  view_362 = None
        permute_137: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_363, [2, 0, 3, 1, 4]);  view_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_13 = torch.ops.aten.unbind.int(permute_137);  permute_137 = None
        getitem_99: "f32[512, 16, 49, 32]" = unbind_13[0]
        getitem_100: "f32[512, 16, 49, 32]" = unbind_13[1]
        getitem_101: "f32[512, 16, 49, 32]" = unbind_13[2];  unbind_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_136: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_99, 0.1767766952966369);  getitem_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_138: "f32[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_100, [0, 1, 3, 2]);  getitem_100 = None
        expand_52: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_136, [512, 16, 49, 32]);  mul_136 = None
        clone_146: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_364: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_146, [8192, 49, 32]);  clone_146 = None
        expand_53: "f32[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_138, [512, 16, 32, 49]);  permute_138 = None
        clone_147: "f32[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
        view_365: "f32[8192, 32, 49]" = torch.ops.aten.reshape.default(clone_147, [8192, 32, 49]);  clone_147 = None
        bmm_26: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_364, view_365)
        view_366: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_26, [512, 16, 49, 49]);  bmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_367: "i64[2401]" = torch.ops.aten.reshape.default(primals_206, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_39: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_205, [view_367]);  primals_205 = view_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_368: "f32[49, 49, 16]" = torch.ops.aten.reshape.default(index_39, [49, 49, -1]);  index_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_139: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_368, [2, 0, 1]);  view_368 = None
        clone_148: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_25: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_148, 0);  clone_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_144: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_366, unsqueeze_25);  view_366 = unsqueeze_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_369: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.reshape.default(add_144, [-1, 4, 16, 49, 49]);  add_144 = None
        unsqueeze_26: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(primals_202, 1);  primals_202 = None
        unsqueeze_27: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, 0);  unsqueeze_26 = None
        add_145: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_369, unsqueeze_27);  view_369 = unsqueeze_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_370: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(add_145, [-1, 16, 49, 49]);  add_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_13: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(view_370, [-1], True)
        sub_43: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_370, amax_13);  view_370 = amax_13 = None
        exp_13: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
        sum_14: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_37: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_54: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_37, [512, 16, 49, 49])
        view_371: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_54, [8192, 49, 49]);  expand_54 = None
        expand_55: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_101, [512, 16, 49, 32]);  getitem_101 = None
        clone_150: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
        view_372: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_150, [8192, 49, 32]);  clone_150 = None
        bmm_27: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_371, view_372);  view_371 = None
        view_373: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_27, [512, 16, 49, 32]);  bmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_140: "f32[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_373, [0, 2, 1, 3]);  view_373 = None
        clone_151: "f32[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format);  permute_140 = None
        view_374: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(clone_151, [512, 49, 512]);  clone_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_375: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_374, [25088, 512]);  view_374 = None
        permute_141: "f32[512, 512]" = torch.ops.aten.permute.default(primals_207, [1, 0])
        addmm_53: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_208, view_375, permute_141);  primals_208 = permute_141 = None
        view_376: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(addmm_53, [512, 49, 512]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_377: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_376, [-1, 7, 7, 512]);  view_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_378: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_377, [-1, 2, 2, 7, 7, 512]);  view_377 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_142: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_378, [0, 1, 3, 2, 4, 5]);  view_378 = None
        clone_153: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_142, memory_format = torch.contiguous_format);  permute_142 = None
        view_379: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_153, [-1, 14, 14, 512]);  clone_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_40: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_379, [None, fmod_10]);  view_379 = None
        index_41: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_40, [None, None, fmod_10]);  index_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_24: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 24)
        inductor_random_default_21: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_24, 'rand');  inductor_lookup_seed_default_24 = None
        lt_24: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_21, 0.9434782639145851);  inductor_random_default_21 = None
        convert_element_type_24: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_24, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_38: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_24, 0.9434782639145851);  convert_element_type_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_137: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_41, div_38);  index_41 = div_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_148: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_357, mul_137);  view_357 = mul_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_380: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_148, [128, -1, 512]);  add_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_30 = torch.ops.aten.var_mean.correction(view_380, [2], correction = 0, keepdim = True)
        getitem_102: "f32[128, 196, 1]" = var_mean_30[0]
        getitem_103: "f32[128, 196, 1]" = var_mean_30[1];  var_mean_30 = None
        add_149: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05);  getitem_102 = None
        rsqrt_30: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
        sub_44: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(view_380, getitem_103);  getitem_103 = None
        mul_138: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_30);  sub_44 = None
        mul_139: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_138, primals_209)
        add_150: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_139, primals_210);  mul_139 = primals_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_381: "f32[25088, 512]" = torch.ops.aten.reshape.default(add_150, [25088, 512]);  add_150 = None
        permute_143: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_211, [1, 0])
        addmm_54: "f32[25088, 2048]" = torch.ops.aten.addmm.default(primals_212, view_381, permute_143);  primals_212 = permute_143 = None
        view_382: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_54, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_140: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_382, 0.5)
        mul_141: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_382, 0.7071067811865476);  view_382 = None
        erf_13: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_141);  mul_141 = None
        add_151: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_142: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_140, add_151);  mul_140 = add_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_383: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_142, [25088, 2048]);  mul_142 = None
        permute_144: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_213, [1, 0])
        addmm_55: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_214, view_383, permute_144);  primals_214 = permute_144 = None
        view_384: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(addmm_55, [128, 196, 512]);  addmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_25: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 25)
        inductor_random_default_20: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_25, 'rand');  inductor_lookup_seed_default_25 = None
        lt_25: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_20, 0.9434782639145851);  inductor_random_default_20 = None
        convert_element_type_25: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_25, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_39: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_25, 0.9434782639145851);  convert_element_type_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_143: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_384, div_39);  view_384 = div_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_152: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_380, mul_143);  view_380 = mul_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_385: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_152, [128, 14, 14, 512]);  add_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_31 = torch.ops.aten.var_mean.correction(view_385, [3], correction = 0, keepdim = True)
        getitem_104: "f32[128, 14, 14, 1]" = var_mean_31[0]
        getitem_105: "f32[128, 14, 14, 1]" = var_mean_31[1];  var_mean_31 = None
        add_153: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
        rsqrt_31: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
        sub_45: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_385, getitem_105);  getitem_105 = None
        mul_144: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_31);  sub_45 = None
        mul_145: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_144, primals_215)
        add_154: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_145, primals_216);  mul_145 = primals_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_386: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(add_154, [128, 2, 7, 2, 7, 512]);  add_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_145: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_386, [0, 1, 3, 2, 4, 5]);  view_386 = None
        clone_156: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_145, memory_format = torch.contiguous_format);  permute_145 = None
        view_387: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_156, [-1, 7, 7, 512]);  clone_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_388: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_387, [-1, 49, 512]);  view_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_389: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_388, [25088, 512]);  view_388 = None
        permute_146: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_217, [1, 0])
        addmm_56: "f32[25088, 1536]" = torch.ops.aten.addmm.default(primals_218, view_389, permute_146);  primals_218 = permute_146 = None
        view_390: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(addmm_56, [512, 49, 1536]);  addmm_56 = None
        view_391: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.reshape.default(view_390, [512, 49, 3, 16, -1]);  view_390 = None
        permute_147: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_391, [2, 0, 3, 1, 4]);  view_391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_14 = torch.ops.aten.unbind.int(permute_147);  permute_147 = None
        getitem_106: "f32[512, 16, 49, 32]" = unbind_14[0]
        getitem_107: "f32[512, 16, 49, 32]" = unbind_14[1]
        getitem_108: "f32[512, 16, 49, 32]" = unbind_14[2];  unbind_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_146: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_106, 0.1767766952966369);  getitem_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_148: "f32[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_107, [0, 1, 3, 2]);  getitem_107 = None
        expand_56: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_146, [512, 16, 49, 32]);  mul_146 = None
        clone_157: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_392: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_157, [8192, 49, 32]);  clone_157 = None
        expand_57: "f32[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_148, [512, 16, 32, 49]);  permute_148 = None
        clone_158: "f32[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_393: "f32[8192, 32, 49]" = torch.ops.aten.reshape.default(clone_158, [8192, 32, 49]);  clone_158 = None
        bmm_28: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_392, view_393)
        view_394: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_28, [512, 16, 49, 49]);  bmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_395: "i64[2401]" = torch.ops.aten.reshape.default(primals_220, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_42: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_219, [view_395]);  primals_219 = view_395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_396: "f32[49, 49, 16]" = torch.ops.aten.reshape.default(index_42, [49, 49, -1]);  index_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_149: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_396, [2, 0, 1]);  view_396 = None
        clone_159: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_149, memory_format = torch.contiguous_format);  permute_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_28: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_159, 0);  clone_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_155: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_394, unsqueeze_28);  view_394 = unsqueeze_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_14: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(add_155, [-1], True)
        sub_46: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_155, amax_14);  add_155 = amax_14 = None
        exp_14: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
        sum_15: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_40: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_58: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_40, [512, 16, 49, 49])
        view_397: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_58, [8192, 49, 49]);  expand_58 = None
        expand_59: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_108, [512, 16, 49, 32]);  getitem_108 = None
        clone_161: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
        view_398: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_161, [8192, 49, 32]);  clone_161 = None
        bmm_29: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_397, view_398);  view_397 = None
        view_399: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_29, [512, 16, 49, 32]);  bmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_150: "f32[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_399, [0, 2, 1, 3]);  view_399 = None
        clone_162: "f32[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
        view_400: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(clone_162, [512, 49, 512]);  clone_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_401: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_400, [25088, 512]);  view_400 = None
        permute_151: "f32[512, 512]" = torch.ops.aten.permute.default(primals_221, [1, 0])
        addmm_57: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_222, view_401, permute_151);  primals_222 = permute_151 = None
        view_402: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(addmm_57, [512, 49, 512]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_403: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_402, [-1, 7, 7, 512]);  view_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_404: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_403, [-1, 2, 2, 7, 7, 512]);  view_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_152: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_404, [0, 1, 3, 2, 4, 5]);  view_404 = None
        clone_164: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_152, memory_format = torch.contiguous_format);  permute_152 = None
        view_405: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_164, [-1, 14, 14, 512]);  clone_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_26: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 26)
        inductor_random_default_19: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_26, 'rand');  inductor_lookup_seed_default_26 = None
        lt_26: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_19, 0.9391304366290569);  inductor_random_default_19 = None
        convert_element_type_26: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_26, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_41: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_26, 0.9391304366290569);  convert_element_type_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_147: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_405, div_41);  view_405 = div_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_156: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_385, mul_147);  view_385 = mul_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_406: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_156, [128, -1, 512]);  add_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_32 = torch.ops.aten.var_mean.correction(view_406, [2], correction = 0, keepdim = True)
        getitem_109: "f32[128, 196, 1]" = var_mean_32[0]
        getitem_110: "f32[128, 196, 1]" = var_mean_32[1];  var_mean_32 = None
        add_157: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_109, 1e-05);  getitem_109 = None
        rsqrt_32: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
        sub_47: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(view_406, getitem_110);  getitem_110 = None
        mul_148: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_32);  sub_47 = None
        mul_149: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_148, primals_223)
        add_158: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_149, primals_224);  mul_149 = primals_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_407: "f32[25088, 512]" = torch.ops.aten.reshape.default(add_158, [25088, 512]);  add_158 = None
        permute_153: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_225, [1, 0])
        addmm_58: "f32[25088, 2048]" = torch.ops.aten.addmm.default(primals_226, view_407, permute_153);  primals_226 = permute_153 = None
        view_408: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_58, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_150: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_408, 0.5)
        mul_151: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_408, 0.7071067811865476);  view_408 = None
        erf_14: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_151);  mul_151 = None
        add_159: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_152: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_150, add_159);  mul_150 = add_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_409: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_152, [25088, 2048]);  mul_152 = None
        permute_154: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_227, [1, 0])
        addmm_59: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_228, view_409, permute_154);  primals_228 = permute_154 = None
        view_410: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(addmm_59, [128, 196, 512]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_27: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 27)
        inductor_random_default_18: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_27, 'rand');  inductor_lookup_seed_default_27 = None
        lt_27: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_18, 0.9391304366290569);  inductor_random_default_18 = None
        convert_element_type_27: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_27, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_42: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_27, 0.9391304366290569);  convert_element_type_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_153: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_410, div_42);  view_410 = div_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_160: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_406, mul_153);  view_406 = mul_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_411: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_160, [128, 14, 14, 512]);  add_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_33 = torch.ops.aten.var_mean.correction(view_411, [3], correction = 0, keepdim = True)
        getitem_111: "f32[128, 14, 14, 1]" = var_mean_33[0]
        getitem_112: "f32[128, 14, 14, 1]" = var_mean_33[1];  var_mean_33 = None
        add_161: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_111, 1e-05);  getitem_111 = None
        rsqrt_33: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
        sub_48: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_411, getitem_112);  getitem_112 = None
        mul_154: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_33);  sub_48 = None
        mul_155: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_154, primals_229)
        add_162: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_155, primals_230);  mul_155 = primals_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_43: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(add_162, [None, fmod_8]);  add_162 = None
        index_44: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_43, [None, None, fmod_8]);  index_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_412: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(index_44, [128, 2, 7, 2, 7, 512]);  index_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_155: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_412, [0, 1, 3, 2, 4, 5]);  view_412 = None
        clone_167: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
        view_413: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_167, [-1, 7, 7, 512]);  clone_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_414: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_413, [-1, 49, 512]);  view_413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_415: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_414, [25088, 512]);  view_414 = None
        permute_156: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_232, [1, 0])
        addmm_60: "f32[25088, 1536]" = torch.ops.aten.addmm.default(primals_233, view_415, permute_156);  primals_233 = permute_156 = None
        view_416: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(addmm_60, [512, 49, 1536]);  addmm_60 = None
        view_417: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.reshape.default(view_416, [512, 49, 3, 16, -1]);  view_416 = None
        permute_157: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_417, [2, 0, 3, 1, 4]);  view_417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_15 = torch.ops.aten.unbind.int(permute_157);  permute_157 = None
        getitem_113: "f32[512, 16, 49, 32]" = unbind_15[0]
        getitem_114: "f32[512, 16, 49, 32]" = unbind_15[1]
        getitem_115: "f32[512, 16, 49, 32]" = unbind_15[2];  unbind_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_156: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_113, 0.1767766952966369);  getitem_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_158: "f32[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_114, [0, 1, 3, 2]);  getitem_114 = None
        expand_60: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_156, [512, 16, 49, 32]);  mul_156 = None
        clone_168: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_418: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_168, [8192, 49, 32]);  clone_168 = None
        expand_61: "f32[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_158, [512, 16, 32, 49]);  permute_158 = None
        clone_169: "f32[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
        view_419: "f32[8192, 32, 49]" = torch.ops.aten.reshape.default(clone_169, [8192, 32, 49]);  clone_169 = None
        bmm_30: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_418, view_419)
        view_420: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_30, [512, 16, 49, 49]);  bmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_421: "i64[2401]" = torch.ops.aten.reshape.default(primals_235, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_45: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_234, [view_421]);  primals_234 = view_421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_422: "f32[49, 49, 16]" = torch.ops.aten.reshape.default(index_45, [49, 49, -1]);  index_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_159: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_422, [2, 0, 1]);  view_422 = None
        clone_170: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_29: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_170, 0);  clone_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_165: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_420, unsqueeze_29);  view_420 = unsqueeze_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_423: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.reshape.default(add_165, [-1, 4, 16, 49, 49]);  add_165 = None
        unsqueeze_30: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(primals_231, 1);  primals_231 = None
        unsqueeze_31: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, 0);  unsqueeze_30 = None
        add_166: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_423, unsqueeze_31);  view_423 = unsqueeze_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_424: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(add_166, [-1, 16, 49, 49]);  add_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_15: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(view_424, [-1], True)
        sub_49: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_424, amax_15);  view_424 = amax_15 = None
        exp_15: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
        sum_16: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_43: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_62: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_43, [512, 16, 49, 49])
        view_425: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_62, [8192, 49, 49]);  expand_62 = None
        expand_63: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_115, [512, 16, 49, 32]);  getitem_115 = None
        clone_172: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
        view_426: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_172, [8192, 49, 32]);  clone_172 = None
        bmm_31: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_425, view_426);  view_425 = None
        view_427: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_31, [512, 16, 49, 32]);  bmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_160: "f32[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_427, [0, 2, 1, 3]);  view_427 = None
        clone_173: "f32[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
        view_428: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(clone_173, [512, 49, 512]);  clone_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_429: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_428, [25088, 512]);  view_428 = None
        permute_161: "f32[512, 512]" = torch.ops.aten.permute.default(primals_236, [1, 0])
        addmm_61: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_237, view_429, permute_161);  primals_237 = permute_161 = None
        view_430: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(addmm_61, [512, 49, 512]);  addmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_431: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_430, [-1, 7, 7, 512]);  view_430 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_432: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_431, [-1, 2, 2, 7, 7, 512]);  view_431 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_162: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_432, [0, 1, 3, 2, 4, 5]);  view_432 = None
        clone_175: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_162, memory_format = torch.contiguous_format);  permute_162 = None
        view_433: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_175, [-1, 14, 14, 512]);  clone_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_46: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_433, [None, fmod_10]);  view_433 = None
        index_47: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_46, [None, None, fmod_10]);  index_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_28: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 28)
        inductor_random_default_17: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_28, 'rand');  inductor_lookup_seed_default_28 = None
        lt_28: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_17, 0.9347826093435287);  inductor_random_default_17 = None
        convert_element_type_28: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_28, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_44: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_28, 0.9347826093435287);  convert_element_type_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_157: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_47, div_44);  index_47 = div_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_169: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_411, mul_157);  view_411 = mul_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_434: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_169, [128, -1, 512]);  add_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_34 = torch.ops.aten.var_mean.correction(view_434, [2], correction = 0, keepdim = True)
        getitem_116: "f32[128, 196, 1]" = var_mean_34[0]
        getitem_117: "f32[128, 196, 1]" = var_mean_34[1];  var_mean_34 = None
        add_170: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-05);  getitem_116 = None
        rsqrt_34: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        sub_50: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(view_434, getitem_117);  getitem_117 = None
        mul_158: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_34);  sub_50 = None
        mul_159: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_158, primals_238)
        add_171: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_159, primals_239);  mul_159 = primals_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_435: "f32[25088, 512]" = torch.ops.aten.reshape.default(add_171, [25088, 512]);  add_171 = None
        permute_163: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_240, [1, 0])
        addmm_62: "f32[25088, 2048]" = torch.ops.aten.addmm.default(primals_241, view_435, permute_163);  primals_241 = permute_163 = None
        view_436: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_62, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_160: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_436, 0.5)
        mul_161: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_436, 0.7071067811865476);  view_436 = None
        erf_15: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_161);  mul_161 = None
        add_172: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_162: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_160, add_172);  mul_160 = add_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_437: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_162, [25088, 2048]);  mul_162 = None
        permute_164: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_242, [1, 0])
        addmm_63: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_243, view_437, permute_164);  primals_243 = permute_164 = None
        view_438: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(addmm_63, [128, 196, 512]);  addmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_29: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 29)
        inductor_random_default_16: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_29, 'rand');  inductor_lookup_seed_default_29 = None
        lt_29: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_16, 0.9347826093435287);  inductor_random_default_16 = None
        convert_element_type_29: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_29, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_45: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_29, 0.9347826093435287);  convert_element_type_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_163: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_438, div_45);  view_438 = div_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_173: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_434, mul_163);  view_434 = mul_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_439: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_173, [128, 14, 14, 512]);  add_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_35 = torch.ops.aten.var_mean.correction(view_439, [3], correction = 0, keepdim = True)
        getitem_118: "f32[128, 14, 14, 1]" = var_mean_35[0]
        getitem_119: "f32[128, 14, 14, 1]" = var_mean_35[1];  var_mean_35 = None
        add_174: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05);  getitem_118 = None
        rsqrt_35: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        sub_51: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_439, getitem_119);  getitem_119 = None
        mul_164: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_35);  sub_51 = None
        mul_165: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_164, primals_244)
        add_175: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_165, primals_245);  mul_165 = primals_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_440: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(add_175, [128, 2, 7, 2, 7, 512]);  add_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_165: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_440, [0, 1, 3, 2, 4, 5]);  view_440 = None
        clone_178: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_165, memory_format = torch.contiguous_format);  permute_165 = None
        view_441: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_178, [-1, 7, 7, 512]);  clone_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_442: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_441, [-1, 49, 512]);  view_441 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_443: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_442, [25088, 512]);  view_442 = None
        permute_166: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_246, [1, 0])
        addmm_64: "f32[25088, 1536]" = torch.ops.aten.addmm.default(primals_247, view_443, permute_166);  primals_247 = permute_166 = None
        view_444: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(addmm_64, [512, 49, 1536]);  addmm_64 = None
        view_445: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.reshape.default(view_444, [512, 49, 3, 16, -1]);  view_444 = None
        permute_167: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_445, [2, 0, 3, 1, 4]);  view_445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_16 = torch.ops.aten.unbind.int(permute_167);  permute_167 = None
        getitem_120: "f32[512, 16, 49, 32]" = unbind_16[0]
        getitem_121: "f32[512, 16, 49, 32]" = unbind_16[1]
        getitem_122: "f32[512, 16, 49, 32]" = unbind_16[2];  unbind_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_166: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_120, 0.1767766952966369);  getitem_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_168: "f32[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_121, [0, 1, 3, 2]);  getitem_121 = None
        expand_64: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_166, [512, 16, 49, 32]);  mul_166 = None
        clone_179: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
        view_446: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_179, [8192, 49, 32]);  clone_179 = None
        expand_65: "f32[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_168, [512, 16, 32, 49]);  permute_168 = None
        clone_180: "f32[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        view_447: "f32[8192, 32, 49]" = torch.ops.aten.reshape.default(clone_180, [8192, 32, 49]);  clone_180 = None
        bmm_32: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_446, view_447)
        view_448: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_32, [512, 16, 49, 49]);  bmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_449: "i64[2401]" = torch.ops.aten.reshape.default(primals_249, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_48: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_248, [view_449]);  primals_248 = view_449 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_450: "f32[49, 49, 16]" = torch.ops.aten.reshape.default(index_48, [49, 49, -1]);  index_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_169: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_450, [2, 0, 1]);  view_450 = None
        clone_181: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_169, memory_format = torch.contiguous_format);  permute_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_32: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_181, 0);  clone_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_176: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_448, unsqueeze_32);  view_448 = unsqueeze_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_16: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(add_176, [-1], True)
        sub_52: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_176, amax_16);  add_176 = amax_16 = None
        exp_16: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
        sum_17: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_46: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_66: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_46, [512, 16, 49, 49])
        view_451: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_66, [8192, 49, 49]);  expand_66 = None
        expand_67: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_122, [512, 16, 49, 32]);  getitem_122 = None
        clone_183: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
        view_452: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_183, [8192, 49, 32]);  clone_183 = None
        bmm_33: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_451, view_452);  view_451 = None
        view_453: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_33, [512, 16, 49, 32]);  bmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_170: "f32[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_453, [0, 2, 1, 3]);  view_453 = None
        clone_184: "f32[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
        view_454: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(clone_184, [512, 49, 512]);  clone_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_455: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_454, [25088, 512]);  view_454 = None
        permute_171: "f32[512, 512]" = torch.ops.aten.permute.default(primals_250, [1, 0])
        addmm_65: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_251, view_455, permute_171);  primals_251 = permute_171 = None
        view_456: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(addmm_65, [512, 49, 512]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_457: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_456, [-1, 7, 7, 512]);  view_456 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_458: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_457, [-1, 2, 2, 7, 7, 512]);  view_457 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_172: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_458, [0, 1, 3, 2, 4, 5]);  view_458 = None
        clone_186: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
        view_459: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_186, [-1, 14, 14, 512]);  clone_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_30: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 30)
        inductor_random_default_15: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_30, 'rand');  inductor_lookup_seed_default_30 = None
        lt_30: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_15, 0.9304347857832909);  inductor_random_default_15 = None
        convert_element_type_30: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_30, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_47: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_30, 0.9304347857832909);  convert_element_type_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_167: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_459, div_47);  view_459 = div_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_177: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_439, mul_167);  view_439 = mul_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_460: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_177, [128, -1, 512]);  add_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_36 = torch.ops.aten.var_mean.correction(view_460, [2], correction = 0, keepdim = True)
        getitem_123: "f32[128, 196, 1]" = var_mean_36[0]
        getitem_124: "f32[128, 196, 1]" = var_mean_36[1];  var_mean_36 = None
        add_178: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_123, 1e-05);  getitem_123 = None
        rsqrt_36: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
        sub_53: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(view_460, getitem_124);  getitem_124 = None
        mul_168: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_36);  sub_53 = None
        mul_169: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_168, primals_252)
        add_179: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_169, primals_253);  mul_169 = primals_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_461: "f32[25088, 512]" = torch.ops.aten.reshape.default(add_179, [25088, 512]);  add_179 = None
        permute_173: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_254, [1, 0])
        addmm_66: "f32[25088, 2048]" = torch.ops.aten.addmm.default(primals_255, view_461, permute_173);  primals_255 = permute_173 = None
        view_462: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_66, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_170: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_462, 0.5)
        mul_171: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_462, 0.7071067811865476);  view_462 = None
        erf_16: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_171);  mul_171 = None
        add_180: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_172: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_170, add_180);  mul_170 = add_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_463: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_172, [25088, 2048]);  mul_172 = None
        permute_174: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_256, [1, 0])
        addmm_67: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_257, view_463, permute_174);  primals_257 = permute_174 = None
        view_464: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(addmm_67, [128, 196, 512]);  addmm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_31: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 31)
        inductor_random_default_14: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_31, 'rand');  inductor_lookup_seed_default_31 = None
        lt_31: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_14, 0.9304347857832909);  inductor_random_default_14 = None
        convert_element_type_31: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_31, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_48: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_31, 0.9304347857832909);  convert_element_type_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_173: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_464, div_48);  view_464 = div_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_181: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_460, mul_173);  view_460 = mul_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_465: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_181, [128, 14, 14, 512]);  add_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_37 = torch.ops.aten.var_mean.correction(view_465, [3], correction = 0, keepdim = True)
        getitem_125: "f32[128, 14, 14, 1]" = var_mean_37[0]
        getitem_126: "f32[128, 14, 14, 1]" = var_mean_37[1];  var_mean_37 = None
        add_182: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_125, 1e-05);  getitem_125 = None
        rsqrt_37: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
        sub_54: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_465, getitem_126);  getitem_126 = None
        mul_174: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_37);  sub_54 = None
        mul_175: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_174, primals_258)
        add_183: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_175, primals_259);  mul_175 = primals_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_49: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(add_183, [None, fmod_8]);  add_183 = None
        index_50: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_49, [None, None, fmod_8]);  index_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_466: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(index_50, [128, 2, 7, 2, 7, 512]);  index_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_175: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_466, [0, 1, 3, 2, 4, 5]);  view_466 = None
        clone_189: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_175, memory_format = torch.contiguous_format);  permute_175 = None
        view_467: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_189, [-1, 7, 7, 512]);  clone_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_468: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_467, [-1, 49, 512]);  view_467 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_469: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_468, [25088, 512]);  view_468 = None
        permute_176: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_261, [1, 0])
        addmm_68: "f32[25088, 1536]" = torch.ops.aten.addmm.default(primals_262, view_469, permute_176);  primals_262 = permute_176 = None
        view_470: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(addmm_68, [512, 49, 1536]);  addmm_68 = None
        view_471: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.reshape.default(view_470, [512, 49, 3, 16, -1]);  view_470 = None
        permute_177: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_471, [2, 0, 3, 1, 4]);  view_471 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_17 = torch.ops.aten.unbind.int(permute_177);  permute_177 = None
        getitem_127: "f32[512, 16, 49, 32]" = unbind_17[0]
        getitem_128: "f32[512, 16, 49, 32]" = unbind_17[1]
        getitem_129: "f32[512, 16, 49, 32]" = unbind_17[2];  unbind_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_176: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_127, 0.1767766952966369);  getitem_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_178: "f32[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_128, [0, 1, 3, 2]);  getitem_128 = None
        expand_68: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_176, [512, 16, 49, 32]);  mul_176 = None
        clone_190: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
        view_472: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_190, [8192, 49, 32]);  clone_190 = None
        expand_69: "f32[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_178, [512, 16, 32, 49]);  permute_178 = None
        clone_191: "f32[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
        view_473: "f32[8192, 32, 49]" = torch.ops.aten.reshape.default(clone_191, [8192, 32, 49]);  clone_191 = None
        bmm_34: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_472, view_473)
        view_474: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_34, [512, 16, 49, 49]);  bmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_475: "i64[2401]" = torch.ops.aten.reshape.default(primals_264, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_51: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_263, [view_475]);  primals_263 = view_475 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_476: "f32[49, 49, 16]" = torch.ops.aten.reshape.default(index_51, [49, 49, -1]);  index_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_179: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_476, [2, 0, 1]);  view_476 = None
        clone_192: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_33: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_192, 0);  clone_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_186: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_474, unsqueeze_33);  view_474 = unsqueeze_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_477: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.reshape.default(add_186, [-1, 4, 16, 49, 49]);  add_186 = None
        unsqueeze_34: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(primals_260, 1);  primals_260 = None
        unsqueeze_35: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, 0);  unsqueeze_34 = None
        add_187: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_477, unsqueeze_35);  view_477 = unsqueeze_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_478: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(add_187, [-1, 16, 49, 49]);  add_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_17: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(view_478, [-1], True)
        sub_55: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_478, amax_17);  view_478 = amax_17 = None
        exp_17: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
        sum_18: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_49: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_70: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_49, [512, 16, 49, 49])
        view_479: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_70, [8192, 49, 49]);  expand_70 = None
        expand_71: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_129, [512, 16, 49, 32]);  getitem_129 = None
        clone_194: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
        view_480: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_194, [8192, 49, 32]);  clone_194 = None
        bmm_35: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_479, view_480);  view_479 = None
        view_481: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_35, [512, 16, 49, 32]);  bmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_180: "f32[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_481, [0, 2, 1, 3]);  view_481 = None
        clone_195: "f32[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_180, memory_format = torch.contiguous_format);  permute_180 = None
        view_482: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(clone_195, [512, 49, 512]);  clone_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_483: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_482, [25088, 512]);  view_482 = None
        permute_181: "f32[512, 512]" = torch.ops.aten.permute.default(primals_265, [1, 0])
        addmm_69: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_266, view_483, permute_181);  primals_266 = permute_181 = None
        view_484: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(addmm_69, [512, 49, 512]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_485: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_484, [-1, 7, 7, 512]);  view_484 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_486: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_485, [-1, 2, 2, 7, 7, 512]);  view_485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_182: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_486, [0, 1, 3, 2, 4, 5]);  view_486 = None
        clone_197: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_182, memory_format = torch.contiguous_format);  permute_182 = None
        view_487: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_197, [-1, 14, 14, 512]);  clone_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_52: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_487, [None, fmod_10]);  view_487 = None
        index_53: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_52, [None, None, fmod_10]);  index_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_32: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 32)
        inductor_random_default_13: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_32, 'rand');  inductor_lookup_seed_default_32 = None
        lt_32: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_13, 0.9260869547724724);  inductor_random_default_13 = None
        convert_element_type_32: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_32, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_50: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_32, 0.9260869547724724);  convert_element_type_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_177: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_53, div_50);  index_53 = div_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_190: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_465, mul_177);  view_465 = mul_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_488: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_190, [128, -1, 512]);  add_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_38 = torch.ops.aten.var_mean.correction(view_488, [2], correction = 0, keepdim = True)
        getitem_130: "f32[128, 196, 1]" = var_mean_38[0]
        getitem_131: "f32[128, 196, 1]" = var_mean_38[1];  var_mean_38 = None
        add_191: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_38: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
        sub_56: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(view_488, getitem_131);  getitem_131 = None
        mul_178: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_38);  sub_56 = None
        mul_179: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_178, primals_267)
        add_192: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_179, primals_268);  mul_179 = primals_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_489: "f32[25088, 512]" = torch.ops.aten.reshape.default(add_192, [25088, 512]);  add_192 = None
        permute_183: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_269, [1, 0])
        addmm_70: "f32[25088, 2048]" = torch.ops.aten.addmm.default(primals_270, view_489, permute_183);  primals_270 = permute_183 = None
        view_490: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_70, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_180: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_490, 0.5)
        mul_181: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_490, 0.7071067811865476);  view_490 = None
        erf_17: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_181);  mul_181 = None
        add_193: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_182: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_180, add_193);  mul_180 = add_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_491: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_182, [25088, 2048]);  mul_182 = None
        permute_184: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_271, [1, 0])
        addmm_71: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_272, view_491, permute_184);  primals_272 = permute_184 = None
        view_492: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(addmm_71, [128, 196, 512]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_33: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 33)
        inductor_random_default_12: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_33, 'rand');  inductor_lookup_seed_default_33 = None
        lt_33: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_12, 0.9260869547724724);  inductor_random_default_12 = None
        convert_element_type_33: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_33, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_51: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_33, 0.9260869547724724);  convert_element_type_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_183: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_492, div_51);  view_492 = div_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_194: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_488, mul_183);  view_488 = mul_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_493: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_194, [128, 14, 14, 512]);  add_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_39 = torch.ops.aten.var_mean.correction(view_493, [3], correction = 0, keepdim = True)
        getitem_132: "f32[128, 14, 14, 1]" = var_mean_39[0]
        getitem_133: "f32[128, 14, 14, 1]" = var_mean_39[1];  var_mean_39 = None
        add_195: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-05);  getitem_132 = None
        rsqrt_39: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_195);  add_195 = None
        sub_57: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_493, getitem_133);  getitem_133 = None
        mul_184: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_39);  sub_57 = None
        mul_185: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_184, primals_273)
        add_196: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_185, primals_274);  mul_185 = primals_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_494: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(add_196, [128, 2, 7, 2, 7, 512]);  add_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_185: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_494, [0, 1, 3, 2, 4, 5]);  view_494 = None
        clone_200: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_185, memory_format = torch.contiguous_format);  permute_185 = None
        view_495: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_200, [-1, 7, 7, 512]);  clone_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_496: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_495, [-1, 49, 512]);  view_495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_497: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_496, [25088, 512]);  view_496 = None
        permute_186: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_275, [1, 0])
        addmm_72: "f32[25088, 1536]" = torch.ops.aten.addmm.default(primals_276, view_497, permute_186);  primals_276 = permute_186 = None
        view_498: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(addmm_72, [512, 49, 1536]);  addmm_72 = None
        view_499: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.reshape.default(view_498, [512, 49, 3, 16, -1]);  view_498 = None
        permute_187: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_499, [2, 0, 3, 1, 4]);  view_499 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_18 = torch.ops.aten.unbind.int(permute_187);  permute_187 = None
        getitem_134: "f32[512, 16, 49, 32]" = unbind_18[0]
        getitem_135: "f32[512, 16, 49, 32]" = unbind_18[1]
        getitem_136: "f32[512, 16, 49, 32]" = unbind_18[2];  unbind_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_186: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_134, 0.1767766952966369);  getitem_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_188: "f32[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_135, [0, 1, 3, 2]);  getitem_135 = None
        expand_72: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_186, [512, 16, 49, 32]);  mul_186 = None
        clone_201: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
        view_500: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_201, [8192, 49, 32]);  clone_201 = None
        expand_73: "f32[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_188, [512, 16, 32, 49]);  permute_188 = None
        clone_202: "f32[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
        view_501: "f32[8192, 32, 49]" = torch.ops.aten.reshape.default(clone_202, [8192, 32, 49]);  clone_202 = None
        bmm_36: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_500, view_501)
        view_502: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_36, [512, 16, 49, 49]);  bmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_503: "i64[2401]" = torch.ops.aten.reshape.default(primals_278, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_54: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_277, [view_503]);  primals_277 = view_503 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_504: "f32[49, 49, 16]" = torch.ops.aten.reshape.default(index_54, [49, 49, -1]);  index_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_189: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_504, [2, 0, 1]);  view_504 = None
        clone_203: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_36: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_203, 0);  clone_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_197: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_502, unsqueeze_36);  view_502 = unsqueeze_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_18: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(add_197, [-1], True)
        sub_58: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_197, amax_18);  add_197 = amax_18 = None
        exp_18: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_58);  sub_58 = None
        sum_19: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_52: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_74: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_52, [512, 16, 49, 49])
        view_505: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_74, [8192, 49, 49]);  expand_74 = None
        expand_75: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_136, [512, 16, 49, 32]);  getitem_136 = None
        clone_205: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
        view_506: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_205, [8192, 49, 32]);  clone_205 = None
        bmm_37: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_505, view_506);  view_505 = None
        view_507: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_37, [512, 16, 49, 32]);  bmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_190: "f32[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_507, [0, 2, 1, 3]);  view_507 = None
        clone_206: "f32[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_190, memory_format = torch.contiguous_format);  permute_190 = None
        view_508: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(clone_206, [512, 49, 512]);  clone_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_509: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_508, [25088, 512]);  view_508 = None
        permute_191: "f32[512, 512]" = torch.ops.aten.permute.default(primals_279, [1, 0])
        addmm_73: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_280, view_509, permute_191);  primals_280 = permute_191 = None
        view_510: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(addmm_73, [512, 49, 512]);  addmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_511: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_510, [-1, 7, 7, 512]);  view_510 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_512: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_511, [-1, 2, 2, 7, 7, 512]);  view_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_192: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_512, [0, 1, 3, 2, 4, 5]);  view_512 = None
        clone_208: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_192, memory_format = torch.contiguous_format);  permute_192 = None
        view_513: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_208, [-1, 14, 14, 512]);  clone_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_34: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 34)
        inductor_random_default_11: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_34, 'rand');  inductor_lookup_seed_default_34 = None
        lt_34: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_11, 0.9217391312122345);  inductor_random_default_11 = None
        convert_element_type_34: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_34, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_53: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_34, 0.9217391312122345);  convert_element_type_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_187: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_513, div_53);  view_513 = div_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_198: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_493, mul_187);  view_493 = mul_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_514: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_198, [128, -1, 512]);  add_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_40 = torch.ops.aten.var_mean.correction(view_514, [2], correction = 0, keepdim = True)
        getitem_137: "f32[128, 196, 1]" = var_mean_40[0]
        getitem_138: "f32[128, 196, 1]" = var_mean_40[1];  var_mean_40 = None
        add_199: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_137, 1e-05);  getitem_137 = None
        rsqrt_40: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
        sub_59: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(view_514, getitem_138);  getitem_138 = None
        mul_188: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_40);  sub_59 = None
        mul_189: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_188, primals_281)
        add_200: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_189, primals_282);  mul_189 = primals_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_515: "f32[25088, 512]" = torch.ops.aten.reshape.default(add_200, [25088, 512]);  add_200 = None
        permute_193: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_283, [1, 0])
        addmm_74: "f32[25088, 2048]" = torch.ops.aten.addmm.default(primals_284, view_515, permute_193);  primals_284 = permute_193 = None
        view_516: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_74, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_190: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_516, 0.5)
        mul_191: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_516, 0.7071067811865476);  view_516 = None
        erf_18: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_191);  mul_191 = None
        add_201: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_192: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_190, add_201);  mul_190 = add_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_517: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_192, [25088, 2048]);  mul_192 = None
        permute_194: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_285, [1, 0])
        addmm_75: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_286, view_517, permute_194);  primals_286 = permute_194 = None
        view_518: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(addmm_75, [128, 196, 512]);  addmm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_35: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 35)
        inductor_random_default_10: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_35, 'rand');  inductor_lookup_seed_default_35 = None
        lt_35: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_10, 0.9217391312122345);  inductor_random_default_10 = None
        convert_element_type_35: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_35, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_54: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_35, 0.9217391312122345);  convert_element_type_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_193: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_518, div_54);  view_518 = div_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_202: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_514, mul_193);  view_514 = mul_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_519: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_202, [128, 14, 14, 512]);  add_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_41 = torch.ops.aten.var_mean.correction(view_519, [3], correction = 0, keepdim = True)
        getitem_139: "f32[128, 14, 14, 1]" = var_mean_41[0]
        getitem_140: "f32[128, 14, 14, 1]" = var_mean_41[1];  var_mean_41 = None
        add_203: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_139, 1e-05);  getitem_139 = None
        rsqrt_41: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
        sub_60: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_519, getitem_140);  getitem_140 = None
        mul_194: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_41);  sub_60 = None
        mul_195: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_194, primals_287)
        add_204: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_195, primals_288);  mul_195 = primals_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_55: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(add_204, [None, fmod_8]);  add_204 = None
        index_56: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_55, [None, None, fmod_8]);  index_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_520: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(index_56, [128, 2, 7, 2, 7, 512]);  index_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_195: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_520, [0, 1, 3, 2, 4, 5]);  view_520 = None
        clone_211: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_195, memory_format = torch.contiguous_format);  permute_195 = None
        view_521: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_211, [-1, 7, 7, 512]);  clone_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_522: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_521, [-1, 49, 512]);  view_521 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_523: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_522, [25088, 512]);  view_522 = None
        permute_196: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_290, [1, 0])
        addmm_76: "f32[25088, 1536]" = torch.ops.aten.addmm.default(primals_291, view_523, permute_196);  primals_291 = permute_196 = None
        view_524: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(addmm_76, [512, 49, 1536]);  addmm_76 = None
        view_525: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.reshape.default(view_524, [512, 49, 3, 16, -1]);  view_524 = None
        permute_197: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_525, [2, 0, 3, 1, 4]);  view_525 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_19 = torch.ops.aten.unbind.int(permute_197);  permute_197 = None
        getitem_141: "f32[512, 16, 49, 32]" = unbind_19[0]
        getitem_142: "f32[512, 16, 49, 32]" = unbind_19[1]
        getitem_143: "f32[512, 16, 49, 32]" = unbind_19[2];  unbind_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_196: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_141, 0.1767766952966369);  getitem_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_198: "f32[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_142, [0, 1, 3, 2]);  getitem_142 = None
        expand_76: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_196, [512, 16, 49, 32]);  mul_196 = None
        clone_212: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
        view_526: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_212, [8192, 49, 32]);  clone_212 = None
        expand_77: "f32[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_198, [512, 16, 32, 49]);  permute_198 = None
        clone_213: "f32[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
        view_527: "f32[8192, 32, 49]" = torch.ops.aten.reshape.default(clone_213, [8192, 32, 49]);  clone_213 = None
        bmm_38: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_526, view_527)
        view_528: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_38, [512, 16, 49, 49]);  bmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_529: "i64[2401]" = torch.ops.aten.reshape.default(primals_293, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_57: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_292, [view_529]);  primals_292 = view_529 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_530: "f32[49, 49, 16]" = torch.ops.aten.reshape.default(index_57, [49, 49, -1]);  index_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_199: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_530, [2, 0, 1]);  view_530 = None
        clone_214: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_37: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_214, 0);  clone_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_207: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_528, unsqueeze_37);  view_528 = unsqueeze_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_531: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.reshape.default(add_207, [-1, 4, 16, 49, 49]);  add_207 = None
        unsqueeze_38: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(primals_289, 1);  primals_289 = None
        unsqueeze_39: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, 0);  unsqueeze_38 = None
        add_208: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_531, unsqueeze_39);  view_531 = unsqueeze_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_532: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(add_208, [-1, 16, 49, 49]);  add_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_19: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(view_532, [-1], True)
        sub_61: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_532, amax_19);  view_532 = amax_19 = None
        exp_19: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_61);  sub_61 = None
        sum_20: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_55: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_78: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_55, [512, 16, 49, 49])
        view_533: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_78, [8192, 49, 49]);  expand_78 = None
        expand_79: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_143, [512, 16, 49, 32]);  getitem_143 = None
        clone_216: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_79, memory_format = torch.contiguous_format);  expand_79 = None
        view_534: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_216, [8192, 49, 32]);  clone_216 = None
        bmm_39: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_533, view_534);  view_533 = None
        view_535: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_39, [512, 16, 49, 32]);  bmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_200: "f32[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_535, [0, 2, 1, 3]);  view_535 = None
        clone_217: "f32[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_200, memory_format = torch.contiguous_format);  permute_200 = None
        view_536: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(clone_217, [512, 49, 512]);  clone_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_537: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_536, [25088, 512]);  view_536 = None
        permute_201: "f32[512, 512]" = torch.ops.aten.permute.default(primals_294, [1, 0])
        addmm_77: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_295, view_537, permute_201);  primals_295 = permute_201 = None
        view_538: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(addmm_77, [512, 49, 512]);  addmm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_539: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_538, [-1, 7, 7, 512]);  view_538 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_540: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_539, [-1, 2, 2, 7, 7, 512]);  view_539 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_202: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_540, [0, 1, 3, 2, 4, 5]);  view_540 = None
        clone_219: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_202, memory_format = torch.contiguous_format);  permute_202 = None
        view_541: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_219, [-1, 14, 14, 512]);  clone_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_58: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_541, [None, fmod_10]);  view_541 = None
        index_59: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_58, [None, None, fmod_10]);  index_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_36: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 36)
        inductor_random_default_9: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_36, 'rand');  inductor_lookup_seed_default_36 = None
        lt_36: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_9, 0.917391300201416);  inductor_random_default_9 = None
        convert_element_type_36: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_36, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_56: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_36, 0.917391300201416);  convert_element_type_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_197: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_59, div_56);  index_59 = div_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_211: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_519, mul_197);  view_519 = mul_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_542: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_211, [128, -1, 512]);  add_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_42 = torch.ops.aten.var_mean.correction(view_542, [2], correction = 0, keepdim = True)
        getitem_144: "f32[128, 196, 1]" = var_mean_42[0]
        getitem_145: "f32[128, 196, 1]" = var_mean_42[1];  var_mean_42 = None
        add_212: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-05);  getitem_144 = None
        rsqrt_42: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_212);  add_212 = None
        sub_62: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(view_542, getitem_145);  getitem_145 = None
        mul_198: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_42);  sub_62 = None
        mul_199: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_198, primals_296)
        add_213: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_199, primals_297);  mul_199 = primals_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_543: "f32[25088, 512]" = torch.ops.aten.reshape.default(add_213, [25088, 512]);  add_213 = None
        permute_203: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_298, [1, 0])
        addmm_78: "f32[25088, 2048]" = torch.ops.aten.addmm.default(primals_299, view_543, permute_203);  primals_299 = permute_203 = None
        view_544: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_78, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_200: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_544, 0.5)
        mul_201: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_544, 0.7071067811865476);  view_544 = None
        erf_19: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_201);  mul_201 = None
        add_214: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_202: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_200, add_214);  mul_200 = add_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_545: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_202, [25088, 2048]);  mul_202 = None
        permute_204: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_300, [1, 0])
        addmm_79: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_301, view_545, permute_204);  primals_301 = permute_204 = None
        view_546: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(addmm_79, [128, 196, 512]);  addmm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_37: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 37)
        inductor_random_default_8: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_37, 'rand');  inductor_lookup_seed_default_37 = None
        lt_37: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_8, 0.917391300201416);  inductor_random_default_8 = None
        convert_element_type_37: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_37, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_57: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_37, 0.917391300201416);  convert_element_type_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_203: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_546, div_57);  view_546 = div_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_215: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_542, mul_203);  view_542 = mul_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_547: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_215, [128, 14, 14, 512]);  add_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_43 = torch.ops.aten.var_mean.correction(view_547, [3], correction = 0, keepdim = True)
        getitem_146: "f32[128, 14, 14, 1]" = var_mean_43[0]
        getitem_147: "f32[128, 14, 14, 1]" = var_mean_43[1];  var_mean_43 = None
        add_216: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_146, 1e-05);  getitem_146 = None
        rsqrt_43: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_216);  add_216 = None
        sub_63: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_547, getitem_147);  getitem_147 = None
        mul_204: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_43);  sub_63 = None
        mul_205: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_204, primals_302)
        add_217: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_205, primals_303);  mul_205 = primals_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_548: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(add_217, [128, 2, 7, 2, 7, 512]);  add_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_205: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_548, [0, 1, 3, 2, 4, 5]);  view_548 = None
        clone_222: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
        view_549: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_222, [-1, 7, 7, 512]);  clone_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_550: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_549, [-1, 49, 512]);  view_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_551: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_550, [25088, 512]);  view_550 = None
        permute_206: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_304, [1, 0])
        addmm_80: "f32[25088, 1536]" = torch.ops.aten.addmm.default(primals_305, view_551, permute_206);  primals_305 = permute_206 = None
        view_552: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(addmm_80, [512, 49, 1536]);  addmm_80 = None
        view_553: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.reshape.default(view_552, [512, 49, 3, 16, -1]);  view_552 = None
        permute_207: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_553, [2, 0, 3, 1, 4]);  view_553 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_20 = torch.ops.aten.unbind.int(permute_207);  permute_207 = None
        getitem_148: "f32[512, 16, 49, 32]" = unbind_20[0]
        getitem_149: "f32[512, 16, 49, 32]" = unbind_20[1]
        getitem_150: "f32[512, 16, 49, 32]" = unbind_20[2];  unbind_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_206: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_148, 0.1767766952966369);  getitem_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_208: "f32[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_149, [0, 1, 3, 2]);  getitem_149 = None
        expand_80: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_206, [512, 16, 49, 32]);  mul_206 = None
        clone_223: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
        view_554: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_223, [8192, 49, 32]);  clone_223 = None
        expand_81: "f32[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_208, [512, 16, 32, 49]);  permute_208 = None
        clone_224: "f32[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
        view_555: "f32[8192, 32, 49]" = torch.ops.aten.reshape.default(clone_224, [8192, 32, 49]);  clone_224 = None
        bmm_40: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_554, view_555)
        view_556: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_40, [512, 16, 49, 49]);  bmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_557: "i64[2401]" = torch.ops.aten.reshape.default(primals_307, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_60: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_306, [view_557]);  primals_306 = view_557 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_558: "f32[49, 49, 16]" = torch.ops.aten.reshape.default(index_60, [49, 49, -1]);  index_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_209: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_558, [2, 0, 1]);  view_558 = None
        clone_225: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_209, memory_format = torch.contiguous_format);  permute_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_40: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_225, 0);  clone_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_218: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_556, unsqueeze_40);  view_556 = unsqueeze_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_20: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(add_218, [-1], True)
        sub_64: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_218, amax_20);  add_218 = amax_20 = None
        exp_20: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_64);  sub_64 = None
        sum_21: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_58: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_82: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_58, [512, 16, 49, 49])
        view_559: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_82, [8192, 49, 49]);  expand_82 = None
        expand_83: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_150, [512, 16, 49, 32]);  getitem_150 = None
        clone_227: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_83, memory_format = torch.contiguous_format);  expand_83 = None
        view_560: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_227, [8192, 49, 32]);  clone_227 = None
        bmm_41: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_559, view_560);  view_559 = None
        view_561: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_41, [512, 16, 49, 32]);  bmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_210: "f32[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_561, [0, 2, 1, 3]);  view_561 = None
        clone_228: "f32[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None
        view_562: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(clone_228, [512, 49, 512]);  clone_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_563: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_562, [25088, 512]);  view_562 = None
        permute_211: "f32[512, 512]" = torch.ops.aten.permute.default(primals_308, [1, 0])
        addmm_81: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_309, view_563, permute_211);  primals_309 = permute_211 = None
        view_564: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(addmm_81, [512, 49, 512]);  addmm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_565: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_564, [-1, 7, 7, 512]);  view_564 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_566: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_565, [-1, 2, 2, 7, 7, 512]);  view_565 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_212: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_566, [0, 1, 3, 2, 4, 5]);  view_566 = None
        clone_230: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_212, memory_format = torch.contiguous_format);  permute_212 = None
        view_567: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_230, [-1, 14, 14, 512]);  clone_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_38: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 38)
        inductor_random_default_7: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_38, 'rand');  inductor_lookup_seed_default_38 = None
        lt_38: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_7, 0.9130434766411781);  inductor_random_default_7 = None
        convert_element_type_38: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_38, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_59: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_38, 0.9130434766411781);  convert_element_type_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_207: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_567, div_59);  view_567 = div_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_219: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_547, mul_207);  view_547 = mul_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_568: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_219, [128, -1, 512]);  add_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_44 = torch.ops.aten.var_mean.correction(view_568, [2], correction = 0, keepdim = True)
        getitem_151: "f32[128, 196, 1]" = var_mean_44[0]
        getitem_152: "f32[128, 196, 1]" = var_mean_44[1];  var_mean_44 = None
        add_220: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_151, 1e-05);  getitem_151 = None
        rsqrt_44: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_220);  add_220 = None
        sub_65: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(view_568, getitem_152);  getitem_152 = None
        mul_208: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_44);  sub_65 = None
        mul_209: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_208, primals_310)
        add_221: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_209, primals_311);  mul_209 = primals_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_569: "f32[25088, 512]" = torch.ops.aten.reshape.default(add_221, [25088, 512]);  add_221 = None
        permute_213: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_312, [1, 0])
        addmm_82: "f32[25088, 2048]" = torch.ops.aten.addmm.default(primals_313, view_569, permute_213);  primals_313 = permute_213 = None
        view_570: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_82, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_210: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_570, 0.5)
        mul_211: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_570, 0.7071067811865476);  view_570 = None
        erf_20: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_211);  mul_211 = None
        add_222: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_212: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_210, add_222);  mul_210 = add_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_571: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_212, [25088, 2048]);  mul_212 = None
        permute_214: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_314, [1, 0])
        addmm_83: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_315, view_571, permute_214);  primals_315 = permute_214 = None
        view_572: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(addmm_83, [128, 196, 512]);  addmm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_39: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 39)
        inductor_random_default_6: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_39, 'rand');  inductor_lookup_seed_default_39 = None
        lt_39: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_6, 0.9130434766411781);  inductor_random_default_6 = None
        convert_element_type_39: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_39, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_60: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_39, 0.9130434766411781);  convert_element_type_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_213: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_572, div_60);  view_572 = div_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_223: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_568, mul_213);  view_568 = mul_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_573: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_223, [128, 14, 14, 512]);  add_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_45 = torch.ops.aten.var_mean.correction(view_573, [3], correction = 0, keepdim = True)
        getitem_153: "f32[128, 14, 14, 1]" = var_mean_45[0]
        getitem_154: "f32[128, 14, 14, 1]" = var_mean_45[1];  var_mean_45 = None
        add_224: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_153, 1e-05);  getitem_153 = None
        rsqrt_45: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_224);  add_224 = None
        sub_66: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_573, getitem_154);  getitem_154 = None
        mul_214: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_45);  sub_66 = None
        mul_215: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_214, primals_316)
        add_225: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_215, primals_317);  mul_215 = primals_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_61: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(add_225, [None, fmod_8]);  add_225 = None
        index_62: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_61, [None, None, fmod_8]);  index_61 = fmod_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_574: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(index_62, [128, 2, 7, 2, 7, 512]);  index_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_215: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_574, [0, 1, 3, 2, 4, 5]);  view_574 = None
        clone_233: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_215, memory_format = torch.contiguous_format);  permute_215 = None
        view_575: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_233, [-1, 7, 7, 512]);  clone_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_576: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_575, [-1, 49, 512]);  view_575 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_577: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_576, [25088, 512]);  view_576 = None
        permute_216: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_319, [1, 0])
        addmm_84: "f32[25088, 1536]" = torch.ops.aten.addmm.default(primals_320, view_577, permute_216);  primals_320 = permute_216 = None
        view_578: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(addmm_84, [512, 49, 1536]);  addmm_84 = None
        view_579: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.reshape.default(view_578, [512, 49, 3, 16, -1]);  view_578 = None
        permute_217: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_579, [2, 0, 3, 1, 4]);  view_579 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_21 = torch.ops.aten.unbind.int(permute_217);  permute_217 = None
        getitem_155: "f32[512, 16, 49, 32]" = unbind_21[0]
        getitem_156: "f32[512, 16, 49, 32]" = unbind_21[1]
        getitem_157: "f32[512, 16, 49, 32]" = unbind_21[2];  unbind_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_216: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_155, 0.1767766952966369);  getitem_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_218: "f32[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_156, [0, 1, 3, 2]);  getitem_156 = None
        expand_84: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_216, [512, 16, 49, 32]);  mul_216 = None
        clone_234: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
        view_580: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_234, [8192, 49, 32]);  clone_234 = None
        expand_85: "f32[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_218, [512, 16, 32, 49]);  permute_218 = None
        clone_235: "f32[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_85, memory_format = torch.contiguous_format);  expand_85 = None
        view_581: "f32[8192, 32, 49]" = torch.ops.aten.reshape.default(clone_235, [8192, 32, 49]);  clone_235 = None
        bmm_42: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_580, view_581)
        view_582: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_42, [512, 16, 49, 49]);  bmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_583: "i64[2401]" = torch.ops.aten.reshape.default(primals_322, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_63: "f32[2401, 16]" = torch.ops.aten.index.Tensor(primals_321, [view_583]);  primals_321 = view_583 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_584: "f32[49, 49, 16]" = torch.ops.aten.reshape.default(index_63, [49, 49, -1]);  index_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_219: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_584, [2, 0, 1]);  view_584 = None
        clone_236: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_219, memory_format = torch.contiguous_format);  permute_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_41: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_236, 0);  clone_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_228: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_582, unsqueeze_41);  view_582 = unsqueeze_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_585: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.reshape.default(add_228, [-1, 4, 16, 49, 49]);  add_228 = None
        unsqueeze_42: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(primals_318, 1);  primals_318 = None
        unsqueeze_43: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, 0);  unsqueeze_42 = None
        add_229: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_585, unsqueeze_43);  view_585 = unsqueeze_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_586: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(add_229, [-1, 16, 49, 49]);  add_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_21: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(view_586, [-1], True)
        sub_67: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_586, amax_21);  view_586 = amax_21 = None
        exp_21: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_67);  sub_67 = None
        sum_22: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_61: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_86: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_61, [512, 16, 49, 49])
        view_587: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_86, [8192, 49, 49]);  expand_86 = None
        expand_87: "f32[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_157, [512, 16, 49, 32]);  getitem_157 = None
        clone_238: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_87, memory_format = torch.contiguous_format);  expand_87 = None
        view_588: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_238, [8192, 49, 32]);  clone_238 = None
        bmm_43: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_587, view_588);  view_587 = None
        view_589: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_43, [512, 16, 49, 32]);  bmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_220: "f32[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_589, [0, 2, 1, 3]);  view_589 = None
        clone_239: "f32[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_220, memory_format = torch.contiguous_format);  permute_220 = None
        view_590: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(clone_239, [512, 49, 512]);  clone_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_591: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_590, [25088, 512]);  view_590 = None
        permute_221: "f32[512, 512]" = torch.ops.aten.permute.default(primals_323, [1, 0])
        addmm_85: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_324, view_591, permute_221);  primals_324 = permute_221 = None
        view_592: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(addmm_85, [512, 49, 512]);  addmm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_593: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_592, [-1, 7, 7, 512]);  view_592 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_594: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_593, [-1, 2, 2, 7, 7, 512]);  view_593 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_222: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_594, [0, 1, 3, 2, 4, 5]);  view_594 = None
        clone_241: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_222, memory_format = torch.contiguous_format);  permute_222 = None
        view_595: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_241, [-1, 14, 14, 512]);  clone_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_64: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_595, [None, fmod_10]);  view_595 = None
        index_65: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_64, [None, None, fmod_10]);  index_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_40: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 40)
        inductor_random_default_5: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_40, 'rand');  inductor_lookup_seed_default_40 = None
        lt_40: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_5, 0.9086956530809402);  inductor_random_default_5 = None
        convert_element_type_40: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_40, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_62: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_40, 0.9086956530809402);  convert_element_type_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_217: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_65, div_62);  index_65 = div_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_232: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_573, mul_217);  view_573 = mul_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_596: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_232, [128, -1, 512]);  add_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_46 = torch.ops.aten.var_mean.correction(view_596, [2], correction = 0, keepdim = True)
        getitem_158: "f32[128, 196, 1]" = var_mean_46[0]
        getitem_159: "f32[128, 196, 1]" = var_mean_46[1];  var_mean_46 = None
        add_233: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-05);  getitem_158 = None
        rsqrt_46: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_233);  add_233 = None
        sub_68: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(view_596, getitem_159);  getitem_159 = None
        mul_218: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_46);  sub_68 = None
        mul_219: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_218, primals_325)
        add_234: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_219, primals_326);  mul_219 = primals_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_597: "f32[25088, 512]" = torch.ops.aten.reshape.default(add_234, [25088, 512]);  add_234 = None
        permute_223: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_327, [1, 0])
        addmm_86: "f32[25088, 2048]" = torch.ops.aten.addmm.default(primals_328, view_597, permute_223);  primals_328 = permute_223 = None
        view_598: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_86, [128, 196, 2048])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_220: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_598, 0.5)
        mul_221: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_598, 0.7071067811865476);  view_598 = None
        erf_21: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_221);  mul_221 = None
        add_235: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_222: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_220, add_235);  mul_220 = add_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_599: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_222, [25088, 2048]);  mul_222 = None
        permute_224: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_329, [1, 0])
        addmm_87: "f32[25088, 512]" = torch.ops.aten.addmm.default(primals_330, view_599, permute_224);  primals_330 = permute_224 = None
        view_600: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(addmm_87, [128, 196, 512]);  addmm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_41: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 41)
        inductor_random_default_4: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_41, 'rand');  inductor_lookup_seed_default_41 = None
        lt_41: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_4, 0.9086956530809402);  inductor_random_default_4 = None
        convert_element_type_41: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_41, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_63: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_41, 0.9086956530809402);  convert_element_type_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_223: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_600, div_63);  view_600 = div_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_236: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_596, mul_223);  view_596 = mul_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_601: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_236, [128, 14, 14, 512]);  add_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:539 in forward, code: x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        view_602: "f32[128, 7, 2, 7, 2, 512]" = torch.ops.aten.reshape.default(view_601, [128, 7, 2, 7, 2, 512]);  view_601 = None
        permute_225: "f32[128, 7, 7, 2, 2, 512]" = torch.ops.aten.permute.default(view_602, [0, 1, 3, 4, 2, 5]);  view_602 = None
        clone_244: "f32[128, 7, 7, 2, 2, 512]" = torch.ops.aten.clone.default(permute_225, memory_format = torch.contiguous_format);  permute_225 = None
        view_603: "f32[128, 7, 7, 2048]" = torch.ops.aten.reshape.default(clone_244, [128, 7, 7, 2048]);  clone_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:540 in forward, code: x = self.norm(x)
        var_mean_47 = torch.ops.aten.var_mean.correction(view_603, [3], correction = 0, keepdim = True)
        getitem_160: "f32[128, 7, 7, 1]" = var_mean_47[0]
        getitem_161: "f32[128, 7, 7, 1]" = var_mean_47[1];  var_mean_47 = None
        add_237: "f32[128, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-05);  getitem_160 = None
        rsqrt_47: "f32[128, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_237);  add_237 = None
        sub_69: "f32[128, 7, 7, 2048]" = torch.ops.aten.sub.Tensor(view_603, getitem_161);  view_603 = getitem_161 = None
        mul_224: "f32[128, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_47);  sub_69 = None
        mul_225: "f32[128, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(mul_224, primals_331)
        add_238: "f32[128, 7, 7, 2048]" = torch.ops.aten.add.Tensor(mul_225, primals_332);  mul_225 = primals_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:541 in forward, code: x = self.reduction(x)
        permute_226: "f32[2048, 1024]" = torch.ops.aten.permute.default(primals_333, [1, 0])
        view_604: "f32[6272, 2048]" = torch.ops.aten.reshape.default(add_238, [6272, 2048]);  add_238 = None
        mm_2: "f32[6272, 1024]" = torch.ops.aten.mm.default(view_604, permute_226);  permute_226 = None
        view_605: "f32[128, 7, 7, 1024]" = torch.ops.aten.reshape.default(mm_2, [128, 7, 7, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_48 = torch.ops.aten.var_mean.correction(view_605, [3], correction = 0, keepdim = True)
        getitem_162: "f32[128, 7, 7, 1]" = var_mean_48[0]
        getitem_163: "f32[128, 7, 7, 1]" = var_mean_48[1];  var_mean_48 = None
        add_239: "f32[128, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem_162, 1e-05);  getitem_162 = None
        rsqrt_48: "f32[128, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_239);  add_239 = None
        sub_70: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(view_605, getitem_163)
        mul_226: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_48);  sub_70 = None
        mul_227: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_226, primals_334);  mul_226 = None
        add_240: "f32[128, 7, 7, 1024]" = torch.ops.aten.add.Tensor(mul_227, primals_335);  mul_227 = primals_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_606: "f32[128, 1, 7, 1, 7, 1024]" = torch.ops.aten.reshape.default(add_240, [128, 1, 7, 1, 7, 1024]);  add_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_227: "f32[128, 1, 1, 7, 7, 1024]" = torch.ops.aten.permute.default(view_606, [0, 1, 3, 2, 4, 5]);  view_606 = None
        view_607: "f32[128, 7, 7, 1024]" = torch.ops.aten.reshape.default(permute_227, [-1, 7, 7, 1024]);  permute_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_608: "f32[128, 49, 1024]" = torch.ops.aten.reshape.default(view_607, [-1, 49, 1024]);  view_607 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_609: "f32[6272, 1024]" = torch.ops.aten.reshape.default(view_608, [6272, 1024]);  view_608 = None
        permute_228: "f32[1024, 3072]" = torch.ops.aten.permute.default(primals_336, [1, 0])
        addmm_88: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_337, view_609, permute_228);  primals_337 = permute_228 = None
        view_610: "f32[128, 49, 3072]" = torch.ops.aten.reshape.default(addmm_88, [128, 49, 3072]);  addmm_88 = None
        view_611: "f32[128, 49, 3, 32, 32]" = torch.ops.aten.reshape.default(view_610, [128, 49, 3, 32, -1]);  view_610 = None
        permute_229: "f32[3, 128, 32, 49, 32]" = torch.ops.aten.permute.default(view_611, [2, 0, 3, 1, 4]);  view_611 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_22 = torch.ops.aten.unbind.int(permute_229);  permute_229 = None
        getitem_164: "f32[128, 32, 49, 32]" = unbind_22[0]
        getitem_165: "f32[128, 32, 49, 32]" = unbind_22[1]
        getitem_166: "f32[128, 32, 49, 32]" = unbind_22[2];  unbind_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_228: "f32[128, 32, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_164, 0.1767766952966369);  getitem_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_230: "f32[128, 32, 32, 49]" = torch.ops.aten.permute.default(getitem_165, [0, 1, 3, 2]);  getitem_165 = None
        expand_88: "f32[128, 32, 49, 32]" = torch.ops.aten.expand.default(mul_228, [128, 32, 49, 32]);  mul_228 = None
        clone_245: "f32[128, 32, 49, 32]" = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
        view_612: "f32[4096, 49, 32]" = torch.ops.aten.reshape.default(clone_245, [4096, 49, 32]);  clone_245 = None
        expand_89: "f32[128, 32, 32, 49]" = torch.ops.aten.expand.default(permute_230, [128, 32, 32, 49]);  permute_230 = None
        clone_246: "f32[128, 32, 32, 49]" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
        view_613: "f32[4096, 32, 49]" = torch.ops.aten.reshape.default(clone_246, [4096, 32, 49]);  clone_246 = None
        bmm_44: "f32[4096, 49, 49]" = torch.ops.aten.bmm.default(view_612, view_613)
        view_614: "f32[128, 32, 49, 49]" = torch.ops.aten.reshape.default(bmm_44, [128, 32, 49, 49]);  bmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_615: "i64[2401]" = torch.ops.aten.reshape.default(primals_339, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_66: "f32[2401, 32]" = torch.ops.aten.index.Tensor(primals_338, [view_615]);  primals_338 = view_615 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_616: "f32[49, 49, 32]" = torch.ops.aten.reshape.default(index_66, [49, 49, -1]);  index_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_231: "f32[32, 49, 49]" = torch.ops.aten.permute.default(view_616, [2, 0, 1]);  view_616 = None
        clone_247: "f32[32, 49, 49]" = torch.ops.aten.clone.default(permute_231, memory_format = torch.contiguous_format);  permute_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_44: "f32[1, 32, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_247, 0);  clone_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_241: "f32[128, 32, 49, 49]" = torch.ops.aten.add.Tensor(view_614, unsqueeze_44);  view_614 = unsqueeze_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_22: "f32[128, 32, 49, 1]" = torch.ops.aten.amax.default(add_241, [-1], True)
        sub_71: "f32[128, 32, 49, 49]" = torch.ops.aten.sub.Tensor(add_241, amax_22);  add_241 = amax_22 = None
        exp_22: "f32[128, 32, 49, 49]" = torch.ops.aten.exp.default(sub_71);  sub_71 = None
        sum_23: "f32[128, 32, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_64: "f32[128, 32, 49, 49]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_90: "f32[128, 32, 49, 49]" = torch.ops.aten.expand.default(div_64, [128, 32, 49, 49])
        view_617: "f32[4096, 49, 49]" = torch.ops.aten.reshape.default(expand_90, [4096, 49, 49]);  expand_90 = None
        expand_91: "f32[128, 32, 49, 32]" = torch.ops.aten.expand.default(getitem_166, [128, 32, 49, 32]);  getitem_166 = None
        clone_249: "f32[128, 32, 49, 32]" = torch.ops.aten.clone.default(expand_91, memory_format = torch.contiguous_format);  expand_91 = None
        view_618: "f32[4096, 49, 32]" = torch.ops.aten.reshape.default(clone_249, [4096, 49, 32]);  clone_249 = None
        bmm_45: "f32[4096, 49, 32]" = torch.ops.aten.bmm.default(view_617, view_618);  view_617 = None
        view_619: "f32[128, 32, 49, 32]" = torch.ops.aten.reshape.default(bmm_45, [128, 32, 49, 32]);  bmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_232: "f32[128, 49, 32, 32]" = torch.ops.aten.permute.default(view_619, [0, 2, 1, 3]);  view_619 = None
        clone_250: "f32[128, 49, 32, 32]" = torch.ops.aten.clone.default(permute_232, memory_format = torch.contiguous_format);  permute_232 = None
        view_620: "f32[128, 49, 1024]" = torch.ops.aten.reshape.default(clone_250, [128, 49, 1024]);  clone_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_621: "f32[6272, 1024]" = torch.ops.aten.reshape.default(view_620, [6272, 1024]);  view_620 = None
        permute_233: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_340, [1, 0])
        addmm_89: "f32[6272, 1024]" = torch.ops.aten.addmm.default(primals_341, view_621, permute_233);  primals_341 = permute_233 = None
        view_622: "f32[128, 49, 1024]" = torch.ops.aten.reshape.default(addmm_89, [128, 49, 1024]);  addmm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_623: "f32[128, 7, 7, 1024]" = torch.ops.aten.reshape.default(view_622, [-1, 7, 7, 1024]);  view_622 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_624: "f32[128, 1, 1, 7, 7, 1024]" = torch.ops.aten.reshape.default(view_623, [-1, 1, 1, 7, 7, 1024]);  view_623 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_234: "f32[128, 1, 7, 1, 7, 1024]" = torch.ops.aten.permute.default(view_624, [0, 1, 3, 2, 4, 5]);  view_624 = None
        view_625: "f32[128, 7, 7, 1024]" = torch.ops.aten.reshape.default(permute_234, [-1, 7, 7, 1024]);  permute_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_42: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 42)
        inductor_random_default_3: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_42, 'rand');  inductor_lookup_seed_default_42 = None
        lt_42: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_3, 0.9043478220701218);  inductor_random_default_3 = None
        convert_element_type_42: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_42, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_65: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_42, 0.9043478220701218);  convert_element_type_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_229: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_625, div_65);  view_625 = div_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_242: "f32[128, 7, 7, 1024]" = torch.ops.aten.add.Tensor(view_605, mul_229);  view_605 = mul_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_626: "f32[128, 49, 1024]" = torch.ops.aten.reshape.default(add_242, [128, -1, 1024]);  add_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_49 = torch.ops.aten.var_mean.correction(view_626, [2], correction = 0, keepdim = True)
        getitem_167: "f32[128, 49, 1]" = var_mean_49[0]
        getitem_168: "f32[128, 49, 1]" = var_mean_49[1];  var_mean_49 = None
        add_243: "f32[128, 49, 1]" = torch.ops.aten.add.Tensor(getitem_167, 1e-05);  getitem_167 = None
        rsqrt_49: "f32[128, 49, 1]" = torch.ops.aten.rsqrt.default(add_243);  add_243 = None
        sub_72: "f32[128, 49, 1024]" = torch.ops.aten.sub.Tensor(view_626, getitem_168);  getitem_168 = None
        mul_230: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_49);  sub_72 = None
        mul_231: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(mul_230, primals_342)
        add_244: "f32[128, 49, 1024]" = torch.ops.aten.add.Tensor(mul_231, primals_343);  mul_231 = primals_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_627: "f32[6272, 1024]" = torch.ops.aten.reshape.default(add_244, [6272, 1024]);  add_244 = None
        permute_235: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_344, [1, 0])
        addmm_90: "f32[6272, 4096]" = torch.ops.aten.addmm.default(primals_345, view_627, permute_235);  primals_345 = permute_235 = None
        view_628: "f32[128, 49, 4096]" = torch.ops.aten.reshape.default(addmm_90, [128, 49, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_232: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(view_628, 0.5)
        mul_233: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(view_628, 0.7071067811865476);  view_628 = None
        erf_22: "f32[128, 49, 4096]" = torch.ops.aten.erf.default(mul_233);  mul_233 = None
        add_245: "f32[128, 49, 4096]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_234: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(mul_232, add_245);  mul_232 = add_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_629: "f32[6272, 4096]" = torch.ops.aten.reshape.default(mul_234, [6272, 4096]);  mul_234 = None
        permute_236: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_346, [1, 0])
        addmm_91: "f32[6272, 1024]" = torch.ops.aten.addmm.default(primals_347, view_629, permute_236);  primals_347 = permute_236 = None
        view_630: "f32[128, 49, 1024]" = torch.ops.aten.reshape.default(addmm_91, [128, 49, 1024]);  addmm_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_43: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 43)
        inductor_random_default_2: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_43, 'rand');  inductor_lookup_seed_default_43 = None
        lt_43: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_2, 0.9043478220701218);  inductor_random_default_2 = None
        convert_element_type_43: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_43, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_66: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_43, 0.9043478220701218);  convert_element_type_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_235: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(view_630, div_66);  view_630 = div_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_246: "f32[128, 49, 1024]" = torch.ops.aten.add.Tensor(view_626, mul_235);  view_626 = mul_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_631: "f32[128, 7, 7, 1024]" = torch.ops.aten.reshape.default(add_246, [128, 7, 7, 1024]);  add_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_50 = torch.ops.aten.var_mean.correction(view_631, [3], correction = 0, keepdim = True)
        getitem_169: "f32[128, 7, 7, 1]" = var_mean_50[0]
        getitem_170: "f32[128, 7, 7, 1]" = var_mean_50[1];  var_mean_50 = None
        add_247: "f32[128, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem_169, 1e-05);  getitem_169 = None
        rsqrt_50: "f32[128, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_247);  add_247 = None
        sub_73: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(view_631, getitem_170);  getitem_170 = None
        mul_236: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_50);  sub_73 = None
        mul_237: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_236, primals_348)
        add_248: "f32[128, 7, 7, 1024]" = torch.ops.aten.add.Tensor(mul_237, primals_349);  mul_237 = primals_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_632: "f32[128, 1, 7, 1, 7, 1024]" = torch.ops.aten.reshape.default(add_248, [128, 1, 7, 1, 7, 1024]);  add_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_237: "f32[128, 1, 1, 7, 7, 1024]" = torch.ops.aten.permute.default(view_632, [0, 1, 3, 2, 4, 5]);  view_632 = None
        view_633: "f32[128, 7, 7, 1024]" = torch.ops.aten.reshape.default(permute_237, [-1, 7, 7, 1024]);  permute_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_634: "f32[128, 49, 1024]" = torch.ops.aten.reshape.default(view_633, [-1, 49, 1024]);  view_633 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        view_635: "f32[6272, 1024]" = torch.ops.aten.reshape.default(view_634, [6272, 1024]);  view_634 = None
        permute_238: "f32[1024, 3072]" = torch.ops.aten.permute.default(primals_350, [1, 0])
        addmm_92: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_351, view_635, permute_238);  primals_351 = permute_238 = None
        view_636: "f32[128, 49, 3072]" = torch.ops.aten.reshape.default(addmm_92, [128, 49, 3072]);  addmm_92 = None
        view_637: "f32[128, 49, 3, 32, 32]" = torch.ops.aten.reshape.default(view_636, [128, 49, 3, 32, -1]);  view_636 = None
        permute_239: "f32[3, 128, 32, 49, 32]" = torch.ops.aten.permute.default(view_637, [2, 0, 3, 1, 4]);  view_637 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_23 = torch.ops.aten.unbind.int(permute_239);  permute_239 = None
        getitem_171: "f32[128, 32, 49, 32]" = unbind_23[0]
        getitem_172: "f32[128, 32, 49, 32]" = unbind_23[1]
        getitem_173: "f32[128, 32, 49, 32]" = unbind_23[2];  unbind_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_238: "f32[128, 32, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_171, 0.1767766952966369);  getitem_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_240: "f32[128, 32, 32, 49]" = torch.ops.aten.permute.default(getitem_172, [0, 1, 3, 2]);  getitem_172 = None
        expand_92: "f32[128, 32, 49, 32]" = torch.ops.aten.expand.default(mul_238, [128, 32, 49, 32]);  mul_238 = None
        clone_254: "f32[128, 32, 49, 32]" = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
        view_638: "f32[4096, 49, 32]" = torch.ops.aten.reshape.default(clone_254, [4096, 49, 32]);  clone_254 = None
        expand_93: "f32[128, 32, 32, 49]" = torch.ops.aten.expand.default(permute_240, [128, 32, 32, 49]);  permute_240 = None
        clone_255: "f32[128, 32, 32, 49]" = torch.ops.aten.clone.default(expand_93, memory_format = torch.contiguous_format);  expand_93 = None
        view_639: "f32[4096, 32, 49]" = torch.ops.aten.reshape.default(clone_255, [4096, 32, 49]);  clone_255 = None
        bmm_46: "f32[4096, 49, 49]" = torch.ops.aten.bmm.default(view_638, view_639)
        view_640: "f32[128, 32, 49, 49]" = torch.ops.aten.reshape.default(bmm_46, [128, 32, 49, 49]);  bmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_641: "i64[2401]" = torch.ops.aten.reshape.default(primals_353, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_67: "f32[2401, 32]" = torch.ops.aten.index.Tensor(primals_352, [view_641]);  primals_352 = view_641 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_642: "f32[49, 49, 32]" = torch.ops.aten.reshape.default(index_67, [49, 49, -1]);  index_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_241: "f32[32, 49, 49]" = torch.ops.aten.permute.default(view_642, [2, 0, 1]);  view_642 = None
        clone_256: "f32[32, 49, 49]" = torch.ops.aten.clone.default(permute_241, memory_format = torch.contiguous_format);  permute_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_45: "f32[1, 32, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_256, 0);  clone_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_249: "f32[128, 32, 49, 49]" = torch.ops.aten.add.Tensor(view_640, unsqueeze_45);  view_640 = unsqueeze_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_23: "f32[128, 32, 49, 1]" = torch.ops.aten.amax.default(add_249, [-1], True)
        sub_74: "f32[128, 32, 49, 49]" = torch.ops.aten.sub.Tensor(add_249, amax_23);  add_249 = amax_23 = None
        exp_23: "f32[128, 32, 49, 49]" = torch.ops.aten.exp.default(sub_74);  sub_74 = None
        sum_24: "f32[128, 32, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_67: "f32[128, 32, 49, 49]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_94: "f32[128, 32, 49, 49]" = torch.ops.aten.expand.default(div_67, [128, 32, 49, 49])
        view_643: "f32[4096, 49, 49]" = torch.ops.aten.reshape.default(expand_94, [4096, 49, 49]);  expand_94 = None
        expand_95: "f32[128, 32, 49, 32]" = torch.ops.aten.expand.default(getitem_173, [128, 32, 49, 32]);  getitem_173 = None
        clone_258: "f32[128, 32, 49, 32]" = torch.ops.aten.clone.default(expand_95, memory_format = torch.contiguous_format);  expand_95 = None
        view_644: "f32[4096, 49, 32]" = torch.ops.aten.reshape.default(clone_258, [4096, 49, 32]);  clone_258 = None
        bmm_47: "f32[4096, 49, 32]" = torch.ops.aten.bmm.default(view_643, view_644);  view_643 = None
        view_645: "f32[128, 32, 49, 32]" = torch.ops.aten.reshape.default(bmm_47, [128, 32, 49, 32]);  bmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_242: "f32[128, 49, 32, 32]" = torch.ops.aten.permute.default(view_645, [0, 2, 1, 3]);  view_645 = None
        clone_259: "f32[128, 49, 32, 32]" = torch.ops.aten.clone.default(permute_242, memory_format = torch.contiguous_format);  permute_242 = None
        view_646: "f32[128, 49, 1024]" = torch.ops.aten.reshape.default(clone_259, [128, 49, 1024]);  clone_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_647: "f32[6272, 1024]" = torch.ops.aten.reshape.default(view_646, [6272, 1024]);  view_646 = None
        permute_243: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_354, [1, 0])
        addmm_93: "f32[6272, 1024]" = torch.ops.aten.addmm.default(primals_355, view_647, permute_243);  primals_355 = permute_243 = None
        view_648: "f32[128, 49, 1024]" = torch.ops.aten.reshape.default(addmm_93, [128, 49, 1024]);  addmm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_649: "f32[128, 7, 7, 1024]" = torch.ops.aten.reshape.default(view_648, [-1, 7, 7, 1024]);  view_648 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_650: "f32[128, 1, 1, 7, 7, 1024]" = torch.ops.aten.reshape.default(view_649, [-1, 1, 1, 7, 7, 1024]);  view_649 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_244: "f32[128, 1, 7, 1, 7, 1024]" = torch.ops.aten.permute.default(view_650, [0, 1, 3, 2, 4, 5]);  view_650 = None
        view_651: "f32[128, 7, 7, 1024]" = torch.ops.aten.reshape.default(permute_244, [-1, 7, 7, 1024]);  permute_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_44: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 44)
        inductor_random_default_1: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default_44, 'rand');  inductor_lookup_seed_default_44 = None
        lt_44: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_1, 0.8999999985098839);  inductor_random_default_1 = None
        convert_element_type_44: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_44, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_68: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_44, 0.8999999985098839);  convert_element_type_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_239: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_651, div_68);  view_651 = div_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_250: "f32[128, 7, 7, 1024]" = torch.ops.aten.add.Tensor(view_631, mul_239);  view_631 = mul_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_652: "f32[128, 49, 1024]" = torch.ops.aten.reshape.default(add_250, [128, -1, 1024]);  add_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_51 = torch.ops.aten.var_mean.correction(view_652, [2], correction = 0, keepdim = True)
        getitem_174: "f32[128, 49, 1]" = var_mean_51[0]
        getitem_175: "f32[128, 49, 1]" = var_mean_51[1];  var_mean_51 = None
        add_251: "f32[128, 49, 1]" = torch.ops.aten.add.Tensor(getitem_174, 1e-05);  getitem_174 = None
        rsqrt_51: "f32[128, 49, 1]" = torch.ops.aten.rsqrt.default(add_251);  add_251 = None
        sub_75: "f32[128, 49, 1024]" = torch.ops.aten.sub.Tensor(view_652, getitem_175);  getitem_175 = None
        mul_240: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_51);  sub_75 = None
        mul_241: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(mul_240, primals_356)
        add_252: "f32[128, 49, 1024]" = torch.ops.aten.add.Tensor(mul_241, primals_357);  mul_241 = primals_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_653: "f32[6272, 1024]" = torch.ops.aten.reshape.default(add_252, [6272, 1024]);  add_252 = None
        permute_245: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_358, [1, 0])
        addmm_94: "f32[6272, 4096]" = torch.ops.aten.addmm.default(primals_359, view_653, permute_245);  primals_359 = permute_245 = None
        view_654: "f32[128, 49, 4096]" = torch.ops.aten.reshape.default(addmm_94, [128, 49, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_242: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(view_654, 0.5)
        mul_243: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(view_654, 0.7071067811865476);  view_654 = None
        erf_23: "f32[128, 49, 4096]" = torch.ops.aten.erf.default(mul_243);  mul_243 = None
        add_253: "f32[128, 49, 4096]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_244: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(mul_242, add_253);  mul_242 = add_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_655: "f32[6272, 4096]" = torch.ops.aten.reshape.default(mul_244, [6272, 4096]);  mul_244 = None
        permute_246: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_360, [1, 0])
        addmm_95: "f32[6272, 1024]" = torch.ops.aten.addmm.default(primals_361, view_655, permute_246);  primals_361 = permute_246 = None
        view_656: "f32[128, 49, 1024]" = torch.ops.aten.reshape.default(addmm_95, [128, 49, 1024]);  addmm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        inductor_lookup_seed_default_45: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 45);  inductor_seeds_default = None
        inductor_random_default: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default_45, 'rand');  inductor_lookup_seed_default_45 = None
        lt_45: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default, 0.8999999985098839);  inductor_random_default = None
        convert_element_type_45: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_45, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_69: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_45, 0.8999999985098839);  convert_element_type_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_245: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(view_656, div_69);  view_656 = div_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_254: "f32[128, 49, 1024]" = torch.ops.aten.add.Tensor(view_652, mul_245);  view_652 = mul_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_657: "f32[128, 7, 7, 1024]" = torch.ops.aten.reshape.default(add_254, [128, 7, 7, 1024]);  add_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:981 in forward_features, code: x = self.norm(x)
        var_mean_52 = torch.ops.aten.var_mean.correction(view_657, [3], correction = 0, keepdim = True)
        getitem_176: "f32[128, 7, 7, 1]" = var_mean_52[0]
        getitem_177: "f32[128, 7, 7, 1]" = var_mean_52[1];  var_mean_52 = None
        add_255: "f32[128, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-05);  getitem_176 = None
        rsqrt_52: "f32[128, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_255);  add_255 = None
        sub_76: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(view_657, getitem_177);  view_657 = getitem_177 = None
        mul_246: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_52);  sub_76 = None
        mul_247: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_246, primals_362)
        add_256: "f32[128, 7, 7, 1024]" = torch.ops.aten.add.Tensor(mul_247, primals_363);  mul_247 = primals_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:65 in forward, code: return x.mean(self.dim, keepdim=not self.flatten)
        mean: "f32[128, 1024]" = torch.ops.aten.mean.dim(add_256, [1, 2]);  add_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:141 in forward, code: x = self.fc(x)
        permute_247: "f32[1024, 1000]" = torch.ops.aten.permute.default(primals_364, [1, 0])
        addmm_96: "f32[128, 1000]" = torch.ops.aten.addmm.default(primals_365, mean, permute_247);  primals_365 = permute_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:981 in forward_features, code: x = self.norm(x)
        div_71: "f32[128, 7, 7, 1]" = torch.ops.aten.div.Tensor(rsqrt_52, 1024);  rsqrt_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_72: "f32[128, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_51, 1024);  rsqrt_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_267: "f32[4096, 32, 49]" = torch.ops.aten.permute.default(view_644, [0, 2, 1]);  view_644 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_269: "f32[4096, 32, 49]" = torch.ops.aten.permute.default(view_638, [0, 2, 1]);  view_638 = None
        permute_270: "f32[4096, 49, 32]" = torch.ops.aten.permute.default(view_639, [0, 2, 1]);  view_639 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_73: "f32[128, 7, 7, 1]" = torch.ops.aten.div.Tensor(rsqrt_50, 1024);  rsqrt_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_74: "f32[128, 49, 1]" = torch.ops.aten.div.Tensor(rsqrt_49, 1024);  rsqrt_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_293: "f32[4096, 32, 49]" = torch.ops.aten.permute.default(view_618, [0, 2, 1]);  view_618 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_295: "f32[4096, 32, 49]" = torch.ops.aten.permute.default(view_612, [0, 2, 1]);  view_612 = None
        permute_296: "f32[4096, 49, 32]" = torch.ops.aten.permute.default(view_613, [0, 2, 1]);  view_613 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:540 in forward, code: x = self.norm(x)
        div_76: "f32[128, 7, 7, 1]" = torch.ops.aten.div.Tensor(rsqrt_47, 2048);  rsqrt_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_77: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 512);  rsqrt_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_324: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_588, [0, 2, 1]);  view_588 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_326: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_580, [0, 2, 1]);  view_580 = None
        permute_327: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_581, [0, 2, 1]);  view_581 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_78: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 512);  rsqrt_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_79: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 512);  rsqrt_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_350: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_560, [0, 2, 1]);  view_560 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_352: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_554, [0, 2, 1]);  view_554 = None
        permute_353: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_555, [0, 2, 1]);  view_555 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_80: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 512);  rsqrt_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_81: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_42, 512);  rsqrt_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_376: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_534, [0, 2, 1]);  view_534 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_378: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_526, [0, 2, 1]);  view_526 = None
        permute_379: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_527, [0, 2, 1]);  view_527 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_82: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_41, 512);  rsqrt_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_83: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 512);  rsqrt_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_402: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_506, [0, 2, 1]);  view_506 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_404: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_500, [0, 2, 1]);  view_500 = None
        permute_405: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_501, [0, 2, 1]);  view_501 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_84: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 512);  rsqrt_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_85: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 512);  rsqrt_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_428: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_480, [0, 2, 1]);  view_480 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_430: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_472, [0, 2, 1]);  view_472 = None
        permute_431: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_473, [0, 2, 1]);  view_473 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_86: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 512);  rsqrt_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_87: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 512);  rsqrt_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_454: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_452, [0, 2, 1]);  view_452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_456: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_446, [0, 2, 1]);  view_446 = None
        permute_457: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_447, [0, 2, 1]);  view_447 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_88: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 512);  rsqrt_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_89: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 512);  rsqrt_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_480: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_426, [0, 2, 1]);  view_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_482: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_418, [0, 2, 1]);  view_418 = None
        permute_483: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_419, [0, 2, 1]);  view_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_90: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 512);  rsqrt_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_91: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 512);  rsqrt_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_506: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_398, [0, 2, 1]);  view_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_508: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_392, [0, 2, 1]);  view_392 = None
        permute_509: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_393, [0, 2, 1]);  view_393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_92: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 512);  rsqrt_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_93: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 512);  rsqrt_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_532: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_372, [0, 2, 1]);  view_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_534: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_364, [0, 2, 1]);  view_364 = None
        permute_535: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_365, [0, 2, 1]);  view_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_94: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 512);  rsqrt_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_95: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 512);  rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_558: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_344, [0, 2, 1]);  view_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_560: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_338, [0, 2, 1]);  view_338 = None
        permute_561: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_339, [0, 2, 1]);  view_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_96: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 512);  rsqrt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_97: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 512);  rsqrt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_584: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_318, [0, 2, 1]);  view_318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_586: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_310, [0, 2, 1]);  view_310 = None
        permute_587: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_311, [0, 2, 1]);  view_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_98: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 512);  rsqrt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_99: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 512);  rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_610: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_290, [0, 2, 1]);  view_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_612: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_284, [0, 2, 1]);  view_284 = None
        permute_613: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_285, [0, 2, 1]);  view_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_100: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 512);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_101: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 512);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_636: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_264, [0, 2, 1]);  view_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_638: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_256, [0, 2, 1]);  view_256 = None
        permute_639: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_257, [0, 2, 1]);  view_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_102: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 512);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_103: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 512);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_662: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_236, [0, 2, 1]);  view_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_664: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_230, [0, 2, 1]);  view_230 = None
        permute_665: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_231, [0, 2, 1]);  view_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_104: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 512);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_105: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 512);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_688: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_690: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_202, [0, 2, 1]);  view_202 = None
        permute_691: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_203, [0, 2, 1]);  view_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_106: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 512);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_107: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 512);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_714: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_182, [0, 2, 1]);  view_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_716: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_176, [0, 2, 1]);  view_176 = None
        permute_717: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_177, [0, 2, 1]);  view_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_108: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 512);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_109: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 512);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_740: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_156, [0, 2, 1]);  view_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_742: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_148, [0, 2, 1]);  view_148 = None
        permute_743: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_149, [0, 2, 1]);  view_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_110: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 512);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_111: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 512);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_766: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_128, [0, 2, 1]);  view_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_768: "f32[8192, 32, 49]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
        permute_769: "f32[8192, 49, 32]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:540 in forward, code: x = self.norm(x)
        div_113: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 1024);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_114: "f32[128, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 256);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_797: "f32[16384, 32, 49]" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_799: "f32[16384, 32, 49]" = torch.ops.aten.permute.default(view_90, [0, 2, 1]);  view_90 = None
        permute_800: "f32[16384, 49, 32]" = torch.ops.aten.permute.default(view_91, [0, 2, 1]);  view_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_115: "f32[128, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 256);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_116: "f32[128, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 256);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_823: "f32[16384, 32, 49]" = torch.ops.aten.permute.default(view_70, [0, 2, 1]);  view_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_825: "f32[16384, 32, 49]" = torch.ops.aten.permute.default(view_64, [0, 2, 1]);  view_64 = None
        permute_826: "f32[16384, 49, 32]" = torch.ops.aten.permute.default(view_65, [0, 2, 1]);  view_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:540 in forward, code: x = self.norm(x)
        div_118: "f32[128, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 512);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_119: "f32[128, 3136, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 128);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_854: "f32[32768, 32, 49]" = torch.ops.aten.permute.default(view_40, [0, 2, 1]);  view_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_856: "f32[32768, 32, 49]" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None
        permute_857: "f32[32768, 49, 32]" = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        div_120: "f32[128, 56, 56, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 128);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        div_121: "f32[128, 3136, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 128);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        permute_880: "f32[32768, 32, 49]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        permute_882: "f32[32768, 32, 49]" = torch.ops.aten.permute.default(view_6, [0, 2, 1]);  view_6 = None
        permute_883: "f32[32768, 49, 32]" = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
        return (addmm_96, primals_1, primals_2, primals_4, primals_5, primals_6, primals_8, primals_11, primals_12, primals_14, primals_16, primals_18, primals_20, primals_23, primals_26, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_38, primals_40, primals_43, primals_44, primals_46, primals_48, primals_50, primals_52, primals_55, primals_58, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_70, primals_72, primals_75, primals_76, primals_78, primals_80, primals_82, primals_84, primals_87, primals_90, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_104, primals_105, primals_107, primals_109, primals_111, primals_113, primals_116, primals_119, primals_120, primals_122, primals_124, primals_126, primals_128, primals_130, primals_133, primals_134, primals_136, primals_138, primals_140, primals_142, primals_145, primals_148, primals_149, primals_151, primals_153, primals_155, primals_157, primals_159, primals_162, primals_163, primals_165, primals_167, primals_169, primals_171, primals_174, primals_177, primals_178, primals_180, primals_182, primals_184, primals_186, primals_188, primals_191, primals_192, primals_194, primals_196, primals_198, primals_200, primals_203, primals_206, primals_207, primals_209, primals_211, primals_213, primals_215, primals_217, primals_220, primals_221, primals_223, primals_225, primals_227, primals_229, primals_232, primals_235, primals_236, primals_238, primals_240, primals_242, primals_244, primals_246, primals_249, primals_250, primals_252, primals_254, primals_256, primals_258, primals_261, primals_264, primals_265, primals_267, primals_269, primals_271, primals_273, primals_275, primals_278, primals_279, primals_281, primals_283, primals_285, primals_287, primals_290, primals_293, primals_294, primals_296, primals_298, primals_300, primals_302, primals_304, primals_307, primals_308, primals_310, primals_312, primals_314, primals_316, primals_319, primals_322, primals_323, primals_325, primals_327, primals_329, primals_331, primals_333, primals_334, primals_336, primals_339, primals_340, primals_342, primals_344, primals_346, primals_348, primals_350, primals_353, primals_354, primals_356, primals_358, primals_360, primals_362, primals_364, convolution, getitem_1, rsqrt, getitem_3, rsqrt_1, view_3, div, view_15, mul_5, view_21, addmm_2, view_23, mul_10, view_29, div_1, view_43, fmod_2, lt, mul_14, view_49, addmm_6, view_51, lt_1, mul_20, view_56, mm, getitem_19, rsqrt_6, view_61, div_4, view_73, lt_2, mul_26, view_79, addmm_10, view_81, lt_3, mul_32, view_87, div_7, view_101, fmod_6, lt_4, mul_36, view_107, addmm_14, view_109, lt_5, mul_42, view_114, mm_1, getitem_35, rsqrt_11, view_119, div_10, view_131, lt_6, mul_48, view_137, addmm_18, view_139, lt_7, mul_54, view_145, div_13, view_159, fmod_10, lt_8, mul_58, view_165, addmm_22, view_167, lt_9, mul_64, view_173, div_16, view_185, lt_10, mul_68, view_191, addmm_26, view_193, lt_11, mul_74, view_199, div_19, view_213, lt_12, mul_78, view_219, addmm_30, view_221, lt_13, mul_84, view_227, div_22, view_239, lt_14, mul_88, view_245, addmm_34, view_247, lt_15, mul_94, view_253, div_25, view_267, lt_16, mul_98, view_273, addmm_38, view_275, lt_17, mul_104, view_281, div_28, view_293, lt_18, mul_108, view_299, addmm_42, view_301, lt_19, mul_114, view_307, div_31, view_321, lt_20, mul_118, view_327, addmm_46, view_329, lt_21, mul_124, view_335, div_34, view_347, lt_22, mul_128, view_353, addmm_50, view_355, lt_23, mul_134, view_361, div_37, view_375, lt_24, mul_138, view_381, addmm_54, view_383, lt_25, mul_144, view_389, div_40, view_401, lt_26, mul_148, view_407, addmm_58, view_409, lt_27, mul_154, view_415, div_43, view_429, lt_28, mul_158, view_435, addmm_62, view_437, lt_29, mul_164, view_443, div_46, view_455, lt_30, mul_168, view_461, addmm_66, view_463, lt_31, mul_174, view_469, div_49, view_483, lt_32, mul_178, view_489, addmm_70, view_491, lt_33, mul_184, view_497, div_52, view_509, lt_34, mul_188, view_515, addmm_74, view_517, lt_35, mul_194, view_523, div_55, view_537, lt_36, mul_198, view_543, addmm_78, view_545, lt_37, mul_204, view_551, div_58, view_563, lt_38, mul_208, view_569, addmm_82, view_571, lt_39, mul_214, view_577, div_61, view_591, lt_40, mul_218, view_597, addmm_86, view_599, lt_41, mul_224, view_604, mm_2, getitem_163, rsqrt_48, view_609, div_64, view_621, lt_42, mul_230, view_627, addmm_90, view_629, lt_43, mul_236, view_635, div_67, view_647, lt_44, mul_240, view_653, addmm_94, view_655, lt_45, mul_246, mean, div_71, div_72, permute_267, permute_269, permute_270, div_73, div_74, permute_293, permute_295, permute_296, div_76, div_77, permute_324, permute_326, permute_327, div_78, div_79, permute_350, permute_352, permute_353, div_80, div_81, permute_376, permute_378, permute_379, div_82, div_83, permute_402, permute_404, permute_405, div_84, div_85, permute_428, permute_430, permute_431, div_86, div_87, permute_454, permute_456, permute_457, div_88, div_89, permute_480, permute_482, permute_483, div_90, div_91, permute_506, permute_508, permute_509, div_92, div_93, permute_532, permute_534, permute_535, div_94, div_95, permute_558, permute_560, permute_561, div_96, div_97, permute_584, permute_586, permute_587, div_98, div_99, permute_610, permute_612, permute_613, div_100, div_101, permute_636, permute_638, permute_639, div_102, div_103, permute_662, permute_664, permute_665, div_104, div_105, permute_688, permute_690, permute_691, div_106, div_107, permute_714, permute_716, permute_717, div_108, div_109, permute_740, permute_742, permute_743, div_110, div_111, permute_766, permute_768, permute_769, div_113, div_114, permute_797, permute_799, permute_800, div_115, div_116, permute_823, permute_825, permute_826, div_118, div_119, permute_854, permute_856, permute_857, div_120, div_121, permute_880, permute_882, permute_883)
