class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[8, 512]", primals_2: "i64[1, 512]", primals_3: "f32[128100, 1536]", primals_4: "f32[512, 1536]", primals_5: "f32[1536]", primals_6: "f32[1536]", primals_7: "f32[1536, 1536]", primals_8: "f32[1536]", primals_9: "f32[1536, 1536]", primals_10: "f32[1536]", primals_11: "f32[1536, 1536]", primals_12: "f32[1536]", primals_13: "f32[1536, 1536]", primals_14: "f32[1536]", primals_15: "f32[1536]", primals_16: "f32[1536]", primals_17: "f32[6144, 1536]", primals_18: "f32[6144]", primals_19: "f32[1536, 6144]", primals_20: "f32[1536]", primals_21: "f32[1536]", primals_22: "f32[1536]", primals_23: "f32[1536, 1536]", primals_24: "f32[1536]", primals_25: "f32[1536, 1536]", primals_26: "f32[1536]", primals_27: "f32[1536, 1536]", primals_28: "f32[1536]", primals_29: "f32[1536, 1536]", primals_30: "f32[1536]", primals_31: "f32[1536]", primals_32: "f32[1536]", primals_33: "f32[6144, 1536]", primals_34: "f32[6144]", primals_35: "f32[1536, 6144]", primals_36: "f32[1536]", primals_37: "f32[1536]", primals_38: "f32[1536]", primals_39: "f32[1536, 1536]", primals_40: "f32[1536]", primals_41: "f32[1536, 1536]", primals_42: "f32[1536]", primals_43: "f32[1536, 1536]", primals_44: "f32[1536]", primals_45: "f32[1536, 1536]", primals_46: "f32[1536]", primals_47: "f32[1536]", primals_48: "f32[1536]", primals_49: "f32[6144, 1536]", primals_50: "f32[6144]", primals_51: "f32[1536, 6144]", primals_52: "f32[1536]", primals_53: "f32[1536]", primals_54: "f32[1536]", primals_55: "f32[1536, 1536]", primals_56: "f32[1536]", primals_57: "f32[1536, 1536]", primals_58: "f32[1536]", primals_59: "f32[1536, 1536]", primals_60: "f32[1536]", primals_61: "f32[1536, 1536]", primals_62: "f32[1536]", primals_63: "f32[1536]", primals_64: "f32[1536]", primals_65: "f32[6144, 1536]", primals_66: "f32[6144]", primals_67: "f32[1536, 6144]", primals_68: "f32[1536]", primals_69: "f32[1536]", primals_70: "f32[1536]", primals_71: "f32[1536, 1536]", primals_72: "f32[1536]", primals_73: "f32[1536, 1536]", primals_74: "f32[1536]", primals_75: "f32[1536, 1536]", primals_76: "f32[1536]", primals_77: "f32[1536, 1536]", primals_78: "f32[1536]", primals_79: "f32[1536]", primals_80: "f32[1536]", primals_81: "f32[6144, 1536]", primals_82: "f32[6144]", primals_83: "f32[1536, 6144]", primals_84: "f32[1536]", primals_85: "f32[1536]", primals_86: "f32[1536]", primals_87: "f32[1536, 1536]", primals_88: "f32[1536]", primals_89: "f32[1536, 1536]", primals_90: "f32[1536]", primals_91: "f32[1536, 1536]", primals_92: "f32[1536]", primals_93: "f32[1536, 1536]", primals_94: "f32[1536]", primals_95: "f32[1536]", primals_96: "f32[1536]", primals_97: "f32[6144, 1536]", primals_98: "f32[6144]", primals_99: "f32[1536, 6144]", primals_100: "f32[1536]", primals_101: "f32[1536]", primals_102: "f32[1536]", primals_103: "f32[1536, 1536]", primals_104: "f32[1536]", primals_105: "f32[1536, 1536]", primals_106: "f32[1536]", primals_107: "f32[1536, 1536]", primals_108: "f32[1536]", primals_109: "f32[1536, 1536]", primals_110: "f32[1536]", primals_111: "f32[1536]", primals_112: "f32[1536]", primals_113: "f32[6144, 1536]", primals_114: "f32[6144]", primals_115: "f32[1536, 6144]", primals_116: "f32[1536]", primals_117: "f32[1536]", primals_118: "f32[1536]", primals_119: "f32[1536, 1536]", primals_120: "f32[1536]", primals_121: "f32[1536, 1536]", primals_122: "f32[1536]", primals_123: "f32[1536, 1536]", primals_124: "f32[1536]", primals_125: "f32[1536, 1536]", primals_126: "f32[1536]", primals_127: "f32[1536]", primals_128: "f32[1536]", primals_129: "f32[6144, 1536]", primals_130: "f32[6144]", primals_131: "f32[1536, 6144]", primals_132: "f32[1536]", primals_133: "f32[1536]", primals_134: "f32[1536]", primals_135: "f32[1536, 1536]", primals_136: "f32[1536]", primals_137: "f32[1536, 1536]", primals_138: "f32[1536]", primals_139: "f32[1536, 1536]", primals_140: "f32[1536]", primals_141: "f32[1536, 1536]", primals_142: "f32[1536]", primals_143: "f32[1536]", primals_144: "f32[1536]", primals_145: "f32[6144, 1536]", primals_146: "f32[6144]", primals_147: "f32[1536, 6144]", primals_148: "f32[1536]", primals_149: "f32[1536]", primals_150: "f32[1536]", primals_151: "f32[1536, 1536]", primals_152: "f32[1536]", primals_153: "f32[1536, 1536]", primals_154: "f32[1536]", primals_155: "f32[1536, 1536]", primals_156: "f32[1536]", primals_157: "f32[1536, 1536]", primals_158: "f32[1536]", primals_159: "f32[1536]", primals_160: "f32[1536]", primals_161: "f32[6144, 1536]", primals_162: "f32[6144]", primals_163: "f32[1536, 6144]", primals_164: "f32[1536]", primals_165: "f32[1536]", primals_166: "f32[1536]", primals_167: "f32[1536, 1536]", primals_168: "f32[1536]", primals_169: "f32[1536, 1536]", primals_170: "f32[1536]", primals_171: "f32[1536, 1536]", primals_172: "f32[1536]", primals_173: "f32[1536, 1536]", primals_174: "f32[1536]", primals_175: "f32[1536]", primals_176: "f32[1536]", primals_177: "f32[6144, 1536]", primals_178: "f32[6144]", primals_179: "f32[1536, 6144]", primals_180: "f32[1536]", primals_181: "f32[1536]", primals_182: "f32[1536]", primals_183: "f32[1536, 1536]", primals_184: "f32[1536]", primals_185: "f32[1536, 1536]", primals_186: "f32[1536]", primals_187: "f32[1536, 1536]", primals_188: "f32[1536]", primals_189: "f32[1536, 1536]", primals_190: "f32[1536]", primals_191: "f32[1536]", primals_192: "f32[1536]", primals_193: "f32[6144, 1536]", primals_194: "f32[6144]", primals_195: "f32[1536, 6144]", primals_196: "f32[1536]", primals_197: "f32[1536]", primals_198: "f32[1536]", primals_199: "f32[1536, 1536]", primals_200: "f32[1536]", primals_201: "f32[1536, 1536]", primals_202: "f32[1536]", primals_203: "f32[1536, 1536]", primals_204: "f32[1536]", primals_205: "f32[1536, 1536]", primals_206: "f32[1536]", primals_207: "f32[1536]", primals_208: "f32[1536]", primals_209: "f32[6144, 1536]", primals_210: "f32[6144]", primals_211: "f32[1536, 6144]", primals_212: "f32[1536]", primals_213: "f32[1536]", primals_214: "f32[1536]", primals_215: "f32[1536, 1536]", primals_216: "f32[1536]", primals_217: "f32[1536, 1536]", primals_218: "f32[1536]", primals_219: "f32[1536, 1536]", primals_220: "f32[1536]", primals_221: "f32[1536, 1536]", primals_222: "f32[1536]", primals_223: "f32[1536]", primals_224: "f32[1536]", primals_225: "f32[6144, 1536]", primals_226: "f32[6144]", primals_227: "f32[1536, 6144]", primals_228: "f32[1536]", primals_229: "f32[1536]", primals_230: "f32[1536]", primals_231: "f32[1536, 1536]", primals_232: "f32[1536]", primals_233: "f32[1536, 1536]", primals_234: "f32[1536]", primals_235: "f32[1536, 1536]", primals_236: "f32[1536]", primals_237: "f32[1536, 1536]", primals_238: "f32[1536]", primals_239: "f32[1536]", primals_240: "f32[1536]", primals_241: "f32[6144, 1536]", primals_242: "f32[6144]", primals_243: "f32[1536, 6144]", primals_244: "f32[1536]", primals_245: "f32[1536]", primals_246: "f32[1536]", primals_247: "f32[1536, 1536]", primals_248: "f32[1536]", primals_249: "f32[1536, 1536]", primals_250: "f32[1536]", primals_251: "f32[1536, 1536]", primals_252: "f32[1536]", primals_253: "f32[1536, 1536]", primals_254: "f32[1536]", primals_255: "f32[1536]", primals_256: "f32[1536]", primals_257: "f32[6144, 1536]", primals_258: "f32[6144]", primals_259: "f32[1536, 6144]", primals_260: "f32[1536]", primals_261: "f32[1536]", primals_262: "f32[1536]", primals_263: "f32[1536, 1536]", primals_264: "f32[1536]", primals_265: "f32[1536, 1536]", primals_266: "f32[1536]", primals_267: "f32[1536, 1536]", primals_268: "f32[1536]", primals_269: "f32[1536, 1536]", primals_270: "f32[1536]", primals_271: "f32[1536]", primals_272: "f32[1536]", primals_273: "f32[6144, 1536]", primals_274: "f32[6144]", primals_275: "f32[1536, 6144]", primals_276: "f32[1536]", primals_277: "f32[1536]", primals_278: "f32[1536]", primals_279: "f32[1536, 1536]", primals_280: "f32[1536]", primals_281: "f32[1536, 1536]", primals_282: "f32[1536]", primals_283: "f32[1536, 1536]", primals_284: "f32[1536]", primals_285: "f32[1536, 1536]", primals_286: "f32[1536]", primals_287: "f32[1536]", primals_288: "f32[1536]", primals_289: "f32[6144, 1536]", primals_290: "f32[6144]", primals_291: "f32[1536, 6144]", primals_292: "f32[1536]", primals_293: "f32[1536]", primals_294: "f32[1536]", primals_295: "f32[1536, 1536]", primals_296: "f32[1536]", primals_297: "f32[1536, 1536]", primals_298: "f32[1536]", primals_299: "f32[1536, 1536]", primals_300: "f32[1536]", primals_301: "f32[1536, 1536]", primals_302: "f32[1536]", primals_303: "f32[1536]", primals_304: "f32[1536]", primals_305: "f32[6144, 1536]", primals_306: "f32[6144]", primals_307: "f32[1536, 6144]", primals_308: "f32[1536]", primals_309: "f32[1536]", primals_310: "f32[1536]", primals_311: "f32[1536, 1536]", primals_312: "f32[1536]", primals_313: "f32[1536, 1536]", primals_314: "f32[1536]", primals_315: "f32[1536, 1536]", primals_316: "f32[1536]", primals_317: "f32[1536, 1536]", primals_318: "f32[1536]", primals_319: "f32[1536]", primals_320: "f32[1536]", primals_321: "f32[6144, 1536]", primals_322: "f32[6144]", primals_323: "f32[1536, 6144]", primals_324: "f32[1536]", primals_325: "f32[1536]", primals_326: "f32[1536]", primals_327: "f32[1536, 1536]", primals_328: "f32[1536]", primals_329: "f32[1536, 1536]", primals_330: "f32[1536]", primals_331: "f32[1536, 1536]", primals_332: "f32[1536]", primals_333: "f32[1536, 1536]", primals_334: "f32[1536]", primals_335: "f32[1536]", primals_336: "f32[1536]", primals_337: "f32[6144, 1536]", primals_338: "f32[6144]", primals_339: "f32[1536, 6144]", primals_340: "f32[1536]", primals_341: "f32[1536]", primals_342: "f32[1536]", primals_343: "f32[1536, 1536]", primals_344: "f32[1536]", primals_345: "f32[1536, 1536]", primals_346: "f32[1536]", primals_347: "f32[1536, 1536]", primals_348: "f32[1536]", primals_349: "f32[1536, 1536]", primals_350: "f32[1536]", primals_351: "f32[1536]", primals_352: "f32[1536]", primals_353: "f32[6144, 1536]", primals_354: "f32[6144]", primals_355: "f32[1536, 6144]", primals_356: "f32[1536]", primals_357: "f32[1536]", primals_358: "f32[1536]", primals_359: "f32[1536, 1536]", primals_360: "f32[1536]", primals_361: "f32[1536, 1536]", primals_362: "f32[1536]", primals_363: "f32[1536, 1536]", primals_364: "f32[1536]", primals_365: "f32[1536, 1536]", primals_366: "f32[1536]", primals_367: "f32[1536]", primals_368: "f32[1536]", primals_369: "f32[6144, 1536]", primals_370: "f32[6144]", primals_371: "f32[1536, 6144]", primals_372: "f32[1536]", primals_373: "f32[1536]", primals_374: "f32[1536]", primals_375: "f32[1536, 1536]", primals_376: "f32[1536]", primals_377: "f32[1536, 1536]", primals_378: "f32[1536]", primals_379: "f32[1536, 1536]", primals_380: "f32[1536]", primals_381: "f32[1536, 1536]", primals_382: "f32[1536]", primals_383: "f32[1536]", primals_384: "f32[1536]", primals_385: "f32[6144, 1536]", primals_386: "f32[6144]", primals_387: "f32[1536, 6144]", primals_388: "f32[1536]", primals_389: "f32[1536]", primals_390: "f32[1536]", primals_391: "f32[1536, 1536]", primals_392: "f32[1536]", primals_393: "f32[1536]", primals_394: "f32[1536]", primals_395: "f32[128100]", primals_396: "i64[8, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:535 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding: "f32[8, 512, 1536]" = torch.ops.aten.embedding.default(primals_3, primals_1, 0)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:538 in forward, code: position_embeddings = self.position_embeddings(position_ids.long())
        embedding_1: "f32[1, 512, 1536]" = torch.ops.aten.embedding.default(primals_4, primals_2);  primals_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:544 in forward, code: embeddings = embeddings + position_embeddings
        add: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(embedding, embedding_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:552 in forward, code: embeddings = self.LayerNorm(embeddings)
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 512, 1]" = var_mean[0]
        getitem_1: "f32[8, 512, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-07);  getitem = None
        rsqrt: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add, getitem_1);  add = None
        mul: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul, primals_5);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:561 in forward, code: embeddings = embeddings * mask
        add_2: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_1, primals_6);  mul_1 = primals_6 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[73]" = torch.ops.prims.inductor_seeds.default(73, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:563 in forward, code: embeddings = self.dropout(embeddings)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_72: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_72, 0.1);  inductor_random_default_72 = None
        mul_3: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt, add_2);  add_2 = None
        mul_4: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_3, 1.1111111111111112);  mul_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_4, [4096, 1536])
        permute: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_7, [1, 0])
        addmm: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_8, view, permute);  primals_8 = permute = None
        view_1: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm, [8, 512, 1536]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_2: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_1, [8, 512, 24, -1]);  view_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_1: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        clone: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_3: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone, [-1, 512, 64]);  clone = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_2: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_9, [1, 0])
        addmm_1: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_10, view, permute_2);  primals_10 = permute_2 = None
        view_5: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_1, [8, 512, 1536]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_6: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_5, [8, 512, 24, -1]);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_3: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        clone_1: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
        view_7: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_1, [-1, 512, 64]);  clone_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_4: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_11, [1, 0])
        addmm_2: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_12, view, permute_4);  primals_12 = permute_4 = None
        view_9: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_2, [8, 512, 1536]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_10: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_9, [8, 512, 24, -1]);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_5: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        clone_2: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_11: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_2, [-1, 512, 64]);  clone_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_6: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
        div: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_6, full_default_1);  permute_6 = None
        bmm: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_3, div)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_12: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm, [-1, 24, 512, 512])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        full_default_2: "b8[8, 1, 512, 512]" = torch.ops.aten.full.default([8, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_3, view_12);  view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(where, [-1], True)
        sub_1: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where, amax);  where = None
        exp: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_71: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default([8, 24, 512, 512], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_1: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_71, 0.1);  inductor_random_default_71 = None
        mul_7: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_1, div_1);  div_1 = None
        mul_8: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_7, 1.1111111111111112);  mul_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_13: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(mul_8, [-1, 512, 512]);  mul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_1: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_13, view_11)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_14: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_1, [-1, 24, 512, 64]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_7: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_3: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_15: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_3, [8, 512, -1]);  clone_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_16: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_15, [4096, 1536]);  view_15 = None
        permute_8: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_13, [1, 0])
        addmm_3: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_14, view_16, permute_8);  primals_14 = permute_8 = None
        view_17: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_3, [8, 512, 1536]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_70: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        gt_2: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_70, 0.1);  inductor_random_default_70 = None
        mul_9: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_2, view_17);  view_17 = None
        mul_10: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_9, 1.1111111111111112);  mul_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_3: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_10, mul_4);  mul_10 = mul_4 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
        getitem_2: "f32[8, 512, 1]" = var_mean_1[0]
        getitem_3: "f32[8, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        add_4: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-07);  getitem_2 = None
        rsqrt_1: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_2: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_3, getitem_3);  add_3 = getitem_3 = None
        mul_11: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_12: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_11, primals_15)
        add_5: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_12, primals_16);  mul_12 = primals_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_18: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_5, [4096, 1536])
        permute_9: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_17, [1, 0])
        addmm_4: "f32[4096, 6144]" = torch.ops.aten.addmm.default(primals_18, view_18, permute_9);  primals_18 = permute_9 = None
        view_19: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_4, [8, 512, 6144])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_13: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_19, 0.5)
        mul_14: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476);  view_19 = None
        erf: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_14);  mul_14 = None
        add_6: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_15: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_13, add_6);  mul_13 = add_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_20: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_15, [4096, 6144]);  mul_15 = None
        permute_10: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_19, [1, 0])
        addmm_5: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_20, view_20, permute_10);  primals_20 = permute_10 = None
        view_21: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_5, [8, 512, 1536]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_3: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_69: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        gt_3: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_69, 0.1);  inductor_random_default_69 = None
        mul_16: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_3, view_21);  view_21 = None
        mul_17: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_16, 1.1111111111111112);  mul_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_7: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_17, add_5);  mul_17 = add_5 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
        getitem_4: "f32[8, 512, 1]" = var_mean_2[0]
        getitem_5: "f32[8, 512, 1]" = var_mean_2[1];  var_mean_2 = None
        add_8: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-07);  getitem_4 = None
        rsqrt_2: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_3: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_7, getitem_5);  add_7 = getitem_5 = None
        mul_18: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_19: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_18, primals_21)
        add_9: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_19, primals_22);  mul_19 = primals_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_22: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_9, [4096, 1536])
        permute_11: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_23, [1, 0])
        addmm_6: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_24, view_22, permute_11);  primals_24 = permute_11 = None
        view_23: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_6, [8, 512, 1536]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_24: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_23, [8, 512, 24, -1]);  view_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_12: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        clone_4: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None
        view_25: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_4, [-1, 512, 64]);  clone_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_13: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_25, [1, 0])
        addmm_7: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_26, view_22, permute_13);  primals_26 = permute_13 = None
        view_27: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_7, [8, 512, 1536]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_28: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_27, [8, 512, 24, -1]);  view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_14: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        clone_5: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
        view_29: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_5, [-1, 512, 64]);  clone_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_15: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_27, [1, 0])
        addmm_8: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_28, view_22, permute_15);  primals_28 = permute_15 = None
        view_31: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_8, [8, 512, 1536]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_32: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_31, [8, 512, 24, -1]);  view_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_16: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
        clone_6: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        view_33: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_6, [-1, 512, 64]);  clone_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_17: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_29, [0, 2, 1]);  view_29 = None
        div_2: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_17, full_default_1);  permute_17 = None
        bmm_2: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_25, div_2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_34: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_2, [-1, 24, 512, 512]);  bmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_1: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_3, view_34);  view_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax_1: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(where_1, [-1], True)
        sub_4: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_1, amax_1);  where_1 = amax_1 = None
        exp_1: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_3: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        inductor_lookup_seed_default_4: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_random_default_68: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default([8, 24, 512, 512], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        gt_4: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_68, 0.1);  inductor_random_default_68 = None
        mul_21: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_4, div_3)
        mul_22: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_21, 1.1111111111111112);  mul_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_35: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(mul_22, [-1, 512, 512]);  mul_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_3: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_35, view_33)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_36: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_3, [-1, 24, 512, 64]);  bmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_18: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_7: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_37: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_7, [8, 512, -1]);  clone_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_38: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_37, [4096, 1536]);  view_37 = None
        permute_19: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_29, [1, 0])
        addmm_9: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_30, view_38, permute_19);  primals_30 = permute_19 = None
        view_39: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_9, [8, 512, 1536]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_5: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_random_default_67: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_5, 'rand');  inductor_lookup_seed_default_5 = None
        gt_5: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_67, 0.1);  inductor_random_default_67 = None
        mul_23: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_5, view_39);  view_39 = None
        mul_24: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_23, 1.1111111111111112);  mul_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_10: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_24, add_9);  mul_24 = add_9 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
        getitem_6: "f32[8, 512, 1]" = var_mean_3[0]
        getitem_7: "f32[8, 512, 1]" = var_mean_3[1];  var_mean_3 = None
        add_11: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-07);  getitem_6 = None
        rsqrt_3: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_5: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_10, getitem_7);  add_10 = getitem_7 = None
        mul_25: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
        mul_26: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_25, primals_31)
        add_12: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_26, primals_32);  mul_26 = primals_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_40: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_12, [4096, 1536])
        permute_20: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_33, [1, 0])
        addmm_10: "f32[4096, 6144]" = torch.ops.aten.addmm.default(primals_34, view_40, permute_20);  primals_34 = permute_20 = None
        view_41: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_10, [8, 512, 6144])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_27: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_41, 0.5)
        mul_28: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476);  view_41 = None
        erf_1: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
        add_13: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_29: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_27, add_13);  mul_27 = add_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_42: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_29, [4096, 6144]);  mul_29 = None
        permute_21: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_35, [1, 0])
        addmm_11: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_36, view_42, permute_21);  primals_36 = permute_21 = None
        view_43: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_11, [8, 512, 1536]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_6: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6)
        inductor_random_default_66: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_6, 'rand');  inductor_lookup_seed_default_6 = None
        gt_6: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_66, 0.1);  inductor_random_default_66 = None
        mul_30: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_6, view_43);  view_43 = None
        mul_31: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_30, 1.1111111111111112);  mul_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_14: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_31, add_12);  mul_31 = add_12 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
        getitem_8: "f32[8, 512, 1]" = var_mean_4[0]
        getitem_9: "f32[8, 512, 1]" = var_mean_4[1];  var_mean_4 = None
        add_15: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-07);  getitem_8 = None
        rsqrt_4: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        sub_6: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_14, getitem_9);  add_14 = getitem_9 = None
        mul_32: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
        mul_33: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_32, primals_37)
        add_16: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_33, primals_38);  mul_33 = primals_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_44: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_16, [4096, 1536])
        permute_22: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_39, [1, 0])
        addmm_12: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_40, view_44, permute_22);  primals_40 = permute_22 = None
        view_45: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_12, [8, 512, 1536]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_46: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_45, [8, 512, 24, -1]);  view_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_23: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
        clone_8: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
        view_47: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_8, [-1, 512, 64]);  clone_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_24: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_41, [1, 0])
        addmm_13: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_42, view_44, permute_24);  primals_42 = permute_24 = None
        view_49: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_13, [8, 512, 1536]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_50: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_49, [8, 512, 24, -1]);  view_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_25: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        clone_9: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_25, memory_format = torch.contiguous_format);  permute_25 = None
        view_51: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_9, [-1, 512, 64]);  clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_26: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_43, [1, 0])
        addmm_14: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_44, view_44, permute_26);  primals_44 = permute_26 = None
        view_53: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_14, [8, 512, 1536]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_54: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_53, [8, 512, 24, -1]);  view_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_27: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
        clone_10: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
        view_55: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_10, [-1, 512, 64]);  clone_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_28: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_51, [0, 2, 1]);  view_51 = None
        div_4: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_28, full_default_1);  permute_28 = None
        bmm_4: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_47, div_4)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_56: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_4, [-1, 24, 512, 512]);  bmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_2: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_3, view_56);  view_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax_2: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(where_2, [-1], True)
        sub_7: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_2, amax_2);  where_2 = amax_2 = None
        exp_2: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_5: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        inductor_lookup_seed_default_7: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 7)
        inductor_random_default_65: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default([8, 24, 512, 512], inductor_lookup_seed_default_7, 'rand');  inductor_lookup_seed_default_7 = None
        gt_7: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_65, 0.1);  inductor_random_default_65 = None
        mul_35: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_7, div_5)
        mul_36: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_35, 1.1111111111111112);  mul_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_57: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(mul_36, [-1, 512, 512]);  mul_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_5: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_57, view_55)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_58: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_5, [-1, 24, 512, 64]);  bmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_29: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_11: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_59: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_11, [8, 512, -1]);  clone_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_60: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_59, [4096, 1536]);  view_59 = None
        permute_30: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_45, [1, 0])
        addmm_15: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_46, view_60, permute_30);  primals_46 = permute_30 = None
        view_61: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_15, [8, 512, 1536]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_8: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 8)
        inductor_random_default_64: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_8, 'rand');  inductor_lookup_seed_default_8 = None
        gt_8: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_64, 0.1);  inductor_random_default_64 = None
        mul_37: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_8, view_61);  view_61 = None
        mul_38: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_37, 1.1111111111111112);  mul_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_17: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_38, add_16);  mul_38 = add_16 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_10: "f32[8, 512, 1]" = var_mean_5[0]
        getitem_11: "f32[8, 512, 1]" = var_mean_5[1];  var_mean_5 = None
        add_18: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-07);  getitem_10 = None
        rsqrt_5: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_8: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_17, getitem_11);  add_17 = getitem_11 = None
        mul_39: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = None
        mul_40: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_39, primals_47)
        add_19: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_40, primals_48);  mul_40 = primals_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_62: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_19, [4096, 1536])
        permute_31: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_49, [1, 0])
        addmm_16: "f32[4096, 6144]" = torch.ops.aten.addmm.default(primals_50, view_62, permute_31);  primals_50 = permute_31 = None
        view_63: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_16, [8, 512, 6144])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_41: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_63, 0.5)
        mul_42: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476);  view_63 = None
        erf_2: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_42);  mul_42 = None
        add_20: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_43: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_41, add_20);  mul_41 = add_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_64: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_43, [4096, 6144]);  mul_43 = None
        permute_32: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_51, [1, 0])
        addmm_17: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_52, view_64, permute_32);  primals_52 = permute_32 = None
        view_65: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_17, [8, 512, 1536]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_9: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 9)
        inductor_random_default_63: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_9, 'rand');  inductor_lookup_seed_default_9 = None
        gt_9: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_63, 0.1);  inductor_random_default_63 = None
        mul_44: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_9, view_65);  view_65 = None
        mul_45: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_44, 1.1111111111111112);  mul_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_21: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_45, add_19);  mul_45 = add_19 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
        getitem_12: "f32[8, 512, 1]" = var_mean_6[0]
        getitem_13: "f32[8, 512, 1]" = var_mean_6[1];  var_mean_6 = None
        add_22: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-07);  getitem_12 = None
        rsqrt_6: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        sub_9: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_21, getitem_13);  add_21 = getitem_13 = None
        mul_46: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = None
        mul_47: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_46, primals_53)
        add_23: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_47, primals_54);  mul_47 = primals_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_66: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_23, [4096, 1536])
        permute_33: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_55, [1, 0])
        addmm_18: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_56, view_66, permute_33);  primals_56 = permute_33 = None
        view_67: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_18, [8, 512, 1536]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_68: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_67, [8, 512, 24, -1]);  view_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_34: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
        clone_12: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
        view_69: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_12, [-1, 512, 64]);  clone_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_35: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_57, [1, 0])
        addmm_19: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_58, view_66, permute_35);  primals_58 = permute_35 = None
        view_71: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_19, [8, 512, 1536]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_72: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_71, [8, 512, 24, -1]);  view_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_36: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        clone_13: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_36, memory_format = torch.contiguous_format);  permute_36 = None
        view_73: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_13, [-1, 512, 64]);  clone_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_37: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_59, [1, 0])
        addmm_20: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_60, view_66, permute_37);  primals_60 = permute_37 = None
        view_75: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_20, [8, 512, 1536]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_76: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_75, [8, 512, 24, -1]);  view_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_38: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
        clone_14: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
        view_77: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_14, [-1, 512, 64]);  clone_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_39: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_73, [0, 2, 1]);  view_73 = None
        div_6: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_39, full_default_1);  permute_39 = None
        bmm_6: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_69, div_6)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_78: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_6, [-1, 24, 512, 512]);  bmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_3: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_3, view_78);  view_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax_3: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(where_3, [-1], True)
        sub_10: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_3, amax_3);  where_3 = amax_3 = None
        exp_3: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_7: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        inductor_lookup_seed_default_10: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 10)
        inductor_random_default_62: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default([8, 24, 512, 512], inductor_lookup_seed_default_10, 'rand');  inductor_lookup_seed_default_10 = None
        gt_10: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_62, 0.1);  inductor_random_default_62 = None
        mul_49: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_10, div_7)
        mul_50: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_49, 1.1111111111111112);  mul_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_79: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(mul_50, [-1, 512, 512]);  mul_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_7: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_79, view_77)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_80: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_7, [-1, 24, 512, 64]);  bmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_40: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_15: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_81: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_15, [8, 512, -1]);  clone_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_82: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_81, [4096, 1536]);  view_81 = None
        permute_41: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_61, [1, 0])
        addmm_21: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_62, view_82, permute_41);  primals_62 = permute_41 = None
        view_83: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_21, [8, 512, 1536]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_11: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 11)
        inductor_random_default_61: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_11, 'rand');  inductor_lookup_seed_default_11 = None
        gt_11: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_61, 0.1);  inductor_random_default_61 = None
        mul_51: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_11, view_83);  view_83 = None
        mul_52: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_51, 1.1111111111111112);  mul_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_24: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_52, add_23);  mul_52 = add_23 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
        getitem_14: "f32[8, 512, 1]" = var_mean_7[0]
        getitem_15: "f32[8, 512, 1]" = var_mean_7[1];  var_mean_7 = None
        add_25: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-07);  getitem_14 = None
        rsqrt_7: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        sub_11: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_24, getitem_15);  add_24 = getitem_15 = None
        mul_53: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = None
        mul_54: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_53, primals_63)
        add_26: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_54, primals_64);  mul_54 = primals_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_84: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_26, [4096, 1536])
        permute_42: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_65, [1, 0])
        addmm_22: "f32[4096, 6144]" = torch.ops.aten.addmm.default(primals_66, view_84, permute_42);  primals_66 = permute_42 = None
        view_85: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_22, [8, 512, 6144])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_55: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_85, 0.5)
        mul_56: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476);  view_85 = None
        erf_3: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_56);  mul_56 = None
        add_27: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_57: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_55, add_27);  mul_55 = add_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_86: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_57, [4096, 6144]);  mul_57 = None
        permute_43: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_67, [1, 0])
        addmm_23: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_68, view_86, permute_43);  primals_68 = permute_43 = None
        view_87: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_23, [8, 512, 1536]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_12: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 12)
        inductor_random_default_60: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_12, 'rand');  inductor_lookup_seed_default_12 = None
        gt_12: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_60, 0.1);  inductor_random_default_60 = None
        mul_58: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_12, view_87);  view_87 = None
        mul_59: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_58, 1.1111111111111112);  mul_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_28: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_59, add_26);  mul_59 = add_26 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
        getitem_16: "f32[8, 512, 1]" = var_mean_8[0]
        getitem_17: "f32[8, 512, 1]" = var_mean_8[1];  var_mean_8 = None
        add_29: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-07);  getitem_16 = None
        rsqrt_8: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        sub_12: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_28, getitem_17);  add_28 = getitem_17 = None
        mul_60: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = None
        mul_61: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_60, primals_69)
        add_30: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_61, primals_70);  mul_61 = primals_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_88: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_30, [4096, 1536])
        permute_44: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_71, [1, 0])
        addmm_24: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_72, view_88, permute_44);  primals_72 = permute_44 = None
        view_89: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_24, [8, 512, 1536]);  addmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_90: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_89, [8, 512, 24, -1]);  view_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_45: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None
        clone_16: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
        view_91: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_16, [-1, 512, 64]);  clone_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_46: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_73, [1, 0])
        addmm_25: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_74, view_88, permute_46);  primals_74 = permute_46 = None
        view_93: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_25, [8, 512, 1536]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_94: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_93, [8, 512, 24, -1]);  view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_47: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
        clone_17: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
        view_95: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_17, [-1, 512, 64]);  clone_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_48: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_75, [1, 0])
        addmm_26: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_76, view_88, permute_48);  primals_76 = permute_48 = None
        view_97: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_26, [8, 512, 1536]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_98: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_97, [8, 512, 24, -1]);  view_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_49: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
        clone_18: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
        view_99: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_18, [-1, 512, 64]);  clone_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_50: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_95, [0, 2, 1]);  view_95 = None
        div_8: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_50, full_default_1);  permute_50 = None
        bmm_8: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_91, div_8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_100: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_8, [-1, 24, 512, 512]);  bmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_4: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_3, view_100);  view_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax_4: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(where_4, [-1], True)
        sub_13: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_4, amax_4);  where_4 = amax_4 = None
        exp_4: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_9: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        inductor_lookup_seed_default_13: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 13)
        inductor_random_default_59: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default([8, 24, 512, 512], inductor_lookup_seed_default_13, 'rand');  inductor_lookup_seed_default_13 = None
        gt_13: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_59, 0.1);  inductor_random_default_59 = None
        mul_63: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_13, div_9)
        mul_64: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_63, 1.1111111111111112);  mul_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_101: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(mul_64, [-1, 512, 512]);  mul_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_9: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_101, view_99)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_102: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_9, [-1, 24, 512, 64]);  bmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_51: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_19: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_103: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_19, [8, 512, -1]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_104: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_103, [4096, 1536]);  view_103 = None
        permute_52: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_77, [1, 0])
        addmm_27: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_78, view_104, permute_52);  primals_78 = permute_52 = None
        view_105: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_27, [8, 512, 1536]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_14: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 14)
        inductor_random_default_58: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_14, 'rand');  inductor_lookup_seed_default_14 = None
        gt_14: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_58, 0.1);  inductor_random_default_58 = None
        mul_65: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_14, view_105);  view_105 = None
        mul_66: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_65, 1.1111111111111112);  mul_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_31: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_66, add_30);  mul_66 = add_30 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
        getitem_18: "f32[8, 512, 1]" = var_mean_9[0]
        getitem_19: "f32[8, 512, 1]" = var_mean_9[1];  var_mean_9 = None
        add_32: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-07);  getitem_18 = None
        rsqrt_9: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        sub_14: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_31, getitem_19);  add_31 = getitem_19 = None
        mul_67: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = None
        mul_68: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_67, primals_79)
        add_33: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_68, primals_80);  mul_68 = primals_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_106: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_33, [4096, 1536])
        permute_53: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_81, [1, 0])
        addmm_28: "f32[4096, 6144]" = torch.ops.aten.addmm.default(primals_82, view_106, permute_53);  primals_82 = permute_53 = None
        view_107: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_28, [8, 512, 6144])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_69: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_107, 0.5)
        mul_70: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476);  view_107 = None
        erf_4: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_70);  mul_70 = None
        add_34: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_71: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_69, add_34);  mul_69 = add_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_108: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_71, [4096, 6144]);  mul_71 = None
        permute_54: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_83, [1, 0])
        addmm_29: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_84, view_108, permute_54);  primals_84 = permute_54 = None
        view_109: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_29, [8, 512, 1536]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_15: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 15)
        inductor_random_default_57: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_15, 'rand');  inductor_lookup_seed_default_15 = None
        gt_15: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_57, 0.1);  inductor_random_default_57 = None
        mul_72: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_15, view_109);  view_109 = None
        mul_73: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_72, 1.1111111111111112);  mul_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_35: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_73, add_33);  mul_73 = add_33 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_20: "f32[8, 512, 1]" = var_mean_10[0]
        getitem_21: "f32[8, 512, 1]" = var_mean_10[1];  var_mean_10 = None
        add_36: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-07);  getitem_20 = None
        rsqrt_10: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_15: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_35, getitem_21);  add_35 = getitem_21 = None
        mul_74: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = None
        mul_75: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_74, primals_85)
        add_37: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_75, primals_86);  mul_75 = primals_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_110: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_37, [4096, 1536])
        permute_55: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_87, [1, 0])
        addmm_30: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_88, view_110, permute_55);  primals_88 = permute_55 = None
        view_111: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_30, [8, 512, 1536]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_112: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_111, [8, 512, 24, -1]);  view_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_56: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
        clone_20: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None
        view_113: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_20, [-1, 512, 64]);  clone_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_57: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_89, [1, 0])
        addmm_31: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_90, view_110, permute_57);  primals_90 = permute_57 = None
        view_115: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_31, [8, 512, 1536]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_116: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_115, [8, 512, 24, -1]);  view_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_58: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
        clone_21: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_58, memory_format = torch.contiguous_format);  permute_58 = None
        view_117: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_21, [-1, 512, 64]);  clone_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_59: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_91, [1, 0])
        addmm_32: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_92, view_110, permute_59);  primals_92 = permute_59 = None
        view_119: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_32, [8, 512, 1536]);  addmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_120: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_119, [8, 512, 24, -1]);  view_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_60: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
        clone_22: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
        view_121: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_22, [-1, 512, 64]);  clone_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_61: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_117, [0, 2, 1]);  view_117 = None
        div_10: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_61, full_default_1);  permute_61 = None
        bmm_10: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_113, div_10)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_122: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_10, [-1, 24, 512, 512]);  bmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_5: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_3, view_122);  view_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax_5: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(where_5, [-1], True)
        sub_16: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_5, amax_5);  where_5 = amax_5 = None
        exp_5: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_11: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        inductor_lookup_seed_default_16: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 16)
        inductor_random_default_56: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default([8, 24, 512, 512], inductor_lookup_seed_default_16, 'rand');  inductor_lookup_seed_default_16 = None
        gt_16: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_56, 0.1);  inductor_random_default_56 = None
        mul_77: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_16, div_11)
        mul_78: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_77, 1.1111111111111112);  mul_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_123: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(mul_78, [-1, 512, 512]);  mul_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_11: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_123, view_121)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_124: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_11, [-1, 24, 512, 64]);  bmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_62: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_23: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_125: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_23, [8, 512, -1]);  clone_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_126: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_125, [4096, 1536]);  view_125 = None
        permute_63: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_93, [1, 0])
        addmm_33: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_94, view_126, permute_63);  primals_94 = permute_63 = None
        view_127: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_33, [8, 512, 1536]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_17: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 17)
        inductor_random_default_55: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_17, 'rand');  inductor_lookup_seed_default_17 = None
        gt_17: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_55, 0.1);  inductor_random_default_55 = None
        mul_79: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_17, view_127);  view_127 = None
        mul_80: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_79, 1.1111111111111112);  mul_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_38: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_80, add_37);  mul_80 = add_37 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
        getitem_22: "f32[8, 512, 1]" = var_mean_11[0]
        getitem_23: "f32[8, 512, 1]" = var_mean_11[1];  var_mean_11 = None
        add_39: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-07);  getitem_22 = None
        rsqrt_11: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        sub_17: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_38, getitem_23);  add_38 = getitem_23 = None
        mul_81: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = None
        mul_82: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_81, primals_95)
        add_40: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_82, primals_96);  mul_82 = primals_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_128: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_40, [4096, 1536])
        permute_64: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_97, [1, 0])
        addmm_34: "f32[4096, 6144]" = torch.ops.aten.addmm.default(primals_98, view_128, permute_64);  primals_98 = permute_64 = None
        view_129: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_34, [8, 512, 6144])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_83: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_129, 0.5)
        mul_84: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476);  view_129 = None
        erf_5: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_84);  mul_84 = None
        add_41: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_85: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_83, add_41);  mul_83 = add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_130: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_85, [4096, 6144]);  mul_85 = None
        permute_65: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_99, [1, 0])
        addmm_35: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_100, view_130, permute_65);  primals_100 = permute_65 = None
        view_131: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_35, [8, 512, 1536]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_18: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 18)
        inductor_random_default_54: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_18, 'rand');  inductor_lookup_seed_default_18 = None
        gt_18: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_54, 0.1);  inductor_random_default_54 = None
        mul_86: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_18, view_131);  view_131 = None
        mul_87: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_86, 1.1111111111111112);  mul_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_42: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_87, add_40);  mul_87 = add_40 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_42, [2], correction = 0, keepdim = True)
        getitem_24: "f32[8, 512, 1]" = var_mean_12[0]
        getitem_25: "f32[8, 512, 1]" = var_mean_12[1];  var_mean_12 = None
        add_43: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-07);  getitem_24 = None
        rsqrt_12: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        sub_18: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_42, getitem_25);  add_42 = getitem_25 = None
        mul_88: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = None
        mul_89: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_88, primals_101)
        add_44: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_89, primals_102);  mul_89 = primals_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_132: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_44, [4096, 1536])
        permute_66: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_103, [1, 0])
        addmm_36: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_104, view_132, permute_66);  primals_104 = permute_66 = None
        view_133: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_36, [8, 512, 1536]);  addmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_134: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_133, [8, 512, 24, -1]);  view_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_67: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None
        clone_24: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None
        view_135: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_24, [-1, 512, 64]);  clone_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_68: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_105, [1, 0])
        addmm_37: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_106, view_132, permute_68);  primals_106 = permute_68 = None
        view_137: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_37, [8, 512, 1536]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_138: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_137, [8, 512, 24, -1]);  view_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_69: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
        clone_25: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_69, memory_format = torch.contiguous_format);  permute_69 = None
        view_139: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_25, [-1, 512, 64]);  clone_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_70: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_107, [1, 0])
        addmm_38: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_108, view_132, permute_70);  primals_108 = permute_70 = None
        view_141: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_38, [8, 512, 1536]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_142: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_141, [8, 512, 24, -1]);  view_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_71: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None
        clone_26: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
        view_143: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_26, [-1, 512, 64]);  clone_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_72: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_139, [0, 2, 1]);  view_139 = None
        div_12: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_72, full_default_1);  permute_72 = None
        bmm_12: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_135, div_12)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_144: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_12, [-1, 24, 512, 512]);  bmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_6: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_3, view_144);  view_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax_6: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(where_6, [-1], True)
        sub_19: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_6, amax_6);  where_6 = amax_6 = None
        exp_6: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_7: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_13: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        inductor_lookup_seed_default_19: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 19)
        inductor_random_default_53: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default([8, 24, 512, 512], inductor_lookup_seed_default_19, 'rand');  inductor_lookup_seed_default_19 = None
        gt_19: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_53, 0.1);  inductor_random_default_53 = None
        mul_91: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_19, div_13)
        mul_92: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_91, 1.1111111111111112);  mul_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_145: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(mul_92, [-1, 512, 512]);  mul_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_13: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_145, view_143)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_146: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_13, [-1, 24, 512, 64]);  bmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_73: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_27: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_147: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_27, [8, 512, -1]);  clone_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_148: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_147, [4096, 1536]);  view_147 = None
        permute_74: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_109, [1, 0])
        addmm_39: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_110, view_148, permute_74);  primals_110 = permute_74 = None
        view_149: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_39, [8, 512, 1536]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_20: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 20)
        inductor_random_default_52: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_20, 'rand');  inductor_lookup_seed_default_20 = None
        gt_20: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_52, 0.1);  inductor_random_default_52 = None
        mul_93: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_20, view_149);  view_149 = None
        mul_94: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_93, 1.1111111111111112);  mul_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_45: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_94, add_44);  mul_94 = add_44 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
        getitem_26: "f32[8, 512, 1]" = var_mean_13[0]
        getitem_27: "f32[8, 512, 1]" = var_mean_13[1];  var_mean_13 = None
        add_46: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-07);  getitem_26 = None
        rsqrt_13: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_20: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_45, getitem_27);  add_45 = getitem_27 = None
        mul_95: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = None
        mul_96: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_95, primals_111)
        add_47: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_96, primals_112);  mul_96 = primals_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_150: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_47, [4096, 1536])
        permute_75: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_113, [1, 0])
        addmm_40: "f32[4096, 6144]" = torch.ops.aten.addmm.default(primals_114, view_150, permute_75);  primals_114 = permute_75 = None
        view_151: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_40, [8, 512, 6144])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_97: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_151, 0.5)
        mul_98: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476);  view_151 = None
        erf_6: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_98);  mul_98 = None
        add_48: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_99: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_97, add_48);  mul_97 = add_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_152: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_99, [4096, 6144]);  mul_99 = None
        permute_76: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_115, [1, 0])
        addmm_41: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_116, view_152, permute_76);  primals_116 = permute_76 = None
        view_153: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_41, [8, 512, 1536]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_21: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 21)
        inductor_random_default_51: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_21, 'rand');  inductor_lookup_seed_default_21 = None
        gt_21: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_51, 0.1);  inductor_random_default_51 = None
        mul_100: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_21, view_153);  view_153 = None
        mul_101: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_100, 1.1111111111111112);  mul_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_49: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_101, add_47);  mul_101 = add_47 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
        getitem_28: "f32[8, 512, 1]" = var_mean_14[0]
        getitem_29: "f32[8, 512, 1]" = var_mean_14[1];  var_mean_14 = None
        add_50: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-07);  getitem_28 = None
        rsqrt_14: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_21: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_49, getitem_29);  add_49 = getitem_29 = None
        mul_102: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = None
        mul_103: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_102, primals_117)
        add_51: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_103, primals_118);  mul_103 = primals_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_154: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_51, [4096, 1536])
        permute_77: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_119, [1, 0])
        addmm_42: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_120, view_154, permute_77);  primals_120 = permute_77 = None
        view_155: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_42, [8, 512, 1536]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_156: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_155, [8, 512, 24, -1]);  view_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_78: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None
        clone_28: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_78, memory_format = torch.contiguous_format);  permute_78 = None
        view_157: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_28, [-1, 512, 64]);  clone_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_79: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_121, [1, 0])
        addmm_43: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_122, view_154, permute_79);  primals_122 = permute_79 = None
        view_159: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_43, [8, 512, 1536]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_160: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_159, [8, 512, 24, -1]);  view_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_80: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None
        clone_29: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
        view_161: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_29, [-1, 512, 64]);  clone_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_81: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_123, [1, 0])
        addmm_44: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_124, view_154, permute_81);  primals_124 = permute_81 = None
        view_163: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_44, [8, 512, 1536]);  addmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_164: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_163, [8, 512, 24, -1]);  view_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_82: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
        clone_30: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        view_165: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_30, [-1, 512, 64]);  clone_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_83: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_161, [0, 2, 1]);  view_161 = None
        div_14: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_83, full_default_1);  permute_83 = None
        bmm_14: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_157, div_14)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_166: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_14, [-1, 24, 512, 512]);  bmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_7: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_3, view_166);  view_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax_7: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(where_7, [-1], True)
        sub_22: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_7, amax_7);  where_7 = amax_7 = None
        exp_7: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_8: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_15: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        inductor_lookup_seed_default_22: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 22)
        inductor_random_default_50: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default([8, 24, 512, 512], inductor_lookup_seed_default_22, 'rand');  inductor_lookup_seed_default_22 = None
        gt_22: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_50, 0.1);  inductor_random_default_50 = None
        mul_105: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_22, div_15)
        mul_106: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_105, 1.1111111111111112);  mul_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_167: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(mul_106, [-1, 512, 512]);  mul_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_15: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_167, view_165)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_168: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_15, [-1, 24, 512, 64]);  bmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_84: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_31: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_169: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_31, [8, 512, -1]);  clone_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_170: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_169, [4096, 1536]);  view_169 = None
        permute_85: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_125, [1, 0])
        addmm_45: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_126, view_170, permute_85);  primals_126 = permute_85 = None
        view_171: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_45, [8, 512, 1536]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_23: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 23)
        inductor_random_default_49: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_23, 'rand');  inductor_lookup_seed_default_23 = None
        gt_23: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_49, 0.1);  inductor_random_default_49 = None
        mul_107: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_23, view_171);  view_171 = None
        mul_108: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_107, 1.1111111111111112);  mul_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_52: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_108, add_51);  mul_108 = add_51 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
        getitem_30: "f32[8, 512, 1]" = var_mean_15[0]
        getitem_31: "f32[8, 512, 1]" = var_mean_15[1];  var_mean_15 = None
        add_53: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-07);  getitem_30 = None
        rsqrt_15: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_23: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_52, getitem_31);  add_52 = getitem_31 = None
        mul_109: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = None
        mul_110: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_109, primals_127)
        add_54: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_110, primals_128);  mul_110 = primals_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_172: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_54, [4096, 1536])
        permute_86: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_129, [1, 0])
        addmm_46: "f32[4096, 6144]" = torch.ops.aten.addmm.default(primals_130, view_172, permute_86);  primals_130 = permute_86 = None
        view_173: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_46, [8, 512, 6144])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_111: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_173, 0.5)
        mul_112: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476);  view_173 = None
        erf_7: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_112);  mul_112 = None
        add_55: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_113: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_111, add_55);  mul_111 = add_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_174: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_113, [4096, 6144]);  mul_113 = None
        permute_87: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_131, [1, 0])
        addmm_47: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_132, view_174, permute_87);  primals_132 = permute_87 = None
        view_175: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_47, [8, 512, 1536]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_24: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 24)
        inductor_random_default_48: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_24, 'rand');  inductor_lookup_seed_default_24 = None
        gt_24: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_48, 0.1);  inductor_random_default_48 = None
        mul_114: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_24, view_175);  view_175 = None
        mul_115: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_114, 1.1111111111111112);  mul_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_56: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_115, add_54);  mul_115 = add_54 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_56, [2], correction = 0, keepdim = True)
        getitem_32: "f32[8, 512, 1]" = var_mean_16[0]
        getitem_33: "f32[8, 512, 1]" = var_mean_16[1];  var_mean_16 = None
        add_57: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-07);  getitem_32 = None
        rsqrt_16: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
        sub_24: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_56, getitem_33);  add_56 = getitem_33 = None
        mul_116: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = None
        mul_117: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_116, primals_133)
        add_58: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_117, primals_134);  mul_117 = primals_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_176: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_58, [4096, 1536])
        permute_88: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_135, [1, 0])
        addmm_48: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_136, view_176, permute_88);  primals_136 = permute_88 = None
        view_177: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_48, [8, 512, 1536]);  addmm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_178: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_177, [8, 512, 24, -1]);  view_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_89: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None
        clone_32: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_89, memory_format = torch.contiguous_format);  permute_89 = None
        view_179: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_32, [-1, 512, 64]);  clone_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_90: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_137, [1, 0])
        addmm_49: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_138, view_176, permute_90);  primals_138 = permute_90 = None
        view_181: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_49, [8, 512, 1536]);  addmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_182: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_181, [8, 512, 24, -1]);  view_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_91: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
        clone_33: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None
        view_183: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_33, [-1, 512, 64]);  clone_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_92: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_139, [1, 0])
        addmm_50: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_140, view_176, permute_92);  primals_140 = permute_92 = None
        view_185: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_50, [8, 512, 1536]);  addmm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_186: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_185, [8, 512, 24, -1]);  view_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_93: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
        clone_34: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
        view_187: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_34, [-1, 512, 64]);  clone_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_94: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_183, [0, 2, 1]);  view_183 = None
        div_16: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_94, full_default_1);  permute_94 = None
        bmm_16: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_179, div_16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_188: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_16, [-1, 24, 512, 512]);  bmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_8: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_3, view_188);  view_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax_8: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(where_8, [-1], True)
        sub_25: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_8, amax_8);  where_8 = amax_8 = None
        exp_8: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_9: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_17: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        inductor_lookup_seed_default_25: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 25)
        inductor_random_default_47: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default([8, 24, 512, 512], inductor_lookup_seed_default_25, 'rand');  inductor_lookup_seed_default_25 = None
        gt_25: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_47, 0.1);  inductor_random_default_47 = None
        mul_119: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_25, div_17)
        mul_120: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_119, 1.1111111111111112);  mul_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_189: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(mul_120, [-1, 512, 512]);  mul_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_17: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_189, view_187)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_190: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_17, [-1, 24, 512, 64]);  bmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_95: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_35: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_191: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_35, [8, 512, -1]);  clone_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_192: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_191, [4096, 1536]);  view_191 = None
        permute_96: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_141, [1, 0])
        addmm_51: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_142, view_192, permute_96);  primals_142 = permute_96 = None
        view_193: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_51, [8, 512, 1536]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_26: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 26)
        inductor_random_default_46: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_26, 'rand');  inductor_lookup_seed_default_26 = None
        gt_26: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_46, 0.1);  inductor_random_default_46 = None
        mul_121: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_26, view_193);  view_193 = None
        mul_122: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_121, 1.1111111111111112);  mul_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_59: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_122, add_58);  mul_122 = add_58 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
        getitem_34: "f32[8, 512, 1]" = var_mean_17[0]
        getitem_35: "f32[8, 512, 1]" = var_mean_17[1];  var_mean_17 = None
        add_60: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-07);  getitem_34 = None
        rsqrt_17: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_26: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_59, getitem_35);  add_59 = getitem_35 = None
        mul_123: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = None
        mul_124: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_123, primals_143)
        add_61: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_124, primals_144);  mul_124 = primals_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_194: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_61, [4096, 1536])
        permute_97: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_145, [1, 0])
        addmm_52: "f32[4096, 6144]" = torch.ops.aten.addmm.default(primals_146, view_194, permute_97);  primals_146 = permute_97 = None
        view_195: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_52, [8, 512, 6144])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_125: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_195, 0.5)
        mul_126: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476);  view_195 = None
        erf_8: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_126);  mul_126 = None
        add_62: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_127: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_125, add_62);  mul_125 = add_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_196: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_127, [4096, 6144]);  mul_127 = None
        permute_98: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_147, [1, 0])
        addmm_53: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_148, view_196, permute_98);  primals_148 = permute_98 = None
        view_197: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_53, [8, 512, 1536]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_27: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 27)
        inductor_random_default_45: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_27, 'rand');  inductor_lookup_seed_default_27 = None
        gt_27: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_45, 0.1);  inductor_random_default_45 = None
        mul_128: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_27, view_197);  view_197 = None
        mul_129: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_128, 1.1111111111111112);  mul_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_63: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_129, add_61);  mul_129 = add_61 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
        getitem_36: "f32[8, 512, 1]" = var_mean_18[0]
        getitem_37: "f32[8, 512, 1]" = var_mean_18[1];  var_mean_18 = None
        add_64: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-07);  getitem_36 = None
        rsqrt_18: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        sub_27: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_63, getitem_37);  add_63 = getitem_37 = None
        mul_130: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = None
        mul_131: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_130, primals_149)
        add_65: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_131, primals_150);  mul_131 = primals_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_198: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_65, [4096, 1536])
        permute_99: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_151, [1, 0])
        addmm_54: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_152, view_198, permute_99);  primals_152 = permute_99 = None
        view_199: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_54, [8, 512, 1536]);  addmm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_200: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_199, [8, 512, 24, -1]);  view_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_100: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None
        clone_36: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_100, memory_format = torch.contiguous_format);  permute_100 = None
        view_201: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_36, [-1, 512, 64]);  clone_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_101: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_153, [1, 0])
        addmm_55: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_154, view_198, permute_101);  primals_154 = permute_101 = None
        view_203: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_55, [8, 512, 1536]);  addmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_204: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_203, [8, 512, 24, -1]);  view_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_102: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None
        clone_37: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
        view_205: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_37, [-1, 512, 64]);  clone_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_103: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_155, [1, 0])
        addmm_56: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_156, view_198, permute_103);  primals_156 = permute_103 = None
        view_207: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_56, [8, 512, 1536]);  addmm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_208: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_207, [8, 512, 24, -1]);  view_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_104: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
        clone_38: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        view_209: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_38, [-1, 512, 64]);  clone_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_105: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_205, [0, 2, 1]);  view_205 = None
        div_18: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_105, full_default_1);  permute_105 = None
        bmm_18: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_201, div_18)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_210: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_18, [-1, 24, 512, 512]);  bmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_9: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_3, view_210);  view_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax_9: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(where_9, [-1], True)
        sub_28: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_9, amax_9);  where_9 = amax_9 = None
        exp_9: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_10: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_19: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        inductor_lookup_seed_default_28: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 28)
        inductor_random_default_44: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default([8, 24, 512, 512], inductor_lookup_seed_default_28, 'rand');  inductor_lookup_seed_default_28 = None
        gt_28: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_44, 0.1);  inductor_random_default_44 = None
        mul_133: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_28, div_19)
        mul_134: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_133, 1.1111111111111112);  mul_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_211: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(mul_134, [-1, 512, 512]);  mul_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_19: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_211, view_209)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_212: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_19, [-1, 24, 512, 64]);  bmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_106: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_39: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_213: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_39, [8, 512, -1]);  clone_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_214: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_213, [4096, 1536]);  view_213 = None
        permute_107: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_157, [1, 0])
        addmm_57: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_158, view_214, permute_107);  primals_158 = permute_107 = None
        view_215: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_57, [8, 512, 1536]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_29: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 29)
        inductor_random_default_43: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_29, 'rand');  inductor_lookup_seed_default_29 = None
        gt_29: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_43, 0.1);  inductor_random_default_43 = None
        mul_135: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_29, view_215);  view_215 = None
        mul_136: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_135, 1.1111111111111112);  mul_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_66: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_136, add_65);  mul_136 = add_65 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_66, [2], correction = 0, keepdim = True)
        getitem_38: "f32[8, 512, 1]" = var_mean_19[0]
        getitem_39: "f32[8, 512, 1]" = var_mean_19[1];  var_mean_19 = None
        add_67: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-07);  getitem_38 = None
        rsqrt_19: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
        sub_29: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_66, getitem_39);  add_66 = getitem_39 = None
        mul_137: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = None
        mul_138: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_137, primals_159)
        add_68: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_138, primals_160);  mul_138 = primals_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_216: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_68, [4096, 1536])
        permute_108: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_161, [1, 0])
        addmm_58: "f32[4096, 6144]" = torch.ops.aten.addmm.default(primals_162, view_216, permute_108);  primals_162 = permute_108 = None
        view_217: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_58, [8, 512, 6144])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_139: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_217, 0.5)
        mul_140: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476);  view_217 = None
        erf_9: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_140);  mul_140 = None
        add_69: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_141: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_139, add_69);  mul_139 = add_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_218: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_141, [4096, 6144]);  mul_141 = None
        permute_109: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_163, [1, 0])
        addmm_59: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_164, view_218, permute_109);  primals_164 = permute_109 = None
        view_219: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_59, [8, 512, 1536]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_30: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 30)
        inductor_random_default_42: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_30, 'rand');  inductor_lookup_seed_default_30 = None
        gt_30: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_42, 0.1);  inductor_random_default_42 = None
        mul_142: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_30, view_219);  view_219 = None
        mul_143: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_142, 1.1111111111111112);  mul_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_70: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_143, add_68);  mul_143 = add_68 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_70, [2], correction = 0, keepdim = True)
        getitem_40: "f32[8, 512, 1]" = var_mean_20[0]
        getitem_41: "f32[8, 512, 1]" = var_mean_20[1];  var_mean_20 = None
        add_71: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-07);  getitem_40 = None
        rsqrt_20: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        sub_30: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_70, getitem_41);  add_70 = getitem_41 = None
        mul_144: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = None
        mul_145: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_144, primals_165)
        add_72: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_145, primals_166);  mul_145 = primals_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_220: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_72, [4096, 1536])
        permute_110: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_167, [1, 0])
        addmm_60: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_168, view_220, permute_110);  primals_168 = permute_110 = None
        view_221: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_60, [8, 512, 1536]);  addmm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_222: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_221, [8, 512, 24, -1]);  view_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_111: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None
        clone_40: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_111, memory_format = torch.contiguous_format);  permute_111 = None
        view_223: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_40, [-1, 512, 64]);  clone_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_112: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_169, [1, 0])
        addmm_61: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_170, view_220, permute_112);  primals_170 = permute_112 = None
        view_225: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_61, [8, 512, 1536]);  addmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_226: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_225, [8, 512, 24, -1]);  view_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_113: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
        clone_41: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_113, memory_format = torch.contiguous_format);  permute_113 = None
        view_227: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_41, [-1, 512, 64]);  clone_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_114: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_171, [1, 0])
        addmm_62: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_172, view_220, permute_114);  primals_172 = permute_114 = None
        view_229: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_62, [8, 512, 1536]);  addmm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_230: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_229, [8, 512, 24, -1]);  view_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_115: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_230, [0, 2, 1, 3]);  view_230 = None
        clone_42: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
        view_231: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_42, [-1, 512, 64]);  clone_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_116: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_227, [0, 2, 1]);  view_227 = None
        div_20: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_116, full_default_1);  permute_116 = None
        bmm_20: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_223, div_20)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_232: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_20, [-1, 24, 512, 512]);  bmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_10: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_3, view_232);  view_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax_10: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(where_10, [-1], True)
        sub_31: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_10, amax_10);  where_10 = amax_10 = None
        exp_10: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_11: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_21: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        inductor_lookup_seed_default_31: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 31)
        inductor_random_default_41: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default([8, 24, 512, 512], inductor_lookup_seed_default_31, 'rand');  inductor_lookup_seed_default_31 = None
        gt_31: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_41, 0.1);  inductor_random_default_41 = None
        mul_147: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_31, div_21)
        mul_148: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_147, 1.1111111111111112);  mul_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_233: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(mul_148, [-1, 512, 512]);  mul_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_21: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_233, view_231)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_234: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_21, [-1, 24, 512, 64]);  bmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_117: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_43: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_235: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_43, [8, 512, -1]);  clone_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_236: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_235, [4096, 1536]);  view_235 = None
        permute_118: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_173, [1, 0])
        addmm_63: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_174, view_236, permute_118);  primals_174 = permute_118 = None
        view_237: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_63, [8, 512, 1536]);  addmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_32: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 32)
        inductor_random_default_40: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_32, 'rand');  inductor_lookup_seed_default_32 = None
        gt_32: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_40, 0.1);  inductor_random_default_40 = None
        mul_149: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_32, view_237);  view_237 = None
        mul_150: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_149, 1.1111111111111112);  mul_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_73: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_150, add_72);  mul_150 = add_72 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
        getitem_42: "f32[8, 512, 1]" = var_mean_21[0]
        getitem_43: "f32[8, 512, 1]" = var_mean_21[1];  var_mean_21 = None
        add_74: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-07);  getitem_42 = None
        rsqrt_21: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_32: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_73, getitem_43);  add_73 = getitem_43 = None
        mul_151: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = None
        mul_152: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_151, primals_175)
        add_75: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_152, primals_176);  mul_152 = primals_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_238: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_75, [4096, 1536])
        permute_119: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_177, [1, 0])
        addmm_64: "f32[4096, 6144]" = torch.ops.aten.addmm.default(primals_178, view_238, permute_119);  primals_178 = permute_119 = None
        view_239: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_64, [8, 512, 6144])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_153: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_239, 0.5)
        mul_154: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476);  view_239 = None
        erf_10: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_154);  mul_154 = None
        add_76: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_155: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_153, add_76);  mul_153 = add_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_240: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_155, [4096, 6144]);  mul_155 = None
        permute_120: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_179, [1, 0])
        addmm_65: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_180, view_240, permute_120);  primals_180 = permute_120 = None
        view_241: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_65, [8, 512, 1536]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_33: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 33)
        inductor_random_default_39: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_33, 'rand');  inductor_lookup_seed_default_33 = None
        gt_33: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_39, 0.1);  inductor_random_default_39 = None
        mul_156: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_33, view_241);  view_241 = None
        mul_157: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_156, 1.1111111111111112);  mul_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_77: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_157, add_75);  mul_157 = add_75 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
        getitem_44: "f32[8, 512, 1]" = var_mean_22[0]
        getitem_45: "f32[8, 512, 1]" = var_mean_22[1];  var_mean_22 = None
        add_78: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-07);  getitem_44 = None
        rsqrt_22: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        sub_33: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_77, getitem_45);  add_77 = getitem_45 = None
        mul_158: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = None
        mul_159: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_158, primals_181)
        add_79: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_159, primals_182);  mul_159 = primals_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_242: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_79, [4096, 1536])
        permute_121: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_183, [1, 0])
        addmm_66: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_184, view_242, permute_121);  primals_184 = permute_121 = None
        view_243: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_66, [8, 512, 1536]);  addmm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_244: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_243, [8, 512, 24, -1]);  view_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_122: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None
        clone_44: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
        view_245: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_44, [-1, 512, 64]);  clone_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_123: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_185, [1, 0])
        addmm_67: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_186, view_242, permute_123);  primals_186 = permute_123 = None
        view_247: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_67, [8, 512, 1536]);  addmm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_248: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_247, [8, 512, 24, -1]);  view_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_124: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_248, [0, 2, 1, 3]);  view_248 = None
        clone_45: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None
        view_249: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_45, [-1, 512, 64]);  clone_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_125: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_187, [1, 0])
        addmm_68: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_188, view_242, permute_125);  primals_188 = permute_125 = None
        view_251: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_68, [8, 512, 1536]);  addmm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_252: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_251, [8, 512, 24, -1]);  view_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_126: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None
        clone_46: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
        view_253: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_46, [-1, 512, 64]);  clone_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_127: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_249, [0, 2, 1]);  view_249 = None
        div_22: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_127, full_default_1);  permute_127 = None
        bmm_22: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_245, div_22)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_254: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_22, [-1, 24, 512, 512]);  bmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_11: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_3, view_254);  view_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax_11: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(where_11, [-1], True)
        sub_34: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_11, amax_11);  where_11 = amax_11 = None
        exp_11: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_12: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_23: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        inductor_lookup_seed_default_34: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 34)
        inductor_random_default_38: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default([8, 24, 512, 512], inductor_lookup_seed_default_34, 'rand');  inductor_lookup_seed_default_34 = None
        gt_34: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_38, 0.1);  inductor_random_default_38 = None
        mul_161: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_34, div_23)
        mul_162: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_161, 1.1111111111111112);  mul_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_255: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(mul_162, [-1, 512, 512]);  mul_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_23: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_255, view_253)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_256: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_23, [-1, 24, 512, 64]);  bmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_128: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_47: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_257: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_47, [8, 512, -1]);  clone_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_258: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_257, [4096, 1536]);  view_257 = None
        permute_129: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_189, [1, 0])
        addmm_69: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_190, view_258, permute_129);  primals_190 = permute_129 = None
        view_259: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_69, [8, 512, 1536]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_35: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 35)
        inductor_random_default_37: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_35, 'rand');  inductor_lookup_seed_default_35 = None
        gt_35: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_37, 0.1);  inductor_random_default_37 = None
        mul_163: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_35, view_259);  view_259 = None
        mul_164: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_163, 1.1111111111111112);  mul_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_80: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_164, add_79);  mul_164 = add_79 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_80, [2], correction = 0, keepdim = True)
        getitem_46: "f32[8, 512, 1]" = var_mean_23[0]
        getitem_47: "f32[8, 512, 1]" = var_mean_23[1];  var_mean_23 = None
        add_81: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-07);  getitem_46 = None
        rsqrt_23: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
        sub_35: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_80, getitem_47);  add_80 = getitem_47 = None
        mul_165: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = None
        mul_166: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_165, primals_191)
        add_82: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_166, primals_192);  mul_166 = primals_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_260: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_82, [4096, 1536])
        permute_130: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_193, [1, 0])
        addmm_70: "f32[4096, 6144]" = torch.ops.aten.addmm.default(primals_194, view_260, permute_130);  primals_194 = permute_130 = None
        view_261: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_70, [8, 512, 6144])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_167: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_261, 0.5)
        mul_168: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476);  view_261 = None
        erf_11: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_168);  mul_168 = None
        add_83: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_169: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_167, add_83);  mul_167 = add_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_262: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_169, [4096, 6144]);  mul_169 = None
        permute_131: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_195, [1, 0])
        addmm_71: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_196, view_262, permute_131);  primals_196 = permute_131 = None
        view_263: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_71, [8, 512, 1536]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_36: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 36)
        inductor_random_default_36: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_36, 'rand');  inductor_lookup_seed_default_36 = None
        gt_36: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_36, 0.1);  inductor_random_default_36 = None
        mul_170: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_36, view_263);  view_263 = None
        mul_171: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_170, 1.1111111111111112);  mul_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_84: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_171, add_82);  mul_171 = add_82 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_84, [2], correction = 0, keepdim = True)
        getitem_48: "f32[8, 512, 1]" = var_mean_24[0]
        getitem_49: "f32[8, 512, 1]" = var_mean_24[1];  var_mean_24 = None
        add_85: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-07);  getitem_48 = None
        rsqrt_24: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        sub_36: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_84, getitem_49);  add_84 = getitem_49 = None
        mul_172: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = None
        mul_173: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_172, primals_197)
        add_86: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_173, primals_198);  mul_173 = primals_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_264: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_86, [4096, 1536])
        permute_132: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_199, [1, 0])
        addmm_72: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_200, view_264, permute_132);  primals_200 = permute_132 = None
        view_265: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_72, [8, 512, 1536]);  addmm_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_266: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_265, [8, 512, 24, -1]);  view_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_133: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_266, [0, 2, 1, 3]);  view_266 = None
        clone_48: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_133, memory_format = torch.contiguous_format);  permute_133 = None
        view_267: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_48, [-1, 512, 64]);  clone_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_134: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_201, [1, 0])
        addmm_73: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_202, view_264, permute_134);  primals_202 = permute_134 = None
        view_269: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_73, [8, 512, 1536]);  addmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_270: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_269, [8, 512, 24, -1]);  view_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_135: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_270, [0, 2, 1, 3]);  view_270 = None
        clone_49: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None
        view_271: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_49, [-1, 512, 64]);  clone_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_136: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_203, [1, 0])
        addmm_74: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_204, view_264, permute_136);  primals_204 = permute_136 = None
        view_273: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_74, [8, 512, 1536]);  addmm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_274: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_273, [8, 512, 24, -1]);  view_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_137: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_274, [0, 2, 1, 3]);  view_274 = None
        clone_50: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
        view_275: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_50, [-1, 512, 64]);  clone_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_138: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_271, [0, 2, 1]);  view_271 = None
        div_24: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_138, full_default_1);  permute_138 = None
        bmm_24: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_267, div_24)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_276: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_24, [-1, 24, 512, 512]);  bmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_12: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_3, view_276);  view_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax_12: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(where_12, [-1], True)
        sub_37: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_12, amax_12);  where_12 = amax_12 = None
        exp_12: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
        sum_13: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_25: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        inductor_lookup_seed_default_37: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 37)
        inductor_random_default_35: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default([8, 24, 512, 512], inductor_lookup_seed_default_37, 'rand');  inductor_lookup_seed_default_37 = None
        gt_37: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_35, 0.1);  inductor_random_default_35 = None
        mul_175: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_37, div_25)
        mul_176: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_175, 1.1111111111111112);  mul_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_277: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(mul_176, [-1, 512, 512]);  mul_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_25: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_277, view_275)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_278: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_25, [-1, 24, 512, 64]);  bmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_139: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_278, [0, 2, 1, 3]);  view_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_51: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_279: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_51, [8, 512, -1]);  clone_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_280: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_279, [4096, 1536]);  view_279 = None
        permute_140: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_205, [1, 0])
        addmm_75: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_206, view_280, permute_140);  primals_206 = permute_140 = None
        view_281: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_75, [8, 512, 1536]);  addmm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_38: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 38)
        inductor_random_default_34: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_38, 'rand');  inductor_lookup_seed_default_38 = None
        gt_38: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_34, 0.1);  inductor_random_default_34 = None
        mul_177: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_38, view_281);  view_281 = None
        mul_178: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_177, 1.1111111111111112);  mul_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_87: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_178, add_86);  mul_178 = add_86 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(add_87, [2], correction = 0, keepdim = True)
        getitem_50: "f32[8, 512, 1]" = var_mean_25[0]
        getitem_51: "f32[8, 512, 1]" = var_mean_25[1];  var_mean_25 = None
        add_88: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-07);  getitem_50 = None
        rsqrt_25: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        sub_38: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_87, getitem_51);  add_87 = getitem_51 = None
        mul_179: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_25);  sub_38 = None
        mul_180: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_179, primals_207)
        add_89: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_180, primals_208);  mul_180 = primals_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_282: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_89, [4096, 1536])
        permute_141: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_209, [1, 0])
        addmm_76: "f32[4096, 6144]" = torch.ops.aten.addmm.default(primals_210, view_282, permute_141);  primals_210 = permute_141 = None
        view_283: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_76, [8, 512, 6144])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_181: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_283, 0.5)
        mul_182: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_283, 0.7071067811865476);  view_283 = None
        erf_12: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_182);  mul_182 = None
        add_90: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_183: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_181, add_90);  mul_181 = add_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_284: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_183, [4096, 6144]);  mul_183 = None
        permute_142: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_211, [1, 0])
        addmm_77: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_212, view_284, permute_142);  primals_212 = permute_142 = None
        view_285: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_77, [8, 512, 1536]);  addmm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_39: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 39)
        inductor_random_default_33: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_39, 'rand');  inductor_lookup_seed_default_39 = None
        gt_39: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_33, 0.1);  inductor_random_default_33 = None
        mul_184: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_39, view_285);  view_285 = None
        mul_185: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_184, 1.1111111111111112);  mul_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_91: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_185, add_89);  mul_185 = add_89 = None
        var_mean_26 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
        getitem_52: "f32[8, 512, 1]" = var_mean_26[0]
        getitem_53: "f32[8, 512, 1]" = var_mean_26[1];  var_mean_26 = None
        add_92: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-07);  getitem_52 = None
        rsqrt_26: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        sub_39: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_91, getitem_53);  add_91 = getitem_53 = None
        mul_186: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_26);  sub_39 = None
        mul_187: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_186, primals_213)
        add_93: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_187, primals_214);  mul_187 = primals_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_286: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_93, [4096, 1536])
        permute_143: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_215, [1, 0])
        addmm_78: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_216, view_286, permute_143);  primals_216 = permute_143 = None
        view_287: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_78, [8, 512, 1536]);  addmm_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_288: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_287, [8, 512, 24, -1]);  view_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_144: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_288, [0, 2, 1, 3]);  view_288 = None
        clone_52: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
        view_289: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_52, [-1, 512, 64]);  clone_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_145: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_217, [1, 0])
        addmm_79: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_218, view_286, permute_145);  primals_218 = permute_145 = None
        view_291: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_79, [8, 512, 1536]);  addmm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_292: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_291, [8, 512, 24, -1]);  view_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_146: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None
        clone_53: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_146, memory_format = torch.contiguous_format);  permute_146 = None
        view_293: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_53, [-1, 512, 64]);  clone_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_147: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_219, [1, 0])
        addmm_80: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_220, view_286, permute_147);  primals_220 = permute_147 = None
        view_295: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_80, [8, 512, 1536]);  addmm_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_296: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_295, [8, 512, 24, -1]);  view_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_148: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_296, [0, 2, 1, 3]);  view_296 = None
        clone_54: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
        view_297: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_54, [-1, 512, 64]);  clone_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_149: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_293, [0, 2, 1]);  view_293 = None
        div_26: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_149, full_default_1);  permute_149 = None
        bmm_26: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_289, div_26)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_298: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_26, [-1, 24, 512, 512]);  bmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_13: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_3, view_298);  view_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax_13: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(where_13, [-1], True)
        sub_40: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_13, amax_13);  where_13 = amax_13 = None
        exp_13: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_40);  sub_40 = None
        sum_14: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_27: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        inductor_lookup_seed_default_40: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 40)
        inductor_random_default_32: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default([8, 24, 512, 512], inductor_lookup_seed_default_40, 'rand');  inductor_lookup_seed_default_40 = None
        gt_40: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_32, 0.1);  inductor_random_default_32 = None
        mul_189: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_40, div_27)
        mul_190: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_189, 1.1111111111111112);  mul_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_299: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(mul_190, [-1, 512, 512]);  mul_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_27: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_299, view_297)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_300: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_27, [-1, 24, 512, 64]);  bmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_150: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_300, [0, 2, 1, 3]);  view_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_55: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_301: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_55, [8, 512, -1]);  clone_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_302: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_301, [4096, 1536]);  view_301 = None
        permute_151: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_221, [1, 0])
        addmm_81: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_222, view_302, permute_151);  primals_222 = permute_151 = None
        view_303: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_81, [8, 512, 1536]);  addmm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_41: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 41)
        inductor_random_default_31: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_41, 'rand');  inductor_lookup_seed_default_41 = None
        gt_41: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_31, 0.1);  inductor_random_default_31 = None
        mul_191: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_41, view_303);  view_303 = None
        mul_192: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_191, 1.1111111111111112);  mul_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_94: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_192, add_93);  mul_192 = add_93 = None
        var_mean_27 = torch.ops.aten.var_mean.correction(add_94, [2], correction = 0, keepdim = True)
        getitem_54: "f32[8, 512, 1]" = var_mean_27[0]
        getitem_55: "f32[8, 512, 1]" = var_mean_27[1];  var_mean_27 = None
        add_95: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-07);  getitem_54 = None
        rsqrt_27: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
        sub_41: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_94, getitem_55);  add_94 = getitem_55 = None
        mul_193: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_27);  sub_41 = None
        mul_194: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_193, primals_223)
        add_96: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_194, primals_224);  mul_194 = primals_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_304: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_96, [4096, 1536])
        permute_152: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_225, [1, 0])
        addmm_82: "f32[4096, 6144]" = torch.ops.aten.addmm.default(primals_226, view_304, permute_152);  primals_226 = permute_152 = None
        view_305: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_82, [8, 512, 6144])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_195: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_305, 0.5)
        mul_196: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_305, 0.7071067811865476);  view_305 = None
        erf_13: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_196);  mul_196 = None
        add_97: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_197: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_195, add_97);  mul_195 = add_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_306: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_197, [4096, 6144]);  mul_197 = None
        permute_153: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_227, [1, 0])
        addmm_83: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_228, view_306, permute_153);  primals_228 = permute_153 = None
        view_307: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_83, [8, 512, 1536]);  addmm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_42: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 42)
        inductor_random_default_30: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_42, 'rand');  inductor_lookup_seed_default_42 = None
        gt_42: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_30, 0.1);  inductor_random_default_30 = None
        mul_198: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_42, view_307);  view_307 = None
        mul_199: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_198, 1.1111111111111112);  mul_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_98: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_199, add_96);  mul_199 = add_96 = None
        var_mean_28 = torch.ops.aten.var_mean.correction(add_98, [2], correction = 0, keepdim = True)
        getitem_56: "f32[8, 512, 1]" = var_mean_28[0]
        getitem_57: "f32[8, 512, 1]" = var_mean_28[1];  var_mean_28 = None
        add_99: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-07);  getitem_56 = None
        rsqrt_28: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
        sub_42: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_98, getitem_57);  add_98 = getitem_57 = None
        mul_200: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_28);  sub_42 = None
        mul_201: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_200, primals_229)
        add_100: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_201, primals_230);  mul_201 = primals_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_308: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_100, [4096, 1536])
        permute_154: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_231, [1, 0])
        addmm_84: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_232, view_308, permute_154);  primals_232 = permute_154 = None
        view_309: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_84, [8, 512, 1536]);  addmm_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_310: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_309, [8, 512, 24, -1]);  view_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_155: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_310, [0, 2, 1, 3]);  view_310 = None
        clone_56: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
        view_311: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_56, [-1, 512, 64]);  clone_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_156: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_233, [1, 0])
        addmm_85: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_234, view_308, permute_156);  primals_234 = permute_156 = None
        view_313: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_85, [8, 512, 1536]);  addmm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_314: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_313, [8, 512, 24, -1]);  view_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_157: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_314, [0, 2, 1, 3]);  view_314 = None
        clone_57: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_157, memory_format = torch.contiguous_format);  permute_157 = None
        view_315: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_57, [-1, 512, 64]);  clone_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_158: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_235, [1, 0])
        addmm_86: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_236, view_308, permute_158);  primals_236 = permute_158 = None
        view_317: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_86, [8, 512, 1536]);  addmm_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_318: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_317, [8, 512, 24, -1]);  view_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_159: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_318, [0, 2, 1, 3]);  view_318 = None
        clone_58: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None
        view_319: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_58, [-1, 512, 64]);  clone_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_160: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_315, [0, 2, 1]);  view_315 = None
        div_28: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_160, full_default_1);  permute_160 = None
        bmm_28: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_311, div_28)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_320: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_28, [-1, 24, 512, 512]);  bmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_14: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_3, view_320);  view_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax_14: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(where_14, [-1], True)
        sub_43: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_14, amax_14);  where_14 = amax_14 = None
        exp_14: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
        sum_15: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_29: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        inductor_lookup_seed_default_43: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 43)
        inductor_random_default_29: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default([8, 24, 512, 512], inductor_lookup_seed_default_43, 'rand');  inductor_lookup_seed_default_43 = None
        gt_43: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_29, 0.1);  inductor_random_default_29 = None
        mul_203: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_43, div_29)
        mul_204: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_203, 1.1111111111111112);  mul_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_321: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(mul_204, [-1, 512, 512]);  mul_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_29: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_321, view_319)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_322: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_29, [-1, 24, 512, 64]);  bmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_161: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_322, [0, 2, 1, 3]);  view_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_59: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_323: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_59, [8, 512, -1]);  clone_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_324: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_323, [4096, 1536]);  view_323 = None
        permute_162: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_237, [1, 0])
        addmm_87: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_238, view_324, permute_162);  primals_238 = permute_162 = None
        view_325: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_87, [8, 512, 1536]);  addmm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_44: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 44)
        inductor_random_default_28: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_44, 'rand');  inductor_lookup_seed_default_44 = None
        gt_44: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_28, 0.1);  inductor_random_default_28 = None
        mul_205: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_44, view_325);  view_325 = None
        mul_206: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_205, 1.1111111111111112);  mul_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_101: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_206, add_100);  mul_206 = add_100 = None
        var_mean_29 = torch.ops.aten.var_mean.correction(add_101, [2], correction = 0, keepdim = True)
        getitem_58: "f32[8, 512, 1]" = var_mean_29[0]
        getitem_59: "f32[8, 512, 1]" = var_mean_29[1];  var_mean_29 = None
        add_102: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-07);  getitem_58 = None
        rsqrt_29: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        sub_44: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_101, getitem_59);  add_101 = getitem_59 = None
        mul_207: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_29);  sub_44 = None
        mul_208: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_207, primals_239)
        add_103: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_208, primals_240);  mul_208 = primals_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_326: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_103, [4096, 1536])
        permute_163: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_241, [1, 0])
        addmm_88: "f32[4096, 6144]" = torch.ops.aten.addmm.default(primals_242, view_326, permute_163);  primals_242 = permute_163 = None
        view_327: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_88, [8, 512, 6144])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_209: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_327, 0.5)
        mul_210: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_327, 0.7071067811865476);  view_327 = None
        erf_14: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_210);  mul_210 = None
        add_104: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_211: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_209, add_104);  mul_209 = add_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_328: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_211, [4096, 6144]);  mul_211 = None
        permute_164: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_243, [1, 0])
        addmm_89: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_244, view_328, permute_164);  primals_244 = permute_164 = None
        view_329: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_89, [8, 512, 1536]);  addmm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_45: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 45)
        inductor_random_default_27: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_45, 'rand');  inductor_lookup_seed_default_45 = None
        gt_45: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_27, 0.1);  inductor_random_default_27 = None
        mul_212: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_45, view_329);  view_329 = None
        mul_213: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_212, 1.1111111111111112);  mul_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_105: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_213, add_103);  mul_213 = add_103 = None
        var_mean_30 = torch.ops.aten.var_mean.correction(add_105, [2], correction = 0, keepdim = True)
        getitem_60: "f32[8, 512, 1]" = var_mean_30[0]
        getitem_61: "f32[8, 512, 1]" = var_mean_30[1];  var_mean_30 = None
        add_106: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-07);  getitem_60 = None
        rsqrt_30: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        sub_45: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_105, getitem_61);  add_105 = getitem_61 = None
        mul_214: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_30);  sub_45 = None
        mul_215: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_214, primals_245)
        add_107: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_215, primals_246);  mul_215 = primals_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_330: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_107, [4096, 1536])
        permute_165: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_247, [1, 0])
        addmm_90: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_248, view_330, permute_165);  primals_248 = permute_165 = None
        view_331: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_90, [8, 512, 1536]);  addmm_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_332: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_331, [8, 512, 24, -1]);  view_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_166: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_332, [0, 2, 1, 3]);  view_332 = None
        clone_60: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_166, memory_format = torch.contiguous_format);  permute_166 = None
        view_333: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_60, [-1, 512, 64]);  clone_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_167: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_249, [1, 0])
        addmm_91: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_250, view_330, permute_167);  primals_250 = permute_167 = None
        view_335: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_91, [8, 512, 1536]);  addmm_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_336: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_335, [8, 512, 24, -1]);  view_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_168: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_336, [0, 2, 1, 3]);  view_336 = None
        clone_61: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None
        view_337: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_61, [-1, 512, 64]);  clone_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_169: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_251, [1, 0])
        addmm_92: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_252, view_330, permute_169);  primals_252 = permute_169 = None
        view_339: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_92, [8, 512, 1536]);  addmm_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_340: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_339, [8, 512, 24, -1]);  view_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_170: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_340, [0, 2, 1, 3]);  view_340 = None
        clone_62: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
        view_341: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_62, [-1, 512, 64]);  clone_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_171: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_337, [0, 2, 1]);  view_337 = None
        div_30: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_171, full_default_1);  permute_171 = None
        bmm_30: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_333, div_30)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_342: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_30, [-1, 24, 512, 512]);  bmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_15: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_3, view_342);  view_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax_15: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(where_15, [-1], True)
        sub_46: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_15, amax_15);  where_15 = amax_15 = None
        exp_15: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
        sum_16: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_31: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        inductor_lookup_seed_default_46: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 46)
        inductor_random_default_26: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default([8, 24, 512, 512], inductor_lookup_seed_default_46, 'rand');  inductor_lookup_seed_default_46 = None
        gt_46: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_26, 0.1);  inductor_random_default_26 = None
        mul_217: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_46, div_31)
        mul_218: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_217, 1.1111111111111112);  mul_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_343: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(mul_218, [-1, 512, 512]);  mul_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_31: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_343, view_341)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_344: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_31, [-1, 24, 512, 64]);  bmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_172: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_63: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_345: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_63, [8, 512, -1]);  clone_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_346: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_345, [4096, 1536]);  view_345 = None
        permute_173: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_253, [1, 0])
        addmm_93: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_254, view_346, permute_173);  primals_254 = permute_173 = None
        view_347: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_93, [8, 512, 1536]);  addmm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_47: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 47)
        inductor_random_default_25: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_47, 'rand');  inductor_lookup_seed_default_47 = None
        gt_47: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_25, 0.1);  inductor_random_default_25 = None
        mul_219: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_47, view_347);  view_347 = None
        mul_220: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_219, 1.1111111111111112);  mul_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_108: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_220, add_107);  mul_220 = add_107 = None
        var_mean_31 = torch.ops.aten.var_mean.correction(add_108, [2], correction = 0, keepdim = True)
        getitem_62: "f32[8, 512, 1]" = var_mean_31[0]
        getitem_63: "f32[8, 512, 1]" = var_mean_31[1];  var_mean_31 = None
        add_109: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-07);  getitem_62 = None
        rsqrt_31: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
        sub_47: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_108, getitem_63);  add_108 = getitem_63 = None
        mul_221: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_31);  sub_47 = None
        mul_222: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_221, primals_255)
        add_110: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_222, primals_256);  mul_222 = primals_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_348: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_110, [4096, 1536])
        permute_174: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_257, [1, 0])
        addmm_94: "f32[4096, 6144]" = torch.ops.aten.addmm.default(primals_258, view_348, permute_174);  primals_258 = permute_174 = None
        view_349: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_94, [8, 512, 6144])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_223: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_349, 0.5)
        mul_224: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_349, 0.7071067811865476);  view_349 = None
        erf_15: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_224);  mul_224 = None
        add_111: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_225: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_223, add_111);  mul_223 = add_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_350: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_225, [4096, 6144]);  mul_225 = None
        permute_175: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_259, [1, 0])
        addmm_95: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_260, view_350, permute_175);  primals_260 = permute_175 = None
        view_351: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_95, [8, 512, 1536]);  addmm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_48: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 48)
        inductor_random_default_24: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_48, 'rand');  inductor_lookup_seed_default_48 = None
        gt_48: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_24, 0.1);  inductor_random_default_24 = None
        mul_226: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_48, view_351);  view_351 = None
        mul_227: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_226, 1.1111111111111112);  mul_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_112: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_227, add_110);  mul_227 = add_110 = None
        var_mean_32 = torch.ops.aten.var_mean.correction(add_112, [2], correction = 0, keepdim = True)
        getitem_64: "f32[8, 512, 1]" = var_mean_32[0]
        getitem_65: "f32[8, 512, 1]" = var_mean_32[1];  var_mean_32 = None
        add_113: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-07);  getitem_64 = None
        rsqrt_32: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
        sub_48: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_112, getitem_65);  add_112 = getitem_65 = None
        mul_228: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_32);  sub_48 = None
        mul_229: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_228, primals_261)
        add_114: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_229, primals_262);  mul_229 = primals_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_352: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_114, [4096, 1536])
        permute_176: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_263, [1, 0])
        addmm_96: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_264, view_352, permute_176);  primals_264 = permute_176 = None
        view_353: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_96, [8, 512, 1536]);  addmm_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_354: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_353, [8, 512, 24, -1]);  view_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_177: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_354, [0, 2, 1, 3]);  view_354 = None
        clone_64: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
        view_355: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_64, [-1, 512, 64]);  clone_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_178: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_265, [1, 0])
        addmm_97: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_266, view_352, permute_178);  primals_266 = permute_178 = None
        view_357: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_97, [8, 512, 1536]);  addmm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_358: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_357, [8, 512, 24, -1]);  view_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_179: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_358, [0, 2, 1, 3]);  view_358 = None
        clone_65: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None
        view_359: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_65, [-1, 512, 64]);  clone_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_180: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_267, [1, 0])
        addmm_98: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_268, view_352, permute_180);  primals_268 = permute_180 = None
        view_361: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_98, [8, 512, 1536]);  addmm_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_362: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_361, [8, 512, 24, -1]);  view_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_181: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_362, [0, 2, 1, 3]);  view_362 = None
        clone_66: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_181, memory_format = torch.contiguous_format);  permute_181 = None
        view_363: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_66, [-1, 512, 64]);  clone_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_182: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_359, [0, 2, 1]);  view_359 = None
        div_32: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_182, full_default_1);  permute_182 = None
        bmm_32: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_355, div_32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_364: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_32, [-1, 24, 512, 512]);  bmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_16: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_3, view_364);  view_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax_16: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(where_16, [-1], True)
        sub_49: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_16, amax_16);  where_16 = amax_16 = None
        exp_16: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
        sum_17: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_33: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        inductor_lookup_seed_default_49: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 49)
        inductor_random_default_23: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default([8, 24, 512, 512], inductor_lookup_seed_default_49, 'rand');  inductor_lookup_seed_default_49 = None
        gt_49: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_23, 0.1);  inductor_random_default_23 = None
        mul_231: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_49, div_33)
        mul_232: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_231, 1.1111111111111112);  mul_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_365: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(mul_232, [-1, 512, 512]);  mul_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_33: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_365, view_363)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_366: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_33, [-1, 24, 512, 64]);  bmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_183: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_366, [0, 2, 1, 3]);  view_366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_67: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_367: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_67, [8, 512, -1]);  clone_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_368: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_367, [4096, 1536]);  view_367 = None
        permute_184: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_269, [1, 0])
        addmm_99: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_270, view_368, permute_184);  primals_270 = permute_184 = None
        view_369: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_99, [8, 512, 1536]);  addmm_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_50: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 50)
        inductor_random_default_22: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_50, 'rand');  inductor_lookup_seed_default_50 = None
        gt_50: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_22, 0.1);  inductor_random_default_22 = None
        mul_233: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_50, view_369);  view_369 = None
        mul_234: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_233, 1.1111111111111112);  mul_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_115: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_234, add_114);  mul_234 = add_114 = None
        var_mean_33 = torch.ops.aten.var_mean.correction(add_115, [2], correction = 0, keepdim = True)
        getitem_66: "f32[8, 512, 1]" = var_mean_33[0]
        getitem_67: "f32[8, 512, 1]" = var_mean_33[1];  var_mean_33 = None
        add_116: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-07);  getitem_66 = None
        rsqrt_33: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        sub_50: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_115, getitem_67);  add_115 = getitem_67 = None
        mul_235: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_33);  sub_50 = None
        mul_236: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_235, primals_271)
        add_117: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_236, primals_272);  mul_236 = primals_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_370: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_117, [4096, 1536])
        permute_185: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_273, [1, 0])
        addmm_100: "f32[4096, 6144]" = torch.ops.aten.addmm.default(primals_274, view_370, permute_185);  primals_274 = permute_185 = None
        view_371: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_100, [8, 512, 6144])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_237: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_371, 0.5)
        mul_238: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_371, 0.7071067811865476);  view_371 = None
        erf_16: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_238);  mul_238 = None
        add_118: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_239: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_237, add_118);  mul_237 = add_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_372: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_239, [4096, 6144]);  mul_239 = None
        permute_186: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_275, [1, 0])
        addmm_101: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_276, view_372, permute_186);  primals_276 = permute_186 = None
        view_373: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_101, [8, 512, 1536]);  addmm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_51: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 51)
        inductor_random_default_21: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_51, 'rand');  inductor_lookup_seed_default_51 = None
        gt_51: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_21, 0.1);  inductor_random_default_21 = None
        mul_240: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_51, view_373);  view_373 = None
        mul_241: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_240, 1.1111111111111112);  mul_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_119: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_241, add_117);  mul_241 = add_117 = None
        var_mean_34 = torch.ops.aten.var_mean.correction(add_119, [2], correction = 0, keepdim = True)
        getitem_68: "f32[8, 512, 1]" = var_mean_34[0]
        getitem_69: "f32[8, 512, 1]" = var_mean_34[1];  var_mean_34 = None
        add_120: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-07);  getitem_68 = None
        rsqrt_34: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
        sub_51: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_119, getitem_69);  add_119 = getitem_69 = None
        mul_242: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_34);  sub_51 = None
        mul_243: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_242, primals_277)
        add_121: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_243, primals_278);  mul_243 = primals_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_374: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_121, [4096, 1536])
        permute_187: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_279, [1, 0])
        addmm_102: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_280, view_374, permute_187);  primals_280 = permute_187 = None
        view_375: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_102, [8, 512, 1536]);  addmm_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_376: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_375, [8, 512, 24, -1]);  view_375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_188: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_376, [0, 2, 1, 3]);  view_376 = None
        clone_68: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
        view_377: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_68, [-1, 512, 64]);  clone_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_189: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_281, [1, 0])
        addmm_103: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_282, view_374, permute_189);  primals_282 = permute_189 = None
        view_379: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_103, [8, 512, 1536]);  addmm_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_380: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_379, [8, 512, 24, -1]);  view_379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_190: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_380, [0, 2, 1, 3]);  view_380 = None
        clone_69: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_190, memory_format = torch.contiguous_format);  permute_190 = None
        view_381: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_69, [-1, 512, 64]);  clone_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_191: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_283, [1, 0])
        addmm_104: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_284, view_374, permute_191);  primals_284 = permute_191 = None
        view_383: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_104, [8, 512, 1536]);  addmm_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_384: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_383, [8, 512, 24, -1]);  view_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_192: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_384, [0, 2, 1, 3]);  view_384 = None
        clone_70: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_192, memory_format = torch.contiguous_format);  permute_192 = None
        view_385: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_70, [-1, 512, 64]);  clone_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_193: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_381, [0, 2, 1]);  view_381 = None
        div_34: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_193, full_default_1);  permute_193 = None
        bmm_34: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_377, div_34)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_386: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_34, [-1, 24, 512, 512]);  bmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_17: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_3, view_386);  view_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax_17: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(where_17, [-1], True)
        sub_52: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_17, amax_17);  where_17 = amax_17 = None
        exp_17: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
        sum_18: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_35: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        inductor_lookup_seed_default_52: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 52)
        inductor_random_default_20: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default([8, 24, 512, 512], inductor_lookup_seed_default_52, 'rand');  inductor_lookup_seed_default_52 = None
        gt_52: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_20, 0.1);  inductor_random_default_20 = None
        mul_245: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_52, div_35)
        mul_246: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_245, 1.1111111111111112);  mul_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_387: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(mul_246, [-1, 512, 512]);  mul_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_35: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_387, view_385)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_388: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_35, [-1, 24, 512, 64]);  bmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_194: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_388, [0, 2, 1, 3]);  view_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_71: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_389: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_71, [8, 512, -1]);  clone_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_390: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_389, [4096, 1536]);  view_389 = None
        permute_195: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_285, [1, 0])
        addmm_105: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_286, view_390, permute_195);  primals_286 = permute_195 = None
        view_391: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_105, [8, 512, 1536]);  addmm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_53: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 53)
        inductor_random_default_19: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_53, 'rand');  inductor_lookup_seed_default_53 = None
        gt_53: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_19, 0.1);  inductor_random_default_19 = None
        mul_247: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_53, view_391);  view_391 = None
        mul_248: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_247, 1.1111111111111112);  mul_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_122: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_248, add_121);  mul_248 = add_121 = None
        var_mean_35 = torch.ops.aten.var_mean.correction(add_122, [2], correction = 0, keepdim = True)
        getitem_70: "f32[8, 512, 1]" = var_mean_35[0]
        getitem_71: "f32[8, 512, 1]" = var_mean_35[1];  var_mean_35 = None
        add_123: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-07);  getitem_70 = None
        rsqrt_35: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
        sub_53: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_122, getitem_71);  add_122 = getitem_71 = None
        mul_249: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_35);  sub_53 = None
        mul_250: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_249, primals_287)
        add_124: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_250, primals_288);  mul_250 = primals_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_392: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_124, [4096, 1536])
        permute_196: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_289, [1, 0])
        addmm_106: "f32[4096, 6144]" = torch.ops.aten.addmm.default(primals_290, view_392, permute_196);  primals_290 = permute_196 = None
        view_393: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_106, [8, 512, 6144])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_251: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_393, 0.5)
        mul_252: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_393, 0.7071067811865476);  view_393 = None
        erf_17: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_252);  mul_252 = None
        add_125: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_253: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_251, add_125);  mul_251 = add_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_394: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_253, [4096, 6144]);  mul_253 = None
        permute_197: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_291, [1, 0])
        addmm_107: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_292, view_394, permute_197);  primals_292 = permute_197 = None
        view_395: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_107, [8, 512, 1536]);  addmm_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_54: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 54)
        inductor_random_default_18: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_54, 'rand');  inductor_lookup_seed_default_54 = None
        gt_54: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_18, 0.1);  inductor_random_default_18 = None
        mul_254: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_54, view_395);  view_395 = None
        mul_255: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_254, 1.1111111111111112);  mul_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_126: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_255, add_124);  mul_255 = add_124 = None
        var_mean_36 = torch.ops.aten.var_mean.correction(add_126, [2], correction = 0, keepdim = True)
        getitem_72: "f32[8, 512, 1]" = var_mean_36[0]
        getitem_73: "f32[8, 512, 1]" = var_mean_36[1];  var_mean_36 = None
        add_127: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-07);  getitem_72 = None
        rsqrt_36: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
        sub_54: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_126, getitem_73);  add_126 = getitem_73 = None
        mul_256: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_36);  sub_54 = None
        mul_257: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_256, primals_293)
        add_128: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_257, primals_294);  mul_257 = primals_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_396: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_128, [4096, 1536])
        permute_198: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_295, [1, 0])
        addmm_108: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_296, view_396, permute_198);  primals_296 = permute_198 = None
        view_397: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_108, [8, 512, 1536]);  addmm_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_398: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_397, [8, 512, 24, -1]);  view_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_199: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_398, [0, 2, 1, 3]);  view_398 = None
        clone_72: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None
        view_399: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_72, [-1, 512, 64]);  clone_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_200: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_297, [1, 0])
        addmm_109: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_298, view_396, permute_200);  primals_298 = permute_200 = None
        view_401: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_109, [8, 512, 1536]);  addmm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_402: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_401, [8, 512, 24, -1]);  view_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_201: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
        clone_73: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_201, memory_format = torch.contiguous_format);  permute_201 = None
        view_403: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_73, [-1, 512, 64]);  clone_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_202: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_299, [1, 0])
        addmm_110: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_300, view_396, permute_202);  primals_300 = permute_202 = None
        view_405: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_110, [8, 512, 1536]);  addmm_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_406: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_405, [8, 512, 24, -1]);  view_405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_203: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_406, [0, 2, 1, 3]);  view_406 = None
        clone_74: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_203, memory_format = torch.contiguous_format);  permute_203 = None
        view_407: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_74, [-1, 512, 64]);  clone_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_204: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_403, [0, 2, 1]);  view_403 = None
        div_36: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_204, full_default_1);  permute_204 = None
        bmm_36: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_399, div_36)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_408: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_36, [-1, 24, 512, 512]);  bmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_18: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_3, view_408);  view_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax_18: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(where_18, [-1], True)
        sub_55: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_18, amax_18);  where_18 = amax_18 = None
        exp_18: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
        sum_19: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_37: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        inductor_lookup_seed_default_55: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 55)
        inductor_random_default_17: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default([8, 24, 512, 512], inductor_lookup_seed_default_55, 'rand');  inductor_lookup_seed_default_55 = None
        gt_55: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_17, 0.1);  inductor_random_default_17 = None
        mul_259: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_55, div_37)
        mul_260: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_259, 1.1111111111111112);  mul_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_409: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(mul_260, [-1, 512, 512]);  mul_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_37: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_409, view_407)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_410: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_37, [-1, 24, 512, 64]);  bmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_205: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_410, [0, 2, 1, 3]);  view_410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_75: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_411: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_75, [8, 512, -1]);  clone_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_412: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_411, [4096, 1536]);  view_411 = None
        permute_206: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_301, [1, 0])
        addmm_111: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_302, view_412, permute_206);  primals_302 = permute_206 = None
        view_413: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_111, [8, 512, 1536]);  addmm_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_56: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 56)
        inductor_random_default_16: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_56, 'rand');  inductor_lookup_seed_default_56 = None
        gt_56: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_16, 0.1);  inductor_random_default_16 = None
        mul_261: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_56, view_413);  view_413 = None
        mul_262: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_261, 1.1111111111111112);  mul_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_129: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_262, add_128);  mul_262 = add_128 = None
        var_mean_37 = torch.ops.aten.var_mean.correction(add_129, [2], correction = 0, keepdim = True)
        getitem_74: "f32[8, 512, 1]" = var_mean_37[0]
        getitem_75: "f32[8, 512, 1]" = var_mean_37[1];  var_mean_37 = None
        add_130: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-07);  getitem_74 = None
        rsqrt_37: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        sub_56: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_129, getitem_75);  add_129 = getitem_75 = None
        mul_263: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_37);  sub_56 = None
        mul_264: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_263, primals_303)
        add_131: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_264, primals_304);  mul_264 = primals_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_414: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_131, [4096, 1536])
        permute_207: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_305, [1, 0])
        addmm_112: "f32[4096, 6144]" = torch.ops.aten.addmm.default(primals_306, view_414, permute_207);  primals_306 = permute_207 = None
        view_415: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_112, [8, 512, 6144])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_265: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_415, 0.5)
        mul_266: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_415, 0.7071067811865476);  view_415 = None
        erf_18: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_266);  mul_266 = None
        add_132: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_267: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_265, add_132);  mul_265 = add_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_416: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_267, [4096, 6144]);  mul_267 = None
        permute_208: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_307, [1, 0])
        addmm_113: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_308, view_416, permute_208);  primals_308 = permute_208 = None
        view_417: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_113, [8, 512, 1536]);  addmm_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_57: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 57)
        inductor_random_default_15: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_57, 'rand');  inductor_lookup_seed_default_57 = None
        gt_57: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_15, 0.1);  inductor_random_default_15 = None
        mul_268: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_57, view_417);  view_417 = None
        mul_269: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_268, 1.1111111111111112);  mul_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_133: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_269, add_131);  mul_269 = add_131 = None
        var_mean_38 = torch.ops.aten.var_mean.correction(add_133, [2], correction = 0, keepdim = True)
        getitem_76: "f32[8, 512, 1]" = var_mean_38[0]
        getitem_77: "f32[8, 512, 1]" = var_mean_38[1];  var_mean_38 = None
        add_134: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-07);  getitem_76 = None
        rsqrt_38: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        sub_57: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_133, getitem_77);  add_133 = getitem_77 = None
        mul_270: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_38);  sub_57 = None
        mul_271: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_270, primals_309)
        add_135: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_271, primals_310);  mul_271 = primals_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_418: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_135, [4096, 1536])
        permute_209: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_311, [1, 0])
        addmm_114: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_312, view_418, permute_209);  primals_312 = permute_209 = None
        view_419: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_114, [8, 512, 1536]);  addmm_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_420: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_419, [8, 512, 24, -1]);  view_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_210: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_420, [0, 2, 1, 3]);  view_420 = None
        clone_76: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None
        view_421: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_76, [-1, 512, 64]);  clone_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_211: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_313, [1, 0])
        addmm_115: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_314, view_418, permute_211);  primals_314 = permute_211 = None
        view_423: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_115, [8, 512, 1536]);  addmm_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_424: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_423, [8, 512, 24, -1]);  view_423 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_212: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_424, [0, 2, 1, 3]);  view_424 = None
        clone_77: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_212, memory_format = torch.contiguous_format);  permute_212 = None
        view_425: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_77, [-1, 512, 64]);  clone_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_213: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_315, [1, 0])
        addmm_116: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_316, view_418, permute_213);  primals_316 = permute_213 = None
        view_427: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_116, [8, 512, 1536]);  addmm_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_428: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_427, [8, 512, 24, -1]);  view_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_214: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_428, [0, 2, 1, 3]);  view_428 = None
        clone_78: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_214, memory_format = torch.contiguous_format);  permute_214 = None
        view_429: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_78, [-1, 512, 64]);  clone_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_215: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_425, [0, 2, 1]);  view_425 = None
        div_38: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_215, full_default_1);  permute_215 = None
        bmm_38: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_421, div_38)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_430: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_38, [-1, 24, 512, 512]);  bmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_19: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_3, view_430);  view_430 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax_19: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(where_19, [-1], True)
        sub_58: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_19, amax_19);  where_19 = amax_19 = None
        exp_19: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_58);  sub_58 = None
        sum_20: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_39: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        inductor_lookup_seed_default_58: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 58)
        inductor_random_default_14: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default([8, 24, 512, 512], inductor_lookup_seed_default_58, 'rand');  inductor_lookup_seed_default_58 = None
        gt_58: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_14, 0.1);  inductor_random_default_14 = None
        mul_273: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_58, div_39)
        mul_274: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_273, 1.1111111111111112);  mul_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_431: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(mul_274, [-1, 512, 512]);  mul_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_39: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_431, view_429)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_432: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_39, [-1, 24, 512, 64]);  bmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_216: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_432, [0, 2, 1, 3]);  view_432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_79: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_433: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_79, [8, 512, -1]);  clone_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_434: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_433, [4096, 1536]);  view_433 = None
        permute_217: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_317, [1, 0])
        addmm_117: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_318, view_434, permute_217);  primals_318 = permute_217 = None
        view_435: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_117, [8, 512, 1536]);  addmm_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_59: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 59)
        inductor_random_default_13: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_59, 'rand');  inductor_lookup_seed_default_59 = None
        gt_59: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_13, 0.1);  inductor_random_default_13 = None
        mul_275: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_59, view_435);  view_435 = None
        mul_276: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_275, 1.1111111111111112);  mul_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_136: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_276, add_135);  mul_276 = add_135 = None
        var_mean_39 = torch.ops.aten.var_mean.correction(add_136, [2], correction = 0, keepdim = True)
        getitem_78: "f32[8, 512, 1]" = var_mean_39[0]
        getitem_79: "f32[8, 512, 1]" = var_mean_39[1];  var_mean_39 = None
        add_137: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-07);  getitem_78 = None
        rsqrt_39: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
        sub_59: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_136, getitem_79);  add_136 = getitem_79 = None
        mul_277: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_39);  sub_59 = None
        mul_278: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_277, primals_319)
        add_138: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_278, primals_320);  mul_278 = primals_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_436: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_138, [4096, 1536])
        permute_218: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_321, [1, 0])
        addmm_118: "f32[4096, 6144]" = torch.ops.aten.addmm.default(primals_322, view_436, permute_218);  primals_322 = permute_218 = None
        view_437: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_118, [8, 512, 6144])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_279: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_437, 0.5)
        mul_280: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_437, 0.7071067811865476);  view_437 = None
        erf_19: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_280);  mul_280 = None
        add_139: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_281: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_279, add_139);  mul_279 = add_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_438: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_281, [4096, 6144]);  mul_281 = None
        permute_219: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_323, [1, 0])
        addmm_119: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_324, view_438, permute_219);  primals_324 = permute_219 = None
        view_439: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_119, [8, 512, 1536]);  addmm_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_60: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 60)
        inductor_random_default_12: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_60, 'rand');  inductor_lookup_seed_default_60 = None
        gt_60: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_12, 0.1);  inductor_random_default_12 = None
        mul_282: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_60, view_439);  view_439 = None
        mul_283: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_282, 1.1111111111111112);  mul_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_140: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_283, add_138);  mul_283 = add_138 = None
        var_mean_40 = torch.ops.aten.var_mean.correction(add_140, [2], correction = 0, keepdim = True)
        getitem_80: "f32[8, 512, 1]" = var_mean_40[0]
        getitem_81: "f32[8, 512, 1]" = var_mean_40[1];  var_mean_40 = None
        add_141: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-07);  getitem_80 = None
        rsqrt_40: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
        sub_60: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_140, getitem_81);  add_140 = getitem_81 = None
        mul_284: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_40);  sub_60 = None
        mul_285: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_284, primals_325)
        add_142: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_285, primals_326);  mul_285 = primals_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_440: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_142, [4096, 1536])
        permute_220: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_327, [1, 0])
        addmm_120: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_328, view_440, permute_220);  primals_328 = permute_220 = None
        view_441: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_120, [8, 512, 1536]);  addmm_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_442: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_441, [8, 512, 24, -1]);  view_441 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_221: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_442, [0, 2, 1, 3]);  view_442 = None
        clone_80: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_221, memory_format = torch.contiguous_format);  permute_221 = None
        view_443: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_80, [-1, 512, 64]);  clone_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_222: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_329, [1, 0])
        addmm_121: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_330, view_440, permute_222);  primals_330 = permute_222 = None
        view_445: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_121, [8, 512, 1536]);  addmm_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_446: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_445, [8, 512, 24, -1]);  view_445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_223: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_446, [0, 2, 1, 3]);  view_446 = None
        clone_81: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_223, memory_format = torch.contiguous_format);  permute_223 = None
        view_447: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_81, [-1, 512, 64]);  clone_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_224: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_331, [1, 0])
        addmm_122: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_332, view_440, permute_224);  primals_332 = permute_224 = None
        view_449: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_122, [8, 512, 1536]);  addmm_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_450: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_449, [8, 512, 24, -1]);  view_449 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_225: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_450, [0, 2, 1, 3]);  view_450 = None
        clone_82: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_225, memory_format = torch.contiguous_format);  permute_225 = None
        view_451: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_82, [-1, 512, 64]);  clone_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_226: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_447, [0, 2, 1]);  view_447 = None
        div_40: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_226, full_default_1);  permute_226 = None
        bmm_40: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_443, div_40)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_452: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_40, [-1, 24, 512, 512]);  bmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_20: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_3, view_452);  view_452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax_20: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(where_20, [-1], True)
        sub_61: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_20, amax_20);  where_20 = amax_20 = None
        exp_20: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_61);  sub_61 = None
        sum_21: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_41: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        inductor_lookup_seed_default_61: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 61)
        inductor_random_default_11: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default([8, 24, 512, 512], inductor_lookup_seed_default_61, 'rand');  inductor_lookup_seed_default_61 = None
        gt_61: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_11, 0.1);  inductor_random_default_11 = None
        mul_287: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_61, div_41)
        mul_288: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_287, 1.1111111111111112);  mul_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_453: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(mul_288, [-1, 512, 512]);  mul_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_41: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_453, view_451)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_454: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_41, [-1, 24, 512, 64]);  bmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_227: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_454, [0, 2, 1, 3]);  view_454 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_83: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_455: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_83, [8, 512, -1]);  clone_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_456: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_455, [4096, 1536]);  view_455 = None
        permute_228: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_333, [1, 0])
        addmm_123: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_334, view_456, permute_228);  primals_334 = permute_228 = None
        view_457: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_123, [8, 512, 1536]);  addmm_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_62: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 62)
        inductor_random_default_10: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_62, 'rand');  inductor_lookup_seed_default_62 = None
        gt_62: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_10, 0.1);  inductor_random_default_10 = None
        mul_289: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_62, view_457);  view_457 = None
        mul_290: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_289, 1.1111111111111112);  mul_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_143: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_290, add_142);  mul_290 = add_142 = None
        var_mean_41 = torch.ops.aten.var_mean.correction(add_143, [2], correction = 0, keepdim = True)
        getitem_82: "f32[8, 512, 1]" = var_mean_41[0]
        getitem_83: "f32[8, 512, 1]" = var_mean_41[1];  var_mean_41 = None
        add_144: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-07);  getitem_82 = None
        rsqrt_41: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
        sub_62: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_143, getitem_83);  add_143 = getitem_83 = None
        mul_291: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_41);  sub_62 = None
        mul_292: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_291, primals_335)
        add_145: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_292, primals_336);  mul_292 = primals_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_458: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_145, [4096, 1536])
        permute_229: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_337, [1, 0])
        addmm_124: "f32[4096, 6144]" = torch.ops.aten.addmm.default(primals_338, view_458, permute_229);  primals_338 = permute_229 = None
        view_459: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_124, [8, 512, 6144])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_293: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_459, 0.5)
        mul_294: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_459, 0.7071067811865476);  view_459 = None
        erf_20: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_294);  mul_294 = None
        add_146: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_295: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_293, add_146);  mul_293 = add_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_460: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_295, [4096, 6144]);  mul_295 = None
        permute_230: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_339, [1, 0])
        addmm_125: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_340, view_460, permute_230);  primals_340 = permute_230 = None
        view_461: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_125, [8, 512, 1536]);  addmm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_63: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 63)
        inductor_random_default_9: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_63, 'rand');  inductor_lookup_seed_default_63 = None
        gt_63: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_9, 0.1);  inductor_random_default_9 = None
        mul_296: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_63, view_461);  view_461 = None
        mul_297: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_296, 1.1111111111111112);  mul_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_147: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_297, add_145);  mul_297 = add_145 = None
        var_mean_42 = torch.ops.aten.var_mean.correction(add_147, [2], correction = 0, keepdim = True)
        getitem_84: "f32[8, 512, 1]" = var_mean_42[0]
        getitem_85: "f32[8, 512, 1]" = var_mean_42[1];  var_mean_42 = None
        add_148: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-07);  getitem_84 = None
        rsqrt_42: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
        sub_63: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_147, getitem_85);  add_147 = getitem_85 = None
        mul_298: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_42);  sub_63 = None
        mul_299: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_298, primals_341)
        add_149: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_299, primals_342);  mul_299 = primals_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_462: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_149, [4096, 1536])
        permute_231: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_343, [1, 0])
        addmm_126: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_344, view_462, permute_231);  primals_344 = permute_231 = None
        view_463: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_126, [8, 512, 1536]);  addmm_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_464: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_463, [8, 512, 24, -1]);  view_463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_232: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_464, [0, 2, 1, 3]);  view_464 = None
        clone_84: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_232, memory_format = torch.contiguous_format);  permute_232 = None
        view_465: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_84, [-1, 512, 64]);  clone_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_233: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_345, [1, 0])
        addmm_127: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_346, view_462, permute_233);  primals_346 = permute_233 = None
        view_467: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_127, [8, 512, 1536]);  addmm_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_468: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_467, [8, 512, 24, -1]);  view_467 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_234: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_468, [0, 2, 1, 3]);  view_468 = None
        clone_85: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_234, memory_format = torch.contiguous_format);  permute_234 = None
        view_469: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_85, [-1, 512, 64]);  clone_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_235: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_347, [1, 0])
        addmm_128: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_348, view_462, permute_235);  primals_348 = permute_235 = None
        view_471: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_128, [8, 512, 1536]);  addmm_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_472: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_471, [8, 512, 24, -1]);  view_471 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_236: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_472, [0, 2, 1, 3]);  view_472 = None
        clone_86: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_236, memory_format = torch.contiguous_format);  permute_236 = None
        view_473: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_86, [-1, 512, 64]);  clone_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_237: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_469, [0, 2, 1]);  view_469 = None
        div_42: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_237, full_default_1);  permute_237 = None
        bmm_42: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_465, div_42)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_474: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_42, [-1, 24, 512, 512]);  bmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_21: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_3, view_474);  view_474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax_21: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(where_21, [-1], True)
        sub_64: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_21, amax_21);  where_21 = amax_21 = None
        exp_21: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_64);  sub_64 = None
        sum_22: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_43: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        inductor_lookup_seed_default_64: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 64)
        inductor_random_default_8: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default([8, 24, 512, 512], inductor_lookup_seed_default_64, 'rand');  inductor_lookup_seed_default_64 = None
        gt_64: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_8, 0.1);  inductor_random_default_8 = None
        mul_301: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_64, div_43)
        mul_302: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_301, 1.1111111111111112);  mul_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_475: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(mul_302, [-1, 512, 512]);  mul_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_43: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_475, view_473)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_476: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_43, [-1, 24, 512, 64]);  bmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_238: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_476, [0, 2, 1, 3]);  view_476 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_87: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_477: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_87, [8, 512, -1]);  clone_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_478: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_477, [4096, 1536]);  view_477 = None
        permute_239: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_349, [1, 0])
        addmm_129: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_350, view_478, permute_239);  primals_350 = permute_239 = None
        view_479: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_129, [8, 512, 1536]);  addmm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_65: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 65)
        inductor_random_default_7: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_65, 'rand');  inductor_lookup_seed_default_65 = None
        gt_65: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_7, 0.1);  inductor_random_default_7 = None
        mul_303: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_65, view_479);  view_479 = None
        mul_304: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_303, 1.1111111111111112);  mul_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_150: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_304, add_149);  mul_304 = add_149 = None
        var_mean_43 = torch.ops.aten.var_mean.correction(add_150, [2], correction = 0, keepdim = True)
        getitem_86: "f32[8, 512, 1]" = var_mean_43[0]
        getitem_87: "f32[8, 512, 1]" = var_mean_43[1];  var_mean_43 = None
        add_151: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-07);  getitem_86 = None
        rsqrt_43: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
        sub_65: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_150, getitem_87);  add_150 = getitem_87 = None
        mul_305: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_43);  sub_65 = None
        mul_306: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_305, primals_351)
        add_152: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_306, primals_352);  mul_306 = primals_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_480: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_152, [4096, 1536])
        permute_240: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_353, [1, 0])
        addmm_130: "f32[4096, 6144]" = torch.ops.aten.addmm.default(primals_354, view_480, permute_240);  primals_354 = permute_240 = None
        view_481: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_130, [8, 512, 6144])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_307: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_481, 0.5)
        mul_308: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_481, 0.7071067811865476);  view_481 = None
        erf_21: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_308);  mul_308 = None
        add_153: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_309: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_307, add_153);  mul_307 = add_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_482: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_309, [4096, 6144]);  mul_309 = None
        permute_241: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_355, [1, 0])
        addmm_131: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_356, view_482, permute_241);  primals_356 = permute_241 = None
        view_483: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_131, [8, 512, 1536]);  addmm_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_66: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 66)
        inductor_random_default_6: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_66, 'rand');  inductor_lookup_seed_default_66 = None
        gt_66: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_6, 0.1);  inductor_random_default_6 = None
        mul_310: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_66, view_483);  view_483 = None
        mul_311: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_310, 1.1111111111111112);  mul_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_154: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_311, add_152);  mul_311 = add_152 = None
        var_mean_44 = torch.ops.aten.var_mean.correction(add_154, [2], correction = 0, keepdim = True)
        getitem_88: "f32[8, 512, 1]" = var_mean_44[0]
        getitem_89: "f32[8, 512, 1]" = var_mean_44[1];  var_mean_44 = None
        add_155: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-07);  getitem_88 = None
        rsqrt_44: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
        sub_66: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_154, getitem_89);  add_154 = getitem_89 = None
        mul_312: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_44);  sub_66 = None
        mul_313: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_312, primals_357)
        add_156: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_313, primals_358);  mul_313 = primals_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_484: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_156, [4096, 1536])
        permute_242: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_359, [1, 0])
        addmm_132: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_360, view_484, permute_242);  primals_360 = permute_242 = None
        view_485: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_132, [8, 512, 1536]);  addmm_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_486: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_485, [8, 512, 24, -1]);  view_485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_243: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_486, [0, 2, 1, 3]);  view_486 = None
        clone_88: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_243, memory_format = torch.contiguous_format);  permute_243 = None
        view_487: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_88, [-1, 512, 64]);  clone_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_244: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_361, [1, 0])
        addmm_133: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_362, view_484, permute_244);  primals_362 = permute_244 = None
        view_489: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_133, [8, 512, 1536]);  addmm_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_490: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_489, [8, 512, 24, -1]);  view_489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_245: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_490, [0, 2, 1, 3]);  view_490 = None
        clone_89: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_245, memory_format = torch.contiguous_format);  permute_245 = None
        view_491: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_89, [-1, 512, 64]);  clone_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_246: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_363, [1, 0])
        addmm_134: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_364, view_484, permute_246);  primals_364 = permute_246 = None
        view_493: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_134, [8, 512, 1536]);  addmm_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_494: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_493, [8, 512, 24, -1]);  view_493 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_247: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_494, [0, 2, 1, 3]);  view_494 = None
        clone_90: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_247, memory_format = torch.contiguous_format);  permute_247 = None
        view_495: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_90, [-1, 512, 64]);  clone_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_248: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_491, [0, 2, 1]);  view_491 = None
        div_44: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_248, full_default_1);  permute_248 = None
        bmm_44: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_487, div_44)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_496: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_44, [-1, 24, 512, 512]);  bmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_22: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_3, view_496);  view_496 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax_22: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(where_22, [-1], True)
        sub_67: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_22, amax_22);  where_22 = amax_22 = None
        exp_22: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_67);  sub_67 = None
        sum_23: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_45: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        inductor_lookup_seed_default_67: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 67)
        inductor_random_default_5: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default([8, 24, 512, 512], inductor_lookup_seed_default_67, 'rand');  inductor_lookup_seed_default_67 = None
        gt_67: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_5, 0.1);  inductor_random_default_5 = None
        mul_315: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_67, div_45)
        mul_316: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_315, 1.1111111111111112);  mul_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_497: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(mul_316, [-1, 512, 512]);  mul_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_45: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_497, view_495)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_498: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_45, [-1, 24, 512, 64]);  bmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_249: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_498, [0, 2, 1, 3]);  view_498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_91: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_499: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_91, [8, 512, -1]);  clone_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_500: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_499, [4096, 1536]);  view_499 = None
        permute_250: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_365, [1, 0])
        addmm_135: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_366, view_500, permute_250);  primals_366 = permute_250 = None
        view_501: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_135, [8, 512, 1536]);  addmm_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_68: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 68)
        inductor_random_default_4: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_68, 'rand');  inductor_lookup_seed_default_68 = None
        gt_68: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_4, 0.1);  inductor_random_default_4 = None
        mul_317: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_68, view_501);  view_501 = None
        mul_318: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_317, 1.1111111111111112);  mul_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_157: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_318, add_156);  mul_318 = add_156 = None
        var_mean_45 = torch.ops.aten.var_mean.correction(add_157, [2], correction = 0, keepdim = True)
        getitem_90: "f32[8, 512, 1]" = var_mean_45[0]
        getitem_91: "f32[8, 512, 1]" = var_mean_45[1];  var_mean_45 = None
        add_158: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-07);  getitem_90 = None
        rsqrt_45: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
        sub_68: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_157, getitem_91);  add_157 = getitem_91 = None
        mul_319: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_45);  sub_68 = None
        mul_320: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_319, primals_367)
        add_159: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_320, primals_368);  mul_320 = primals_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_502: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_159, [4096, 1536])
        permute_251: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_369, [1, 0])
        addmm_136: "f32[4096, 6144]" = torch.ops.aten.addmm.default(primals_370, view_502, permute_251);  primals_370 = permute_251 = None
        view_503: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_136, [8, 512, 6144])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_321: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_503, 0.5)
        mul_322: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_503, 0.7071067811865476);  view_503 = None
        erf_22: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_322);  mul_322 = None
        add_160: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_323: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_321, add_160);  mul_321 = add_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_504: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_323, [4096, 6144]);  mul_323 = None
        permute_252: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_371, [1, 0])
        addmm_137: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_372, view_504, permute_252);  primals_372 = permute_252 = None
        view_505: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_137, [8, 512, 1536]);  addmm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_69: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 69)
        inductor_random_default_3: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_69, 'rand');  inductor_lookup_seed_default_69 = None
        gt_69: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_3, 0.1);  inductor_random_default_3 = None
        mul_324: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_69, view_505);  view_505 = None
        mul_325: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_324, 1.1111111111111112);  mul_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_161: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_325, add_159);  mul_325 = add_159 = None
        var_mean_46 = torch.ops.aten.var_mean.correction(add_161, [2], correction = 0, keepdim = True)
        getitem_92: "f32[8, 512, 1]" = var_mean_46[0]
        getitem_93: "f32[8, 512, 1]" = var_mean_46[1];  var_mean_46 = None
        add_162: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-07);  getitem_92 = None
        rsqrt_46: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
        sub_69: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_161, getitem_93);  add_161 = getitem_93 = None
        mul_326: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_46);  sub_69 = None
        mul_327: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_326, primals_373)
        add_163: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_327, primals_374);  mul_327 = primals_374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_506: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_163, [4096, 1536])
        permute_253: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_375, [1, 0])
        addmm_138: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_376, view_506, permute_253);  primals_376 = permute_253 = None
        view_507: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_138, [8, 512, 1536]);  addmm_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_508: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_507, [8, 512, 24, -1]);  view_507 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_254: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_508, [0, 2, 1, 3]);  view_508 = None
        clone_92: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_254, memory_format = torch.contiguous_format);  permute_254 = None
        view_509: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_92, [-1, 512, 64]);  clone_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_255: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_377, [1, 0])
        addmm_139: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_378, view_506, permute_255);  primals_378 = permute_255 = None
        view_511: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_139, [8, 512, 1536]);  addmm_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_512: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_511, [8, 512, 24, -1]);  view_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_256: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_512, [0, 2, 1, 3]);  view_512 = None
        clone_93: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_256, memory_format = torch.contiguous_format);  permute_256 = None
        view_513: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_93, [-1, 512, 64]);  clone_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_257: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_379, [1, 0])
        addmm_140: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_380, view_506, permute_257);  primals_380 = permute_257 = None
        view_515: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_140, [8, 512, 1536]);  addmm_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_516: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_515, [8, 512, 24, -1]);  view_515 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_258: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_516, [0, 2, 1, 3]);  view_516 = None
        clone_94: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_258, memory_format = torch.contiguous_format);  permute_258 = None
        view_517: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_94, [-1, 512, 64]);  clone_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_259: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_513, [0, 2, 1]);  view_513 = None
        div_46: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_259, full_default_1);  permute_259 = full_default_1 = None
        bmm_46: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_509, div_46)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_518: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_46, [-1, 24, 512, 512]);  bmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_23: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_3, view_518);  full_default_2 = full_default_3 = view_518 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax_23: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(where_23, [-1], True)
        sub_70: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_23, amax_23);  where_23 = amax_23 = None
        exp_23: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_70);  sub_70 = None
        sum_24: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_47: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        inductor_lookup_seed_default_70: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 70)
        inductor_random_default_2: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default([8, 24, 512, 512], inductor_lookup_seed_default_70, 'rand');  inductor_lookup_seed_default_70 = None
        gt_70: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_2, 0.1);  inductor_random_default_2 = None
        mul_329: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_70, div_47)
        mul_330: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_329, 1.1111111111111112);  mul_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_519: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(mul_330, [-1, 512, 512]);  mul_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_47: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_519, view_517)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        view_520: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_47, [-1, 24, 512, 64]);  bmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_260: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_520, [0, 2, 1, 3]);  view_520 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:268 in forward, code: .contiguous()
        clone_95: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_521: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_95, [8, 512, -1]);  clone_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_522: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_521, [4096, 1536]);  view_521 = None
        permute_261: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_381, [1, 0])
        addmm_141: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_382, view_522, permute_261);  primals_382 = permute_261 = None
        view_523: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_141, [8, 512, 1536]);  addmm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_71: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 71)
        inductor_random_default_1: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_71, 'rand');  inductor_lookup_seed_default_71 = None
        gt_71: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.1);  inductor_random_default_1 = None
        mul_331: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_71, view_523);  view_523 = None
        mul_332: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_331, 1.1111111111111112);  mul_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_164: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_332, add_163);  mul_332 = add_163 = None
        var_mean_47 = torch.ops.aten.var_mean.correction(add_164, [2], correction = 0, keepdim = True)
        getitem_94: "f32[8, 512, 1]" = var_mean_47[0]
        getitem_95: "f32[8, 512, 1]" = var_mean_47[1];  var_mean_47 = None
        add_165: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-07);  getitem_94 = None
        rsqrt_47: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
        sub_71: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_164, getitem_95);  add_164 = getitem_95 = None
        mul_333: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_47);  sub_71 = None
        mul_334: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_333, primals_383)
        add_166: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_334, primals_384);  mul_334 = primals_384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_524: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_166, [4096, 1536])
        permute_262: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_385, [1, 0])
        addmm_142: "f32[4096, 6144]" = torch.ops.aten.addmm.default(primals_386, view_524, permute_262);  primals_386 = permute_262 = None
        view_525: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_142, [8, 512, 6144])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_335: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_525, 0.5)
        mul_336: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_525, 0.7071067811865476);  view_525 = None
        erf_23: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_336);  mul_336 = None
        add_167: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_337: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_335, add_167);  mul_335 = add_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_526: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_337, [4096, 6144]);  mul_337 = None
        permute_263: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_387, [1, 0])
        addmm_143: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_388, view_526, permute_263);  primals_388 = permute_263 = None
        view_527: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_143, [8, 512, 1536]);  addmm_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_72: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 72);  inductor_seeds_default = None
        inductor_random_default: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default_72, 'rand');  inductor_lookup_seed_default_72 = None
        gt_72: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_338: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_72, view_527);  view_527 = None
        mul_339: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_338, 1.1111111111111112);  mul_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_168: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_339, add_166);  mul_339 = add_166 = None
        var_mean_48 = torch.ops.aten.var_mean.correction(add_168, [2], correction = 0, keepdim = True)
        getitem_96: "f32[8, 512, 1]" = var_mean_48[0]
        getitem_97: "f32[8, 512, 1]" = var_mean_48[1];  var_mean_48 = None
        add_169: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-07);  getitem_96 = None
        rsqrt_48: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_169);  add_169 = None
        sub_72: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_168, getitem_97);  add_168 = getitem_97 = None
        mul_340: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_48);  sub_72 = None
        mul_341: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_340, primals_389)
        add_170: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_341, primals_390);  mul_341 = primals_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:818 in forward, code: hidden_states = self.dense(hidden_states)
        view_528: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_170, [4096, 1536]);  add_170 = None
        permute_264: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_391, [1, 0])
        addmm_144: "f32[4096, 1536]" = torch.ops.aten.addmm.default(primals_392, view_528, permute_264);  primals_392 = permute_264 = None
        view_529: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_144, [8, 512, 1536])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_342: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(view_529, 0.5)
        mul_343: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(view_529, 0.7071067811865476);  view_529 = None
        erf_24: "f32[8, 512, 1536]" = torch.ops.aten.erf.default(mul_343);  mul_343 = None
        add_171: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
        mul_344: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_342, add_171);  mul_342 = add_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:820 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        var_mean_49 = torch.ops.aten.var_mean.correction(mul_344, [2], correction = 0, keepdim = True)
        getitem_98: "f32[8, 512, 1]" = var_mean_49[0]
        getitem_99: "f32[8, 512, 1]" = var_mean_49[1];  var_mean_49 = None
        add_172: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-07);  getitem_98 = None
        rsqrt_49: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
        sub_73: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_344, getitem_99);  mul_344 = None
        mul_345: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_49);  sub_73 = None
        mul_346: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_345, primals_393);  mul_345 = None
        add_173: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_346, primals_394);  mul_346 = primals_394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:838 in forward, code: hidden_states = self.decoder(hidden_states)
        view_530: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_173, [4096, 1536]);  add_173 = None
        permute_265: "f32[1536, 128100]" = torch.ops.aten.permute.default(primals_3, [1, 0])
        addmm_145: "f32[4096, 128100]" = torch.ops.aten.addmm.default(primals_395, view_530, permute_265);  primals_395 = permute_265 = None
        view_531: "f32[8, 512, 128100]" = torch.ops.aten.reshape.default(addmm_145, [8, 512, 128100]);  addmm_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:968 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        view_532: "f32[4096, 128100]" = torch.ops.aten.reshape.default(view_531, [-1, 128100])
        view_533: "i64[4096]" = torch.ops.aten.reshape.default(primals_396, [-1])
        amax_24: "f32[4096, 1]" = torch.ops.aten.amax.default(view_532, [1], True)
        sub_74: "f32[4096, 128100]" = torch.ops.aten.sub.Tensor(view_532, amax_24);  view_532 = None
        exp_24: "f32[4096, 128100]" = torch.ops.aten.exp.default(sub_74)
        sum_25: "f32[4096, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log: "f32[4096, 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_75: "f32[4096, 128100]" = torch.ops.aten.sub.Tensor(sub_74, log);  sub_74 = None
        ne: "b8[4096]" = torch.ops.aten.ne.Scalar(view_533, -100)
        full_default_73: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_24: "i64[4096]" = torch.ops.aten.where.self(ne, view_533, full_default_73);  view_533 = full_default_73 = None
        unsqueeze_4: "i64[4096, 1]" = torch.ops.aten.unsqueeze.default(where_24, 1);  where_24 = None
        gather: "f32[4096, 1]" = torch.ops.aten.gather.default(sub_75, 1, unsqueeze_4);  sub_75 = unsqueeze_4 = None
        squeeze_1: "f32[4096]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[4096]" = torch.ops.aten.neg.default(squeeze_1);  squeeze_1 = None
        full_default_74: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_25: "f32[4096]" = torch.ops.aten.where.self(ne, neg, full_default_74);  neg = full_default_74 = None
        sum_26: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type_24: "f32[]" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
        sum_27: "f32[]" = torch.ops.aten.sum.default(where_25);  where_25 = None
        div_48: "f32[]" = torch.ops.aten.div.Tensor(sum_27, convert_element_type_24);  sum_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_51: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_48, 1536);  rsqrt_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_52: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_47, 1536);  rsqrt_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_287: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_519, [0, 2, 1]);  view_519 = None
        permute_288: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_517, [0, 2, 1]);  view_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_289: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_509, [0, 2, 1]);  view_509 = None
        permute_290: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_46, [0, 2, 1]);  div_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_54: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 1536);  rsqrt_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_55: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 1536);  rsqrt_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_320: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_497, [0, 2, 1]);  view_497 = None
        permute_321: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_495, [0, 2, 1]);  view_495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_322: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_487, [0, 2, 1]);  view_487 = None
        permute_323: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_44, [0, 2, 1]);  div_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_57: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 1536);  rsqrt_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_58: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 1536);  rsqrt_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_353: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_475, [0, 2, 1]);  view_475 = None
        permute_354: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_473, [0, 2, 1]);  view_473 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_355: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_465, [0, 2, 1]);  view_465 = None
        permute_356: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_42, [0, 2, 1]);  div_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_60: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_42, 1536);  rsqrt_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_61: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_41, 1536);  rsqrt_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_386: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_453, [0, 2, 1]);  view_453 = None
        permute_387: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_451, [0, 2, 1]);  view_451 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_388: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_443, [0, 2, 1]);  view_443 = None
        permute_389: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_40, [0, 2, 1]);  div_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_63: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 1536);  rsqrt_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_64: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 1536);  rsqrt_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_419: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_431, [0, 2, 1]);  view_431 = None
        permute_420: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_429, [0, 2, 1]);  view_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_421: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_421, [0, 2, 1]);  view_421 = None
        permute_422: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_38, [0, 2, 1]);  div_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_66: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 1536);  rsqrt_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_67: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 1536);  rsqrt_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_452: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_409, [0, 2, 1]);  view_409 = None
        permute_453: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_407, [0, 2, 1]);  view_407 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_454: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_399, [0, 2, 1]);  view_399 = None
        permute_455: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_36, [0, 2, 1]);  div_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_69: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 1536);  rsqrt_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_70: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 1536);  rsqrt_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_485: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_387, [0, 2, 1]);  view_387 = None
        permute_486: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_385, [0, 2, 1]);  view_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_487: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_377, [0, 2, 1]);  view_377 = None
        permute_488: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_34, [0, 2, 1]);  div_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_72: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 1536);  rsqrt_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_73: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 1536);  rsqrt_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_518: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_365, [0, 2, 1]);  view_365 = None
        permute_519: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_363, [0, 2, 1]);  view_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_520: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_355, [0, 2, 1]);  view_355 = None
        permute_521: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_32, [0, 2, 1]);  div_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_75: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 1536);  rsqrt_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_76: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 1536);  rsqrt_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_551: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_343, [0, 2, 1]);  view_343 = None
        permute_552: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_341, [0, 2, 1]);  view_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_553: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_333, [0, 2, 1]);  view_333 = None
        permute_554: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_30, [0, 2, 1]);  div_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_78: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 1536);  rsqrt_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_79: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 1536);  rsqrt_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_584: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_321, [0, 2, 1]);  view_321 = None
        permute_585: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_319, [0, 2, 1]);  view_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_586: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_311, [0, 2, 1]);  view_311 = None
        permute_587: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_28, [0, 2, 1]);  div_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_81: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 1536);  rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_82: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 1536);  rsqrt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_617: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_299, [0, 2, 1]);  view_299 = None
        permute_618: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_297, [0, 2, 1]);  view_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_619: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_289, [0, 2, 1]);  view_289 = None
        permute_620: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_26, [0, 2, 1]);  div_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_84: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 1536);  rsqrt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_85: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 1536);  rsqrt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_650: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_277, [0, 2, 1]);  view_277 = None
        permute_651: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_275, [0, 2, 1]);  view_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_652: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_267, [0, 2, 1]);  view_267 = None
        permute_653: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_24, [0, 2, 1]);  div_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_87: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 1536);  rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_88: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 1536);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_683: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_255, [0, 2, 1]);  view_255 = None
        permute_684: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_253, [0, 2, 1]);  view_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_685: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_245, [0, 2, 1]);  view_245 = None
        permute_686: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_22, [0, 2, 1]);  div_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_90: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 1536);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_91: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 1536);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_716: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_233, [0, 2, 1]);  view_233 = None
        permute_717: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_231, [0, 2, 1]);  view_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_718: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_223, [0, 2, 1]);  view_223 = None
        permute_719: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_20, [0, 2, 1]);  div_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_93: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 1536);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_94: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 1536);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_749: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_211, [0, 2, 1]);  view_211 = None
        permute_750: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_209, [0, 2, 1]);  view_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_751: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_201, [0, 2, 1]);  view_201 = None
        permute_752: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_18, [0, 2, 1]);  div_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_96: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 1536);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_97: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 1536);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_782: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None
        permute_783: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_187, [0, 2, 1]);  view_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_784: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_179, [0, 2, 1]);  view_179 = None
        permute_785: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_16, [0, 2, 1]);  div_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_99: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 1536);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_100: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 1536);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_815: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None
        permute_816: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_165, [0, 2, 1]);  view_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_817: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_157, [0, 2, 1]);  view_157 = None
        permute_818: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_14, [0, 2, 1]);  div_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_102: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 1536);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_103: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 1536);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_848: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None
        permute_849: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_143, [0, 2, 1]);  view_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_850: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_135, [0, 2, 1]);  view_135 = None
        permute_851: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_12, [0, 2, 1]);  div_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_105: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 1536);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_106: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 1536);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_881: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
        permute_882: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_121, [0, 2, 1]);  view_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_883: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_113, [0, 2, 1]);  view_113 = None
        permute_884: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_10, [0, 2, 1]);  div_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_108: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 1536);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_109: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 1536);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_914: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_101, [0, 2, 1]);  view_101 = None
        permute_915: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_916: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_91, [0, 2, 1]);  view_91 = None
        permute_917: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_8, [0, 2, 1]);  div_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_111: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 1536);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_112: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 1536);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_947: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None
        permute_948: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_77, [0, 2, 1]);  view_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_949: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_69, [0, 2, 1]);  view_69 = None
        permute_950: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_6, [0, 2, 1]);  div_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_114: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 1536);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_115: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 1536);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_980: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_57, [0, 2, 1]);  view_57 = None
        permute_981: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_55, [0, 2, 1]);  view_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_982: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_47, [0, 2, 1]);  view_47 = None
        permute_983: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_4, [0, 2, 1]);  div_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_117: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 1536);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_118: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 1536);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_1013: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_35, [0, 2, 1]);  view_35 = None
        permute_1014: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_1015: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_25, [0, 2, 1]);  view_25 = None
        permute_1016: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_2, [0, 2, 1]);  div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_120: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 1536);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_121: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 1536);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_1046: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
        permute_1047: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_1048: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_3, [0, 2, 1]);  view_3 = None
        permute_1049: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div, [0, 2, 1]);  div = None
        return (div_48, view_531, primals_1, primals_2, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, primals_149, primals_151, primals_153, primals_155, primals_157, primals_159, primals_161, primals_163, primals_165, primals_167, primals_169, primals_171, primals_173, primals_175, primals_177, primals_179, primals_181, primals_183, primals_185, primals_187, primals_189, primals_191, primals_193, primals_195, primals_197, primals_199, primals_201, primals_203, primals_205, primals_207, primals_209, primals_211, primals_213, primals_215, primals_217, primals_219, primals_221, primals_223, primals_225, primals_227, primals_229, primals_231, primals_233, primals_235, primals_237, primals_239, primals_241, primals_243, primals_245, primals_247, primals_249, primals_251, primals_253, primals_255, primals_257, primals_259, primals_261, primals_263, primals_265, primals_267, primals_269, primals_271, primals_273, primals_275, primals_277, primals_279, primals_281, primals_283, primals_285, primals_287, primals_289, primals_291, primals_293, primals_295, primals_297, primals_299, primals_301, primals_303, primals_305, primals_307, primals_309, primals_311, primals_313, primals_315, primals_317, primals_319, primals_321, primals_323, primals_325, primals_327, primals_329, primals_331, primals_333, primals_335, primals_337, primals_339, primals_341, primals_343, primals_345, primals_347, primals_349, primals_351, primals_353, primals_355, primals_357, primals_359, primals_361, primals_363, primals_365, primals_367, primals_369, primals_371, primals_373, primals_375, primals_377, primals_379, primals_381, primals_383, primals_385, primals_387, primals_389, primals_391, primals_393, primals_396, embedding, embedding_1, getitem_1, rsqrt, gt, view, bmm, amax, sum_1, gt_1, view_16, gt_2, mul_11, view_18, addmm_4, view_20, gt_3, mul_18, view_22, div_3, gt_4, view_38, gt_5, mul_25, view_40, addmm_10, view_42, gt_6, mul_32, view_44, div_5, gt_7, view_60, gt_8, mul_39, view_62, addmm_16, view_64, gt_9, mul_46, view_66, div_7, gt_10, view_82, gt_11, mul_53, view_84, addmm_22, view_86, gt_12, mul_60, view_88, div_9, gt_13, view_104, gt_14, mul_67, view_106, addmm_28, view_108, gt_15, mul_74, view_110, div_11, gt_16, view_126, gt_17, mul_81, view_128, addmm_34, view_130, gt_18, mul_88, view_132, div_13, gt_19, view_148, gt_20, mul_95, view_150, addmm_40, view_152, gt_21, mul_102, view_154, div_15, gt_22, view_170, gt_23, mul_109, view_172, addmm_46, view_174, gt_24, mul_116, view_176, div_17, gt_25, view_192, gt_26, mul_123, view_194, addmm_52, view_196, gt_27, mul_130, view_198, div_19, gt_28, view_214, gt_29, mul_137, view_216, addmm_58, view_218, gt_30, mul_144, view_220, div_21, gt_31, view_236, gt_32, mul_151, view_238, addmm_64, view_240, gt_33, mul_158, view_242, div_23, gt_34, view_258, gt_35, mul_165, view_260, addmm_70, view_262, gt_36, mul_172, view_264, div_25, gt_37, view_280, gt_38, mul_179, view_282, addmm_76, view_284, gt_39, mul_186, view_286, div_27, gt_40, view_302, gt_41, mul_193, view_304, addmm_82, view_306, gt_42, mul_200, view_308, div_29, gt_43, view_324, gt_44, mul_207, view_326, addmm_88, view_328, gt_45, mul_214, view_330, div_31, gt_46, view_346, gt_47, mul_221, view_348, addmm_94, view_350, gt_48, mul_228, view_352, div_33, gt_49, view_368, gt_50, mul_235, view_370, addmm_100, view_372, gt_51, mul_242, view_374, div_35, gt_52, view_390, gt_53, mul_249, view_392, addmm_106, view_394, gt_54, mul_256, view_396, div_37, gt_55, view_412, gt_56, mul_263, view_414, addmm_112, view_416, gt_57, mul_270, view_418, div_39, gt_58, view_434, gt_59, mul_277, view_436, addmm_118, view_438, gt_60, mul_284, view_440, div_41, gt_61, view_456, gt_62, mul_291, view_458, addmm_124, view_460, gt_63, mul_298, view_462, div_43, gt_64, view_478, gt_65, mul_305, view_480, addmm_130, view_482, gt_66, mul_312, view_484, div_45, gt_67, view_500, gt_68, mul_319, view_502, addmm_136, view_504, gt_69, mul_326, view_506, div_47, gt_70, view_522, gt_71, mul_333, view_524, addmm_142, view_526, gt_72, mul_340, view_528, addmm_144, getitem_99, rsqrt_49, view_530, view_531, amax_24, log, convert_element_type_24, div_51, div_52, permute_287, permute_288, permute_289, permute_290, div_54, div_55, permute_320, permute_321, permute_322, permute_323, div_57, div_58, permute_353, permute_354, permute_355, permute_356, div_60, div_61, permute_386, permute_387, permute_388, permute_389, div_63, div_64, permute_419, permute_420, permute_421, permute_422, div_66, div_67, permute_452, permute_453, permute_454, permute_455, div_69, div_70, permute_485, permute_486, permute_487, permute_488, div_72, div_73, permute_518, permute_519, permute_520, permute_521, div_75, div_76, permute_551, permute_552, permute_553, permute_554, div_78, div_79, permute_584, permute_585, permute_586, permute_587, div_81, div_82, permute_617, permute_618, permute_619, permute_620, div_84, div_85, permute_650, permute_651, permute_652, permute_653, div_87, div_88, permute_683, permute_684, permute_685, permute_686, div_90, div_91, permute_716, permute_717, permute_718, permute_719, div_93, div_94, permute_749, permute_750, permute_751, permute_752, div_96, div_97, permute_782, permute_783, permute_784, permute_785, div_99, div_100, permute_815, permute_816, permute_817, permute_818, div_102, div_103, permute_848, permute_849, permute_850, permute_851, div_105, div_106, permute_881, permute_882, permute_883, permute_884, div_108, div_109, permute_914, permute_915, permute_916, permute_917, div_111, div_112, permute_947, permute_948, permute_949, permute_950, div_114, div_115, permute_980, permute_981, permute_982, permute_983, div_117, div_118, permute_1013, permute_1014, permute_1015, permute_1016, div_120, div_121, permute_1046, permute_1047, permute_1048, permute_1049)
