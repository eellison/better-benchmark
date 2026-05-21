class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[4, 512]", primals_2: "f32[50257, 1280]", primals_3: "f32[1024, 1280]", primals_4: "f32[1280]", primals_5: "f32[1280]", primals_6: "f32[3840]", primals_7: "f32[1280, 3840]", primals_8: "f32[1280]", primals_9: "f32[1280, 1280]", primals_10: "f32[1280]", primals_11: "f32[1280]", primals_12: "f32[5120]", primals_13: "f32[1280, 5120]", primals_14: "f32[1280]", primals_15: "f32[5120, 1280]", primals_16: "f32[1280]", primals_17: "f32[1280]", primals_18: "f32[3840]", primals_19: "f32[1280, 3840]", primals_20: "f32[1280]", primals_21: "f32[1280, 1280]", primals_22: "f32[1280]", primals_23: "f32[1280]", primals_24: "f32[5120]", primals_25: "f32[1280, 5120]", primals_26: "f32[1280]", primals_27: "f32[5120, 1280]", primals_28: "f32[1280]", primals_29: "f32[1280]", primals_30: "f32[3840]", primals_31: "f32[1280, 3840]", primals_32: "f32[1280]", primals_33: "f32[1280, 1280]", primals_34: "f32[1280]", primals_35: "f32[1280]", primals_36: "f32[5120]", primals_37: "f32[1280, 5120]", primals_38: "f32[1280]", primals_39: "f32[5120, 1280]", primals_40: "f32[1280]", primals_41: "f32[1280]", primals_42: "f32[3840]", primals_43: "f32[1280, 3840]", primals_44: "f32[1280]", primals_45: "f32[1280, 1280]", primals_46: "f32[1280]", primals_47: "f32[1280]", primals_48: "f32[5120]", primals_49: "f32[1280, 5120]", primals_50: "f32[1280]", primals_51: "f32[5120, 1280]", primals_52: "f32[1280]", primals_53: "f32[1280]", primals_54: "f32[3840]", primals_55: "f32[1280, 3840]", primals_56: "f32[1280]", primals_57: "f32[1280, 1280]", primals_58: "f32[1280]", primals_59: "f32[1280]", primals_60: "f32[5120]", primals_61: "f32[1280, 5120]", primals_62: "f32[1280]", primals_63: "f32[5120, 1280]", primals_64: "f32[1280]", primals_65: "f32[1280]", primals_66: "f32[3840]", primals_67: "f32[1280, 3840]", primals_68: "f32[1280]", primals_69: "f32[1280, 1280]", primals_70: "f32[1280]", primals_71: "f32[1280]", primals_72: "f32[5120]", primals_73: "f32[1280, 5120]", primals_74: "f32[1280]", primals_75: "f32[5120, 1280]", primals_76: "f32[1280]", primals_77: "f32[1280]", primals_78: "f32[3840]", primals_79: "f32[1280, 3840]", primals_80: "f32[1280]", primals_81: "f32[1280, 1280]", primals_82: "f32[1280]", primals_83: "f32[1280]", primals_84: "f32[5120]", primals_85: "f32[1280, 5120]", primals_86: "f32[1280]", primals_87: "f32[5120, 1280]", primals_88: "f32[1280]", primals_89: "f32[1280]", primals_90: "f32[3840]", primals_91: "f32[1280, 3840]", primals_92: "f32[1280]", primals_93: "f32[1280, 1280]", primals_94: "f32[1280]", primals_95: "f32[1280]", primals_96: "f32[5120]", primals_97: "f32[1280, 5120]", primals_98: "f32[1280]", primals_99: "f32[5120, 1280]", primals_100: "f32[1280]", primals_101: "f32[1280]", primals_102: "f32[3840]", primals_103: "f32[1280, 3840]", primals_104: "f32[1280]", primals_105: "f32[1280, 1280]", primals_106: "f32[1280]", primals_107: "f32[1280]", primals_108: "f32[5120]", primals_109: "f32[1280, 5120]", primals_110: "f32[1280]", primals_111: "f32[5120, 1280]", primals_112: "f32[1280]", primals_113: "f32[1280]", primals_114: "f32[3840]", primals_115: "f32[1280, 3840]", primals_116: "f32[1280]", primals_117: "f32[1280, 1280]", primals_118: "f32[1280]", primals_119: "f32[1280]", primals_120: "f32[5120]", primals_121: "f32[1280, 5120]", primals_122: "f32[1280]", primals_123: "f32[5120, 1280]", primals_124: "f32[1280]", primals_125: "f32[1280]", primals_126: "f32[3840]", primals_127: "f32[1280, 3840]", primals_128: "f32[1280]", primals_129: "f32[1280, 1280]", primals_130: "f32[1280]", primals_131: "f32[1280]", primals_132: "f32[5120]", primals_133: "f32[1280, 5120]", primals_134: "f32[1280]", primals_135: "f32[5120, 1280]", primals_136: "f32[1280]", primals_137: "f32[1280]", primals_138: "f32[3840]", primals_139: "f32[1280, 3840]", primals_140: "f32[1280]", primals_141: "f32[1280, 1280]", primals_142: "f32[1280]", primals_143: "f32[1280]", primals_144: "f32[5120]", primals_145: "f32[1280, 5120]", primals_146: "f32[1280]", primals_147: "f32[5120, 1280]", primals_148: "f32[1280]", primals_149: "f32[1280]", primals_150: "f32[3840]", primals_151: "f32[1280, 3840]", primals_152: "f32[1280]", primals_153: "f32[1280, 1280]", primals_154: "f32[1280]", primals_155: "f32[1280]", primals_156: "f32[5120]", primals_157: "f32[1280, 5120]", primals_158: "f32[1280]", primals_159: "f32[5120, 1280]", primals_160: "f32[1280]", primals_161: "f32[1280]", primals_162: "f32[3840]", primals_163: "f32[1280, 3840]", primals_164: "f32[1280]", primals_165: "f32[1280, 1280]", primals_166: "f32[1280]", primals_167: "f32[1280]", primals_168: "f32[5120]", primals_169: "f32[1280, 5120]", primals_170: "f32[1280]", primals_171: "f32[5120, 1280]", primals_172: "f32[1280]", primals_173: "f32[1280]", primals_174: "f32[3840]", primals_175: "f32[1280, 3840]", primals_176: "f32[1280]", primals_177: "f32[1280, 1280]", primals_178: "f32[1280]", primals_179: "f32[1280]", primals_180: "f32[5120]", primals_181: "f32[1280, 5120]", primals_182: "f32[1280]", primals_183: "f32[5120, 1280]", primals_184: "f32[1280]", primals_185: "f32[1280]", primals_186: "f32[3840]", primals_187: "f32[1280, 3840]", primals_188: "f32[1280]", primals_189: "f32[1280, 1280]", primals_190: "f32[1280]", primals_191: "f32[1280]", primals_192: "f32[5120]", primals_193: "f32[1280, 5120]", primals_194: "f32[1280]", primals_195: "f32[5120, 1280]", primals_196: "f32[1280]", primals_197: "f32[1280]", primals_198: "f32[3840]", primals_199: "f32[1280, 3840]", primals_200: "f32[1280]", primals_201: "f32[1280, 1280]", primals_202: "f32[1280]", primals_203: "f32[1280]", primals_204: "f32[5120]", primals_205: "f32[1280, 5120]", primals_206: "f32[1280]", primals_207: "f32[5120, 1280]", primals_208: "f32[1280]", primals_209: "f32[1280]", primals_210: "f32[3840]", primals_211: "f32[1280, 3840]", primals_212: "f32[1280]", primals_213: "f32[1280, 1280]", primals_214: "f32[1280]", primals_215: "f32[1280]", primals_216: "f32[5120]", primals_217: "f32[1280, 5120]", primals_218: "f32[1280]", primals_219: "f32[5120, 1280]", primals_220: "f32[1280]", primals_221: "f32[1280]", primals_222: "f32[3840]", primals_223: "f32[1280, 3840]", primals_224: "f32[1280]", primals_225: "f32[1280, 1280]", primals_226: "f32[1280]", primals_227: "f32[1280]", primals_228: "f32[5120]", primals_229: "f32[1280, 5120]", primals_230: "f32[1280]", primals_231: "f32[5120, 1280]", primals_232: "f32[1280]", primals_233: "f32[1280]", primals_234: "f32[3840]", primals_235: "f32[1280, 3840]", primals_236: "f32[1280]", primals_237: "f32[1280, 1280]", primals_238: "f32[1280]", primals_239: "f32[1280]", primals_240: "f32[5120]", primals_241: "f32[1280, 5120]", primals_242: "f32[1280]", primals_243: "f32[5120, 1280]", primals_244: "f32[1280]", primals_245: "f32[1280]", primals_246: "f32[3840]", primals_247: "f32[1280, 3840]", primals_248: "f32[1280]", primals_249: "f32[1280, 1280]", primals_250: "f32[1280]", primals_251: "f32[1280]", primals_252: "f32[5120]", primals_253: "f32[1280, 5120]", primals_254: "f32[1280]", primals_255: "f32[5120, 1280]", primals_256: "f32[1280]", primals_257: "f32[1280]", primals_258: "f32[3840]", primals_259: "f32[1280, 3840]", primals_260: "f32[1280]", primals_261: "f32[1280, 1280]", primals_262: "f32[1280]", primals_263: "f32[1280]", primals_264: "f32[5120]", primals_265: "f32[1280, 5120]", primals_266: "f32[1280]", primals_267: "f32[5120, 1280]", primals_268: "f32[1280]", primals_269: "f32[1280]", primals_270: "f32[3840]", primals_271: "f32[1280, 3840]", primals_272: "f32[1280]", primals_273: "f32[1280, 1280]", primals_274: "f32[1280]", primals_275: "f32[1280]", primals_276: "f32[5120]", primals_277: "f32[1280, 5120]", primals_278: "f32[1280]", primals_279: "f32[5120, 1280]", primals_280: "f32[1280]", primals_281: "f32[1280]", primals_282: "f32[3840]", primals_283: "f32[1280, 3840]", primals_284: "f32[1280]", primals_285: "f32[1280, 1280]", primals_286: "f32[1280]", primals_287: "f32[1280]", primals_288: "f32[5120]", primals_289: "f32[1280, 5120]", primals_290: "f32[1280]", primals_291: "f32[5120, 1280]", primals_292: "f32[1280]", primals_293: "f32[1280]", primals_294: "f32[3840]", primals_295: "f32[1280, 3840]", primals_296: "f32[1280]", primals_297: "f32[1280, 1280]", primals_298: "f32[1280]", primals_299: "f32[1280]", primals_300: "f32[5120]", primals_301: "f32[1280, 5120]", primals_302: "f32[1280]", primals_303: "f32[5120, 1280]", primals_304: "f32[1280]", primals_305: "f32[1280]", primals_306: "f32[3840]", primals_307: "f32[1280, 3840]", primals_308: "f32[1280]", primals_309: "f32[1280, 1280]", primals_310: "f32[1280]", primals_311: "f32[1280]", primals_312: "f32[5120]", primals_313: "f32[1280, 5120]", primals_314: "f32[1280]", primals_315: "f32[5120, 1280]", primals_316: "f32[1280]", primals_317: "f32[1280]", primals_318: "f32[3840]", primals_319: "f32[1280, 3840]", primals_320: "f32[1280]", primals_321: "f32[1280, 1280]", primals_322: "f32[1280]", primals_323: "f32[1280]", primals_324: "f32[5120]", primals_325: "f32[1280, 5120]", primals_326: "f32[1280]", primals_327: "f32[5120, 1280]", primals_328: "f32[1280]", primals_329: "f32[1280]", primals_330: "f32[3840]", primals_331: "f32[1280, 3840]", primals_332: "f32[1280]", primals_333: "f32[1280, 1280]", primals_334: "f32[1280]", primals_335: "f32[1280]", primals_336: "f32[5120]", primals_337: "f32[1280, 5120]", primals_338: "f32[1280]", primals_339: "f32[5120, 1280]", primals_340: "f32[1280]", primals_341: "f32[1280]", primals_342: "f32[3840]", primals_343: "f32[1280, 3840]", primals_344: "f32[1280]", primals_345: "f32[1280, 1280]", primals_346: "f32[1280]", primals_347: "f32[1280]", primals_348: "f32[5120]", primals_349: "f32[1280, 5120]", primals_350: "f32[1280]", primals_351: "f32[5120, 1280]", primals_352: "f32[1280]", primals_353: "f32[1280]", primals_354: "f32[3840]", primals_355: "f32[1280, 3840]", primals_356: "f32[1280]", primals_357: "f32[1280, 1280]", primals_358: "f32[1280]", primals_359: "f32[1280]", primals_360: "f32[5120]", primals_361: "f32[1280, 5120]", primals_362: "f32[1280]", primals_363: "f32[5120, 1280]", primals_364: "f32[1280]", primals_365: "f32[1280]", primals_366: "f32[3840]", primals_367: "f32[1280, 3840]", primals_368: "f32[1280]", primals_369: "f32[1280, 1280]", primals_370: "f32[1280]", primals_371: "f32[1280]", primals_372: "f32[5120]", primals_373: "f32[1280, 5120]", primals_374: "f32[1280]", primals_375: "f32[5120, 1280]", primals_376: "f32[1280]", primals_377: "f32[1280]", primals_378: "f32[3840]", primals_379: "f32[1280, 3840]", primals_380: "f32[1280]", primals_381: "f32[1280, 1280]", primals_382: "f32[1280]", primals_383: "f32[1280]", primals_384: "f32[5120]", primals_385: "f32[1280, 5120]", primals_386: "f32[1280]", primals_387: "f32[5120, 1280]", primals_388: "f32[1280]", primals_389: "f32[1280]", primals_390: "f32[3840]", primals_391: "f32[1280, 3840]", primals_392: "f32[1280]", primals_393: "f32[1280, 1280]", primals_394: "f32[1280]", primals_395: "f32[1280]", primals_396: "f32[5120]", primals_397: "f32[1280, 5120]", primals_398: "f32[1280]", primals_399: "f32[5120, 1280]", primals_400: "f32[1280]", primals_401: "f32[1280]", primals_402: "f32[3840]", primals_403: "f32[1280, 3840]", primals_404: "f32[1280]", primals_405: "f32[1280, 1280]", primals_406: "f32[1280]", primals_407: "f32[1280]", primals_408: "f32[5120]", primals_409: "f32[1280, 5120]", primals_410: "f32[1280]", primals_411: "f32[5120, 1280]", primals_412: "f32[1280]", primals_413: "f32[1280]", primals_414: "f32[3840]", primals_415: "f32[1280, 3840]", primals_416: "f32[1280]", primals_417: "f32[1280, 1280]", primals_418: "f32[1280]", primals_419: "f32[1280]", primals_420: "f32[5120]", primals_421: "f32[1280, 5120]", primals_422: "f32[1280]", primals_423: "f32[5120, 1280]", primals_424: "f32[1280]", primals_425: "f32[1280]", primals_426: "f32[3840]", primals_427: "f32[1280, 3840]", primals_428: "f32[1280]", primals_429: "f32[1280, 1280]", primals_430: "f32[1280]", primals_431: "f32[1280]", primals_432: "f32[5120]", primals_433: "f32[1280, 5120]", primals_434: "f32[1280]", primals_435: "f32[5120, 1280]", primals_436: "f32[1280]", primals_437: "f32[1280]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:577 in forward, code: inputs_embeds = self.wte(input_ids)
        embedding: "f32[4, 512, 1280]" = torch.ops.aten.embedding.default(primals_2, primals_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:581 in forward, code: position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        iota: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[512]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:582 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:584 in forward, code: position_embeds = self.wpe(position_ids)
        embedding_1: "f32[1, 512, 1280]" = torch.ops.aten.embedding.default(primals_3, unsqueeze);  primals_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:585 in forward, code: hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)
        add_1: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:875 in _preprocess_mask_arguments, code: position_ids = position_ids.expand(batch_size, -1)
        expand: "i64[4, 512]" = torch.ops.aten.expand.default(unsqueeze, [4, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:765 in find_packed_sequence_indices, code: first_dummy_value = position_ids[:, :1] - 1  # We just need the diff on this first value to be 1
        slice_1: "i64[4, 1]" = torch.ops.aten.slice.Tensor(expand, 1, 0, 1)
        sub: "i64[4, 1]" = torch.ops.aten.sub.Tensor(slice_1, 1);  slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:766 in find_packed_sequence_indices, code: position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
        cat: "i64[4, 513]" = torch.ops.aten.cat.default([sub, expand], -1);  sub = expand = None
        slice_2: "i64[4, 512]" = torch.ops.aten.slice.Tensor(cat, -1, 0, 512)
        slice_3: "i64[4, 512]" = torch.ops.aten.slice.Tensor(cat, -1, 1, 513);  cat = None
        sub_1: "i64[4, 512]" = torch.ops.aten.sub.Tensor(slice_3, slice_2);  slice_3 = slice_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:767 in find_packed_sequence_indices, code: packed_sequence_mask = (position_diff != 1).cumsum(-1)
        ne: "b8[4, 512]" = torch.ops.aten.ne.Scalar(sub_1, 1);  sub_1 = None
        cumsum: "i64[4, 512]" = torch.ops.aten.cumsum.default(ne, -1);  ne = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:509 in sdpa_mask, code: batch_arange = torch.arange(batch_size, device=device)
        iota_1: "i64[4]" = torch.ops.prims.iota.default(4, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:360 in _non_vmap_expansion_sdpa, code: batch_indices = batch_indices[:, None, None, None]
        unsqueeze_1: "i64[4, 1]" = torch.ops.aten.unsqueeze.default(iota_1, 1);  iota_1 = None
        unsqueeze_2: "i64[4, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None
        unsqueeze_3: "i64[4, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 3);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_5: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1)
        unsqueeze_6: "i64[1, 1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 3)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_9: "i64[1, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 2);  unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:52 in and_mask, code: result = q_idx.new_ones((), dtype=torch.bool)
        full_default: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 512, 512]" = torch.ops.aten.le.Tensor(unsqueeze_9, unsqueeze_6)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_and.Tensor(full_default, le);  full_default = le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:168 in inner_mask, code: return packed_sequence_mask[batch_idx, q_idx] == packed_sequence_mask[batch_idx, kv_idx]
        index: "i64[4, 1, 512, 1]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_6]);  unsqueeze_6 = None
        index_1: "i64[4, 1, 1, 512]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_9]);  cumsum = unsqueeze_3 = unsqueeze_9 = None
        eq: "b8[4, 1, 512, 512]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and_1: "b8[4, 1, 512, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, eq);  bitwise_and = eq = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_1: "b8[4, 1, 512, 512]" = torch.ops.aten.expand.default(bitwise_and_1, [4, -1, 512, 512]);  bitwise_and_1 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[73]" = torch.ops.prims.inductor_seeds.default(73, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:612 in forward, code: hidden_states = self.drop(hidden_states)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_72: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_72, 0.1);  inductor_random_default_72 = None
        mul: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt, add_1);  add_1 = None
        mul_1: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(mul_1, [2], correction = 0, keepdim = True)
        getitem: "f32[4, 512, 1]" = var_mean[0]
        getitem_1: "f32[4, 512, 1]" = var_mean[1];  var_mean = None
        add_4: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_2: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_1, getitem_1);  getitem_1 = None
        mul_2: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = None
        mul_3: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_2, primals_4)
        add_5: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_3, primals_5);  mul_3 = primals_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_1: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_5, [-1, 1280]);  add_5 = None
        addmm: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_6, view_1, primals_7);  primals_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_2: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm, [4, 512, 3840]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split = torch.ops.aten.split.Tensor(view_2, 1280, 2);  view_2 = None
        getitem_2: "f32[4, 512, 1280]" = split[0]
        getitem_3: "f32[4, 512, 1280]" = split[1]
        getitem_4: "f32[4, 512, 1280]" = split[2];  split = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_3: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_3, [4, 512, -1, 64]);  getitem_3 = None
        permute: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_4: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_4, [4, 512, -1, 64]);  getitem_4 = None
        permute_1: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_5: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_2, [4, 512, -1, 64]);  getitem_2 = None
        permute_2: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_1, full_default_2, full_default_1);  expand_1 = full_default_2 = full_default_1 = None
        expand_2: "f32[4, 20, 512, 512]" = torch.ops.aten.expand.default(where, [4, 20, 512, 512])
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_2, permute, permute_1, expand_2, True, 0.1, scale = 0.125)
        getitem_5: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention[0]
        getitem_6: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention[1]
        getitem_7: "i64[]" = _scaled_dot_product_efficient_attention[2]
        getitem_8: "i64[]" = _scaled_dot_product_efficient_attention[3];  _scaled_dot_product_efficient_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_3: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_6: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_3, [4, 512, -1]);  permute_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_7: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_6, [-1, 1280]);  view_6 = None
        addmm_1: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_8, view_7, primals_9);  primals_8 = view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_8: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_1, [4, 512, 1280]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_71: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_1: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_71, 0.1);  inductor_random_default_71 = None
        mul_4: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_1, view_8);  view_8 = None
        mul_5: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_4, 1.1111111111111112);  mul_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_6: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_5, mul_1);  mul_5 = mul_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_6, [2], correction = 0, keepdim = True)
        getitem_9: "f32[4, 512, 1]" = var_mean_1[0]
        getitem_10: "f32[4, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        add_7: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_9, 1e-05);  getitem_9 = None
        rsqrt_1: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        sub_3: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_6, getitem_10);  getitem_10 = None
        mul_6: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = None
        mul_7: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_6, primals_10)
        add_8: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_7, primals_11);  mul_7 = primals_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_9: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_8, [-1, 1280]);  add_8 = None
        addmm_2: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_12, view_9, primals_13);  primals_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_10: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_2, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_8: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_10, 0.5)
        pow_1: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_10, 3.0)
        mul_9: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_9: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_10, mul_9);  view_10 = mul_9 = None
        mul_10: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_9, 0.7978845608028654);  add_9 = None
        tanh: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_10);  mul_10 = None
        add_10: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_11: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_8, add_10);  mul_8 = add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_11: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_11, [-1, 5120]);  mul_11 = None
        addmm_3: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_14, view_11, primals_15);  primals_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_12: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_3, [4, 512, 1280]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_70: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        gt_2: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_70, 0.1);  inductor_random_default_70 = None
        mul_12: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_2, view_12);  view_12 = None
        mul_13: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_12, 1.1111111111111112);  mul_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_11: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_6, mul_13);  add_6 = mul_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_11: "f32[4, 512, 1]" = var_mean_2[0]
        getitem_12: "f32[4, 512, 1]" = var_mean_2[1];  var_mean_2 = None
        add_12: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_2: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        sub_4: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_11, getitem_12);  getitem_12 = None
        mul_14: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = None
        mul_15: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_14, primals_16)
        add_13: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_15, primals_17);  mul_15 = primals_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_13: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_13, [-1, 1280]);  add_13 = None
        addmm_4: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_18, view_13, primals_19);  primals_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_14: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_4, [4, 512, 3840]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_1 = torch.ops.aten.split.Tensor(view_14, 1280, 2);  view_14 = None
        getitem_13: "f32[4, 512, 1280]" = split_1[0]
        getitem_14: "f32[4, 512, 1280]" = split_1[1]
        getitem_15: "f32[4, 512, 1280]" = split_1[2];  split_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_15: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_14, [4, 512, -1, 64]);  getitem_14 = None
        permute_4: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_16: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_15, [4, 512, -1, 64]);  getitem_15 = None
        permute_5: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_17: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_13, [4, 512, -1, 64]);  getitem_13 = None
        permute_6: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_6, permute_4, permute_5, expand_2, True, 0.1, scale = 0.125)
        getitem_16: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_1[0]
        getitem_17: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention_1[1]
        getitem_18: "i64[]" = _scaled_dot_product_efficient_attention_1[2]
        getitem_19: "i64[]" = _scaled_dot_product_efficient_attention_1[3];  _scaled_dot_product_efficient_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_18: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_7, [4, 512, -1]);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_19: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_18, [-1, 1280]);  view_18 = None
        addmm_5: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_20, view_19, primals_21);  primals_20 = view_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_20: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_5, [4, 512, 1280]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_3: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_69: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        gt_3: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_69, 0.1);  inductor_random_default_69 = None
        mul_16: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_3, view_20);  view_20 = None
        mul_17: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_16, 1.1111111111111112);  mul_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_14: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_17, add_11);  mul_17 = add_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_3 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
        getitem_20: "f32[4, 512, 1]" = var_mean_3[0]
        getitem_21: "f32[4, 512, 1]" = var_mean_3[1];  var_mean_3 = None
        add_15: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_3: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        sub_5: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_14, getitem_21);  getitem_21 = None
        mul_18: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
        mul_19: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_18, primals_22)
        add_16: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_19, primals_23);  mul_19 = primals_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_21: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_16, [-1, 1280]);  add_16 = None
        addmm_6: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_24, view_21, primals_25);  primals_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_22: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_6, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_20: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_22, 0.5)
        pow_2: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_22, 3.0)
        mul_21: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_17: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_22, mul_21);  view_22 = mul_21 = None
        mul_22: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_17, 0.7978845608028654);  add_17 = None
        tanh_1: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
        add_18: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_23: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_20, add_18);  mul_20 = add_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_23: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_23, [-1, 5120]);  mul_23 = None
        addmm_7: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_26, view_23, primals_27);  primals_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_24: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_7, [4, 512, 1280]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_4: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_random_default_68: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        gt_4: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_68, 0.1);  inductor_random_default_68 = None
        mul_24: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_4, view_24);  view_24 = None
        mul_25: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_24, 1.1111111111111112);  mul_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_19: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_14, mul_25);  add_14 = mul_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
        getitem_22: "f32[4, 512, 1]" = var_mean_4[0]
        getitem_23: "f32[4, 512, 1]" = var_mean_4[1];  var_mean_4 = None
        add_20: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_4: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        sub_6: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_19, getitem_23);  getitem_23 = None
        mul_26: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
        mul_27: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_26, primals_28)
        add_21: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_27, primals_29);  mul_27 = primals_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_25: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_21, [-1, 1280]);  add_21 = None
        addmm_8: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_30, view_25, primals_31);  primals_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_26: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_8, [4, 512, 3840]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_2 = torch.ops.aten.split.Tensor(view_26, 1280, 2);  view_26 = None
        getitem_24: "f32[4, 512, 1280]" = split_2[0]
        getitem_25: "f32[4, 512, 1280]" = split_2[1]
        getitem_26: "f32[4, 512, 1280]" = split_2[2];  split_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_27: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_25, [4, 512, -1, 64]);  getitem_25 = None
        permute_8: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_28: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_26, [4, 512, -1, 64]);  getitem_26 = None
        permute_9: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_29: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_24, [4, 512, -1, 64]);  getitem_24 = None
        permute_10: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_10, permute_8, permute_9, expand_2, True, 0.1, scale = 0.125)
        getitem_27: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_2[0]
        getitem_28: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention_2[1]
        getitem_29: "i64[]" = _scaled_dot_product_efficient_attention_2[2]
        getitem_30: "i64[]" = _scaled_dot_product_efficient_attention_2[3];  _scaled_dot_product_efficient_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_11: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_30: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_11, [4, 512, -1]);  permute_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_31: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_30, [-1, 1280]);  view_30 = None
        addmm_9: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_32, view_31, primals_33);  primals_32 = view_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_32: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_9, [4, 512, 1280]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_5: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_random_default_67: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_5, 'rand');  inductor_lookup_seed_default_5 = None
        gt_5: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_67, 0.1);  inductor_random_default_67 = None
        mul_28: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_5, view_32);  view_32 = None
        mul_29: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_28, 1.1111111111111112);  mul_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_22: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_29, add_19);  mul_29 = add_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_5 = torch.ops.aten.var_mean.correction(add_22, [2], correction = 0, keepdim = True)
        getitem_31: "f32[4, 512, 1]" = var_mean_5[0]
        getitem_32: "f32[4, 512, 1]" = var_mean_5[1];  var_mean_5 = None
        add_23: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_31, 1e-05);  getitem_31 = None
        rsqrt_5: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        sub_7: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_22, getitem_32);  getitem_32 = None
        mul_30: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_5);  sub_7 = None
        mul_31: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_30, primals_34)
        add_24: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_31, primals_35);  mul_31 = primals_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_33: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_24, [-1, 1280]);  add_24 = None
        addmm_10: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_36, view_33, primals_37);  primals_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_34: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_10, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_32: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_34, 0.5)
        pow_3: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_34, 3.0)
        mul_33: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_25: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_34, mul_33);  view_34 = mul_33 = None
        mul_34: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_25, 0.7978845608028654);  add_25 = None
        tanh_2: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_34);  mul_34 = None
        add_26: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_35: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_32, add_26);  mul_32 = add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_35: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_35, [-1, 5120]);  mul_35 = None
        addmm_11: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_38, view_35, primals_39);  primals_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_36: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_11, [4, 512, 1280]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_6: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6)
        inductor_random_default_66: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_6, 'rand');  inductor_lookup_seed_default_6 = None
        gt_6: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_66, 0.1);  inductor_random_default_66 = None
        mul_36: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_6, view_36);  view_36 = None
        mul_37: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_36, 1.1111111111111112);  mul_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_27: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_22, mul_37);  add_22 = mul_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_6 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_33: "f32[4, 512, 1]" = var_mean_6[0]
        getitem_34: "f32[4, 512, 1]" = var_mean_6[1];  var_mean_6 = None
        add_28: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_33, 1e-05);  getitem_33 = None
        rsqrt_6: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_8: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_27, getitem_34);  getitem_34 = None
        mul_38: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_6);  sub_8 = None
        mul_39: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_38, primals_40)
        add_29: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_39, primals_41);  mul_39 = primals_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_37: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_29, [-1, 1280]);  add_29 = None
        addmm_12: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_42, view_37, primals_43);  primals_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_38: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_12, [4, 512, 3840]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_3 = torch.ops.aten.split.Tensor(view_38, 1280, 2);  view_38 = None
        getitem_35: "f32[4, 512, 1280]" = split_3[0]
        getitem_36: "f32[4, 512, 1280]" = split_3[1]
        getitem_37: "f32[4, 512, 1280]" = split_3[2];  split_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_39: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_36, [4, 512, -1, 64]);  getitem_36 = None
        permute_12: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_39, [0, 2, 1, 3]);  view_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_40: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_37, [4, 512, -1, 64]);  getitem_37 = None
        permute_13: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_41: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_35, [4, 512, -1, 64]);  getitem_35 = None
        permute_14: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_14, permute_12, permute_13, expand_2, True, 0.1, scale = 0.125)
        getitem_38: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_3[0]
        getitem_39: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention_3[1]
        getitem_40: "i64[]" = _scaled_dot_product_efficient_attention_3[2]
        getitem_41: "i64[]" = _scaled_dot_product_efficient_attention_3[3];  _scaled_dot_product_efficient_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_15: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_42: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_15, [4, 512, -1]);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_43: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_42, [-1, 1280]);  view_42 = None
        addmm_13: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_44, view_43, primals_45);  primals_44 = view_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_44: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_13, [4, 512, 1280]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_7: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 7)
        inductor_random_default_65: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_7, 'rand');  inductor_lookup_seed_default_7 = None
        gt_7: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_65, 0.1);  inductor_random_default_65 = None
        mul_40: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_7, view_44);  view_44 = None
        mul_41: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_40, 1.1111111111111112);  mul_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_30: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_41, add_27);  mul_41 = add_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_7 = torch.ops.aten.var_mean.correction(add_30, [2], correction = 0, keepdim = True)
        getitem_42: "f32[4, 512, 1]" = var_mean_7[0]
        getitem_43: "f32[4, 512, 1]" = var_mean_7[1];  var_mean_7 = None
        add_31: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_7: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        sub_9: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_30, getitem_43);  getitem_43 = None
        mul_42: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_7);  sub_9 = None
        mul_43: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_42, primals_46)
        add_32: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_43, primals_47);  mul_43 = primals_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_45: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_32, [-1, 1280]);  add_32 = None
        addmm_14: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_48, view_45, primals_49);  primals_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_46: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_14, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_44: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_46, 0.5)
        pow_4: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_46, 3.0)
        mul_45: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_33: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_46, mul_45);  view_46 = mul_45 = None
        mul_46: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_33, 0.7978845608028654);  add_33 = None
        tanh_3: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
        add_34: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_47: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_44, add_34);  mul_44 = add_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_47: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_47, [-1, 5120]);  mul_47 = None
        addmm_15: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_50, view_47, primals_51);  primals_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_48: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_15, [4, 512, 1280]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_8: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 8)
        inductor_random_default_64: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_8, 'rand');  inductor_lookup_seed_default_8 = None
        gt_8: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_64, 0.1);  inductor_random_default_64 = None
        mul_48: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_8, view_48);  view_48 = None
        mul_49: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_48, 1.1111111111111112);  mul_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_35: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_30, mul_49);  add_30 = mul_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_8 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_44: "f32[4, 512, 1]" = var_mean_8[0]
        getitem_45: "f32[4, 512, 1]" = var_mean_8[1];  var_mean_8 = None
        add_36: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_8: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_10: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_35, getitem_45);  getitem_45 = None
        mul_50: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_8);  sub_10 = None
        mul_51: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_50, primals_52)
        add_37: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_51, primals_53);  mul_51 = primals_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_49: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_37, [-1, 1280]);  add_37 = None
        addmm_16: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_54, view_49, primals_55);  primals_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_50: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_16, [4, 512, 3840]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_4 = torch.ops.aten.split.Tensor(view_50, 1280, 2);  view_50 = None
        getitem_46: "f32[4, 512, 1280]" = split_4[0]
        getitem_47: "f32[4, 512, 1280]" = split_4[1]
        getitem_48: "f32[4, 512, 1280]" = split_4[2];  split_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_51: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_47, [4, 512, -1, 64]);  getitem_47 = None
        permute_16: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_52: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_48, [4, 512, -1, 64]);  getitem_48 = None
        permute_17: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_53: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_46, [4, 512, -1, 64]);  getitem_46 = None
        permute_18: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_18, permute_16, permute_17, expand_2, True, 0.1, scale = 0.125)
        getitem_49: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_4[0]
        getitem_50: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention_4[1]
        getitem_51: "i64[]" = _scaled_dot_product_efficient_attention_4[2]
        getitem_52: "i64[]" = _scaled_dot_product_efficient_attention_4[3];  _scaled_dot_product_efficient_attention_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_19: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_54: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_19, [4, 512, -1]);  permute_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_55: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_54, [-1, 1280]);  view_54 = None
        addmm_17: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_56, view_55, primals_57);  primals_56 = view_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_56: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_17, [4, 512, 1280]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_9: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 9)
        inductor_random_default_63: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_9, 'rand');  inductor_lookup_seed_default_9 = None
        gt_9: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_63, 0.1);  inductor_random_default_63 = None
        mul_52: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_9, view_56);  view_56 = None
        mul_53: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_52, 1.1111111111111112);  mul_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_38: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_53, add_35);  mul_53 = add_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_9 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
        getitem_53: "f32[4, 512, 1]" = var_mean_9[0]
        getitem_54: "f32[4, 512, 1]" = var_mean_9[1];  var_mean_9 = None
        add_39: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_53, 1e-05);  getitem_53 = None
        rsqrt_9: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        sub_11: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_38, getitem_54);  getitem_54 = None
        mul_54: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_9);  sub_11 = None
        mul_55: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_54, primals_58)
        add_40: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_55, primals_59);  mul_55 = primals_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_57: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_40, [-1, 1280]);  add_40 = None
        addmm_18: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_60, view_57, primals_61);  primals_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_58: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_18, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_56: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_58, 0.5)
        pow_5: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_58, 3.0)
        mul_57: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_41: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_58, mul_57);  view_58 = mul_57 = None
        mul_58: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_41, 0.7978845608028654);  add_41 = None
        tanh_4: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_58);  mul_58 = None
        add_42: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_59: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_56, add_42);  mul_56 = add_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_59: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_59, [-1, 5120]);  mul_59 = None
        addmm_19: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_62, view_59, primals_63);  primals_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_60: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_19, [4, 512, 1280]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_10: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 10)
        inductor_random_default_62: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_10, 'rand');  inductor_lookup_seed_default_10 = None
        gt_10: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_62, 0.1);  inductor_random_default_62 = None
        mul_60: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_10, view_60);  view_60 = None
        mul_61: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_60, 1.1111111111111112);  mul_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_43: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_38, mul_61);  add_38 = mul_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_10 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_55: "f32[4, 512, 1]" = var_mean_10[0]
        getitem_56: "f32[4, 512, 1]" = var_mean_10[1];  var_mean_10 = None
        add_44: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_55, 1e-05);  getitem_55 = None
        rsqrt_10: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_12: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_43, getitem_56);  getitem_56 = None
        mul_62: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_10);  sub_12 = None
        mul_63: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_62, primals_64)
        add_45: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_63, primals_65);  mul_63 = primals_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_61: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_45, [-1, 1280]);  add_45 = None
        addmm_20: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_66, view_61, primals_67);  primals_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_62: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_20, [4, 512, 3840]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_5 = torch.ops.aten.split.Tensor(view_62, 1280, 2);  view_62 = None
        getitem_57: "f32[4, 512, 1280]" = split_5[0]
        getitem_58: "f32[4, 512, 1280]" = split_5[1]
        getitem_59: "f32[4, 512, 1280]" = split_5[2];  split_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_63: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_58, [4, 512, -1, 64]);  getitem_58 = None
        permute_20: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_63, [0, 2, 1, 3]);  view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_64: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_59, [4, 512, -1, 64]);  getitem_59 = None
        permute_21: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_65: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_57, [4, 512, -1, 64]);  getitem_57 = None
        permute_22: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_22, permute_20, permute_21, expand_2, True, 0.1, scale = 0.125)
        getitem_60: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_5[0]
        getitem_61: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention_5[1]
        getitem_62: "i64[]" = _scaled_dot_product_efficient_attention_5[2]
        getitem_63: "i64[]" = _scaled_dot_product_efficient_attention_5[3];  _scaled_dot_product_efficient_attention_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_23: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_66: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_23, [4, 512, -1]);  permute_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_67: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_66, [-1, 1280]);  view_66 = None
        addmm_21: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_68, view_67, primals_69);  primals_68 = view_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_68: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_21, [4, 512, 1280]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_11: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 11)
        inductor_random_default_61: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_11, 'rand');  inductor_lookup_seed_default_11 = None
        gt_11: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_61, 0.1);  inductor_random_default_61 = None
        mul_64: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_11, view_68);  view_68 = None
        mul_65: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_64, 1.1111111111111112);  mul_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_46: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_65, add_43);  mul_65 = add_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_11 = torch.ops.aten.var_mean.correction(add_46, [2], correction = 0, keepdim = True)
        getitem_64: "f32[4, 512, 1]" = var_mean_11[0]
        getitem_65: "f32[4, 512, 1]" = var_mean_11[1];  var_mean_11 = None
        add_47: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_11: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        sub_13: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_46, getitem_65);  getitem_65 = None
        mul_66: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_11);  sub_13 = None
        mul_67: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_66, primals_70)
        add_48: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_67, primals_71);  mul_67 = primals_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_69: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_48, [-1, 1280]);  add_48 = None
        addmm_22: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_72, view_69, primals_73);  primals_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_70: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_22, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_68: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_70, 0.5)
        pow_6: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_70, 3.0)
        mul_69: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_49: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_70, mul_69);  view_70 = mul_69 = None
        mul_70: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_49, 0.7978845608028654);  add_49 = None
        tanh_5: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
        add_50: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_71: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_68, add_50);  mul_68 = add_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_71: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_71, [-1, 5120]);  mul_71 = None
        addmm_23: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_74, view_71, primals_75);  primals_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_72: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_23, [4, 512, 1280]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_12: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 12)
        inductor_random_default_60: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_12, 'rand');  inductor_lookup_seed_default_12 = None
        gt_12: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_60, 0.1);  inductor_random_default_60 = None
        mul_72: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_12, view_72);  view_72 = None
        mul_73: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_72, 1.1111111111111112);  mul_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_51: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_46, mul_73);  add_46 = mul_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_12 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
        getitem_66: "f32[4, 512, 1]" = var_mean_12[0]
        getitem_67: "f32[4, 512, 1]" = var_mean_12[1];  var_mean_12 = None
        add_52: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_12: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        sub_14: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_51, getitem_67);  getitem_67 = None
        mul_74: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_12);  sub_14 = None
        mul_75: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_74, primals_76)
        add_53: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_75, primals_77);  mul_75 = primals_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_73: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_53, [-1, 1280]);  add_53 = None
        addmm_24: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_78, view_73, primals_79);  primals_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_74: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_24, [4, 512, 3840]);  addmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_6 = torch.ops.aten.split.Tensor(view_74, 1280, 2);  view_74 = None
        getitem_68: "f32[4, 512, 1280]" = split_6[0]
        getitem_69: "f32[4, 512, 1280]" = split_6[1]
        getitem_70: "f32[4, 512, 1280]" = split_6[2];  split_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_75: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_69, [4, 512, -1, 64]);  getitem_69 = None
        permute_24: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_76: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_70, [4, 512, -1, 64]);  getitem_70 = None
        permute_25: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_77: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_68, [4, 512, -1, 64]);  getitem_68 = None
        permute_26: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_77, [0, 2, 1, 3]);  view_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_26, permute_24, permute_25, expand_2, True, 0.1, scale = 0.125)
        getitem_71: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_6[0]
        getitem_72: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention_6[1]
        getitem_73: "i64[]" = _scaled_dot_product_efficient_attention_6[2]
        getitem_74: "i64[]" = _scaled_dot_product_efficient_attention_6[3];  _scaled_dot_product_efficient_attention_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_27: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_78: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_27, [4, 512, -1]);  permute_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_79: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_78, [-1, 1280]);  view_78 = None
        addmm_25: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_80, view_79, primals_81);  primals_80 = view_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_80: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_25, [4, 512, 1280]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_13: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 13)
        inductor_random_default_59: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_13, 'rand');  inductor_lookup_seed_default_13 = None
        gt_13: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_59, 0.1);  inductor_random_default_59 = None
        mul_76: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_13, view_80);  view_80 = None
        mul_77: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_76, 1.1111111111111112);  mul_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_54: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_77, add_51);  mul_77 = add_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_13 = torch.ops.aten.var_mean.correction(add_54, [2], correction = 0, keepdim = True)
        getitem_75: "f32[4, 512, 1]" = var_mean_13[0]
        getitem_76: "f32[4, 512, 1]" = var_mean_13[1];  var_mean_13 = None
        add_55: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_75, 1e-05);  getitem_75 = None
        rsqrt_13: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
        sub_15: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_54, getitem_76);  getitem_76 = None
        mul_78: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_13);  sub_15 = None
        mul_79: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_78, primals_82)
        add_56: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_79, primals_83);  mul_79 = primals_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_81: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_56, [-1, 1280]);  add_56 = None
        addmm_26: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_84, view_81, primals_85);  primals_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_82: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_26, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_80: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_82, 0.5)
        pow_7: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_82, 3.0)
        mul_81: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_57: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_82, mul_81);  view_82 = mul_81 = None
        mul_82: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_57, 0.7978845608028654);  add_57 = None
        tanh_6: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_82);  mul_82 = None
        add_58: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_83: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_80, add_58);  mul_80 = add_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_83: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_83, [-1, 5120]);  mul_83 = None
        addmm_27: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_86, view_83, primals_87);  primals_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_84: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_27, [4, 512, 1280]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_14: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 14)
        inductor_random_default_58: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_14, 'rand');  inductor_lookup_seed_default_14 = None
        gt_14: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_58, 0.1);  inductor_random_default_58 = None
        mul_84: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_14, view_84);  view_84 = None
        mul_85: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_84, 1.1111111111111112);  mul_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_59: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_54, mul_85);  add_54 = mul_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_14 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
        getitem_77: "f32[4, 512, 1]" = var_mean_14[0]
        getitem_78: "f32[4, 512, 1]" = var_mean_14[1];  var_mean_14 = None
        add_60: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_77, 1e-05);  getitem_77 = None
        rsqrt_14: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_16: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_59, getitem_78);  getitem_78 = None
        mul_86: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_14);  sub_16 = None
        mul_87: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_86, primals_88)
        add_61: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_87, primals_89);  mul_87 = primals_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_85: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_61, [-1, 1280]);  add_61 = None
        addmm_28: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_90, view_85, primals_91);  primals_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_86: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_28, [4, 512, 3840]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_7 = torch.ops.aten.split.Tensor(view_86, 1280, 2);  view_86 = None
        getitem_79: "f32[4, 512, 1280]" = split_7[0]
        getitem_80: "f32[4, 512, 1280]" = split_7[1]
        getitem_81: "f32[4, 512, 1280]" = split_7[2];  split_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_87: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_80, [4, 512, -1, 64]);  getitem_80 = None
        permute_28: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_88: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_81, [4, 512, -1, 64]);  getitem_81 = None
        permute_29: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_89: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_79, [4, 512, -1, 64]);  getitem_79 = None
        permute_30: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_30, permute_28, permute_29, expand_2, True, 0.1, scale = 0.125)
        getitem_82: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_7[0]
        getitem_83: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention_7[1]
        getitem_84: "i64[]" = _scaled_dot_product_efficient_attention_7[2]
        getitem_85: "i64[]" = _scaled_dot_product_efficient_attention_7[3];  _scaled_dot_product_efficient_attention_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_31: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_90: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_31, [4, 512, -1]);  permute_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_91: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_90, [-1, 1280]);  view_90 = None
        addmm_29: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_92, view_91, primals_93);  primals_92 = view_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_92: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_29, [4, 512, 1280]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_15: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 15)
        inductor_random_default_57: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_15, 'rand');  inductor_lookup_seed_default_15 = None
        gt_15: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_57, 0.1);  inductor_random_default_57 = None
        mul_88: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_15, view_92);  view_92 = None
        mul_89: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_88, 1.1111111111111112);  mul_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_62: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_89, add_59);  mul_89 = add_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_15 = torch.ops.aten.var_mean.correction(add_62, [2], correction = 0, keepdim = True)
        getitem_86: "f32[4, 512, 1]" = var_mean_15[0]
        getitem_87: "f32[4, 512, 1]" = var_mean_15[1];  var_mean_15 = None
        add_63: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
        rsqrt_15: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
        sub_17: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_62, getitem_87);  getitem_87 = None
        mul_90: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_15);  sub_17 = None
        mul_91: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_90, primals_94)
        add_64: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_91, primals_95);  mul_91 = primals_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_93: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_64, [-1, 1280]);  add_64 = None
        addmm_30: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_96, view_93, primals_97);  primals_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_94: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_30, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_92: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_94, 0.5)
        pow_8: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_94, 3.0)
        mul_93: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_65: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_94, mul_93);  view_94 = mul_93 = None
        mul_94: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_65, 0.7978845608028654);  add_65 = None
        tanh_7: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
        add_66: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_95: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_92, add_66);  mul_92 = add_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_95: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_95, [-1, 5120]);  mul_95 = None
        addmm_31: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_98, view_95, primals_99);  primals_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_96: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_31, [4, 512, 1280]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_16: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 16)
        inductor_random_default_56: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_16, 'rand');  inductor_lookup_seed_default_16 = None
        gt_16: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_56, 0.1);  inductor_random_default_56 = None
        mul_96: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_16, view_96);  view_96 = None
        mul_97: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_96, 1.1111111111111112);  mul_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_67: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_62, mul_97);  add_62 = mul_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_16 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
        getitem_88: "f32[4, 512, 1]" = var_mean_16[0]
        getitem_89: "f32[4, 512, 1]" = var_mean_16[1];  var_mean_16 = None
        add_68: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_16: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        sub_18: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_67, getitem_89);  getitem_89 = None
        mul_98: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_16);  sub_18 = None
        mul_99: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_98, primals_100)
        add_69: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_99, primals_101);  mul_99 = primals_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_97: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_69, [-1, 1280]);  add_69 = None
        addmm_32: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_102, view_97, primals_103);  primals_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_98: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_32, [4, 512, 3840]);  addmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_8 = torch.ops.aten.split.Tensor(view_98, 1280, 2);  view_98 = None
        getitem_90: "f32[4, 512, 1280]" = split_8[0]
        getitem_91: "f32[4, 512, 1280]" = split_8[1]
        getitem_92: "f32[4, 512, 1280]" = split_8[2];  split_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_99: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_91, [4, 512, -1, 64]);  getitem_91 = None
        permute_32: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_100: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_92, [4, 512, -1, 64]);  getitem_92 = None
        permute_33: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_101: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_90, [4, 512, -1, 64]);  getitem_90 = None
        permute_34: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_34, permute_32, permute_33, expand_2, True, 0.1, scale = 0.125)
        getitem_93: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_8[0]
        getitem_94: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention_8[1]
        getitem_95: "i64[]" = _scaled_dot_product_efficient_attention_8[2]
        getitem_96: "i64[]" = _scaled_dot_product_efficient_attention_8[3];  _scaled_dot_product_efficient_attention_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_35: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_102: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_35, [4, 512, -1]);  permute_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_103: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_102, [-1, 1280]);  view_102 = None
        addmm_33: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_104, view_103, primals_105);  primals_104 = view_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_104: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_33, [4, 512, 1280]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_17: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 17)
        inductor_random_default_55: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_17, 'rand');  inductor_lookup_seed_default_17 = None
        gt_17: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_55, 0.1);  inductor_random_default_55 = None
        mul_100: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_17, view_104);  view_104 = None
        mul_101: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_100, 1.1111111111111112);  mul_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_70: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_101, add_67);  mul_101 = add_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_17 = torch.ops.aten.var_mean.correction(add_70, [2], correction = 0, keepdim = True)
        getitem_97: "f32[4, 512, 1]" = var_mean_17[0]
        getitem_98: "f32[4, 512, 1]" = var_mean_17[1];  var_mean_17 = None
        add_71: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_97, 1e-05);  getitem_97 = None
        rsqrt_17: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        sub_19: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_70, getitem_98);  getitem_98 = None
        mul_102: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_17);  sub_19 = None
        mul_103: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_102, primals_106)
        add_72: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_103, primals_107);  mul_103 = primals_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_105: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_72, [-1, 1280]);  add_72 = None
        addmm_34: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_108, view_105, primals_109);  primals_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_106: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_34, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_104: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_106, 0.5)
        pow_9: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_106, 3.0)
        mul_105: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_73: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_106, mul_105);  view_106 = mul_105 = None
        mul_106: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_73, 0.7978845608028654);  add_73 = None
        tanh_8: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_106);  mul_106 = None
        add_74: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_107: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_104, add_74);  mul_104 = add_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_107: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_107, [-1, 5120]);  mul_107 = None
        addmm_35: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_110, view_107, primals_111);  primals_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_108: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_35, [4, 512, 1280]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_18: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 18)
        inductor_random_default_54: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_18, 'rand');  inductor_lookup_seed_default_18 = None
        gt_18: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_54, 0.1);  inductor_random_default_54 = None
        mul_108: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_18, view_108);  view_108 = None
        mul_109: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_108, 1.1111111111111112);  mul_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_75: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_70, mul_109);  add_70 = mul_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_18 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
        getitem_99: "f32[4, 512, 1]" = var_mean_18[0]
        getitem_100: "f32[4, 512, 1]" = var_mean_18[1];  var_mean_18 = None
        add_76: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_99, 1e-05);  getitem_99 = None
        rsqrt_18: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        sub_20: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_75, getitem_100);  getitem_100 = None
        mul_110: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_18);  sub_20 = None
        mul_111: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_110, primals_112)
        add_77: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_111, primals_113);  mul_111 = primals_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_109: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_77, [-1, 1280]);  add_77 = None
        addmm_36: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_114, view_109, primals_115);  primals_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_110: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_36, [4, 512, 3840]);  addmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_9 = torch.ops.aten.split.Tensor(view_110, 1280, 2);  view_110 = None
        getitem_101: "f32[4, 512, 1280]" = split_9[0]
        getitem_102: "f32[4, 512, 1280]" = split_9[1]
        getitem_103: "f32[4, 512, 1280]" = split_9[2];  split_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_111: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_102, [4, 512, -1, 64]);  getitem_102 = None
        permute_36: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_111, [0, 2, 1, 3]);  view_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_112: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_103, [4, 512, -1, 64]);  getitem_103 = None
        permute_37: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_113: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_101, [4, 512, -1, 64]);  getitem_101 = None
        permute_38: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_38, permute_36, permute_37, expand_2, True, 0.1, scale = 0.125)
        getitem_104: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_9[0]
        getitem_105: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention_9[1]
        getitem_106: "i64[]" = _scaled_dot_product_efficient_attention_9[2]
        getitem_107: "i64[]" = _scaled_dot_product_efficient_attention_9[3];  _scaled_dot_product_efficient_attention_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_39: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_114: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_39, [4, 512, -1]);  permute_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_115: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_114, [-1, 1280]);  view_114 = None
        addmm_37: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_116, view_115, primals_117);  primals_116 = view_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_116: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_37, [4, 512, 1280]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_19: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 19)
        inductor_random_default_53: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_19, 'rand');  inductor_lookup_seed_default_19 = None
        gt_19: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_53, 0.1);  inductor_random_default_53 = None
        mul_112: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_19, view_116);  view_116 = None
        mul_113: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_112, 1.1111111111111112);  mul_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_78: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_113, add_75);  mul_113 = add_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_19 = torch.ops.aten.var_mean.correction(add_78, [2], correction = 0, keepdim = True)
        getitem_108: "f32[4, 512, 1]" = var_mean_19[0]
        getitem_109: "f32[4, 512, 1]" = var_mean_19[1];  var_mean_19 = None
        add_79: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
        rsqrt_19: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
        sub_21: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_78, getitem_109);  getitem_109 = None
        mul_114: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_19);  sub_21 = None
        mul_115: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_114, primals_118)
        add_80: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_115, primals_119);  mul_115 = primals_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_117: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_80, [-1, 1280]);  add_80 = None
        addmm_38: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_120, view_117, primals_121);  primals_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_118: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_38, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_116: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_118, 0.5)
        pow_10: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_118, 3.0)
        mul_117: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_81: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_118, mul_117);  view_118 = mul_117 = None
        mul_118: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_81, 0.7978845608028654);  add_81 = None
        tanh_9: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_118);  mul_118 = None
        add_82: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_119: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_116, add_82);  mul_116 = add_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_119: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_119, [-1, 5120]);  mul_119 = None
        addmm_39: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_122, view_119, primals_123);  primals_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_120: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_39, [4, 512, 1280]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_20: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 20)
        inductor_random_default_52: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_20, 'rand');  inductor_lookup_seed_default_20 = None
        gt_20: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_52, 0.1);  inductor_random_default_52 = None
        mul_120: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_20, view_120);  view_120 = None
        mul_121: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_120, 1.1111111111111112);  mul_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_83: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_78, mul_121);  add_78 = mul_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_20 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
        getitem_110: "f32[4, 512, 1]" = var_mean_20[0]
        getitem_111: "f32[4, 512, 1]" = var_mean_20[1];  var_mean_20 = None
        add_84: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
        rsqrt_20: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        sub_22: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_83, getitem_111);  getitem_111 = None
        mul_122: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_20);  sub_22 = None
        mul_123: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_122, primals_124)
        add_85: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_123, primals_125);  mul_123 = primals_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_121: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_85, [-1, 1280]);  add_85 = None
        addmm_40: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_126, view_121, primals_127);  primals_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_122: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_40, [4, 512, 3840]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_10 = torch.ops.aten.split.Tensor(view_122, 1280, 2);  view_122 = None
        getitem_112: "f32[4, 512, 1280]" = split_10[0]
        getitem_113: "f32[4, 512, 1280]" = split_10[1]
        getitem_114: "f32[4, 512, 1280]" = split_10[2];  split_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_123: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_113, [4, 512, -1, 64]);  getitem_113 = None
        permute_40: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_124: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_114, [4, 512, -1, 64]);  getitem_114 = None
        permute_41: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_125: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_112, [4, 512, -1, 64]);  getitem_112 = None
        permute_42: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_125, [0, 2, 1, 3]);  view_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_42, permute_40, permute_41, expand_2, True, 0.1, scale = 0.125)
        getitem_115: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_10[0]
        getitem_116: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention_10[1]
        getitem_117: "i64[]" = _scaled_dot_product_efficient_attention_10[2]
        getitem_118: "i64[]" = _scaled_dot_product_efficient_attention_10[3];  _scaled_dot_product_efficient_attention_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_43: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_126: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_43, [4, 512, -1]);  permute_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_127: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_126, [-1, 1280]);  view_126 = None
        addmm_41: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_128, view_127, primals_129);  primals_128 = view_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_128: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_41, [4, 512, 1280]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_21: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 21)
        inductor_random_default_51: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_21, 'rand');  inductor_lookup_seed_default_21 = None
        gt_21: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_51, 0.1);  inductor_random_default_51 = None
        mul_124: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_21, view_128);  view_128 = None
        mul_125: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_124, 1.1111111111111112);  mul_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_86: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_125, add_83);  mul_125 = add_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_21 = torch.ops.aten.var_mean.correction(add_86, [2], correction = 0, keepdim = True)
        getitem_119: "f32[4, 512, 1]" = var_mean_21[0]
        getitem_120: "f32[4, 512, 1]" = var_mean_21[1];  var_mean_21 = None
        add_87: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_119, 1e-05);  getitem_119 = None
        rsqrt_21: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
        sub_23: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_86, getitem_120);  getitem_120 = None
        mul_126: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_21);  sub_23 = None
        mul_127: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_126, primals_130)
        add_88: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_127, primals_131);  mul_127 = primals_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_129: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_88, [-1, 1280]);  add_88 = None
        addmm_42: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_132, view_129, primals_133);  primals_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_130: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_42, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_128: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_130, 0.5)
        pow_11: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_130, 3.0)
        mul_129: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_89: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_130, mul_129);  view_130 = mul_129 = None
        mul_130: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_89, 0.7978845608028654);  add_89 = None
        tanh_10: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_130);  mul_130 = None
        add_90: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_131: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_128, add_90);  mul_128 = add_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_131: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_131, [-1, 5120]);  mul_131 = None
        addmm_43: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_134, view_131, primals_135);  primals_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_132: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_43, [4, 512, 1280]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_22: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 22)
        inductor_random_default_50: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_22, 'rand');  inductor_lookup_seed_default_22 = None
        gt_22: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_50, 0.1);  inductor_random_default_50 = None
        mul_132: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_22, view_132);  view_132 = None
        mul_133: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_132, 1.1111111111111112);  mul_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_91: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_86, mul_133);  add_86 = mul_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_22 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
        getitem_121: "f32[4, 512, 1]" = var_mean_22[0]
        getitem_122: "f32[4, 512, 1]" = var_mean_22[1];  var_mean_22 = None
        add_92: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_121, 1e-05);  getitem_121 = None
        rsqrt_22: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        sub_24: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_91, getitem_122);  getitem_122 = None
        mul_134: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_22);  sub_24 = None
        mul_135: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_134, primals_136)
        add_93: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_135, primals_137);  mul_135 = primals_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_133: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_93, [-1, 1280]);  add_93 = None
        addmm_44: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_138, view_133, primals_139);  primals_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_134: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_44, [4, 512, 3840]);  addmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_11 = torch.ops.aten.split.Tensor(view_134, 1280, 2);  view_134 = None
        getitem_123: "f32[4, 512, 1280]" = split_11[0]
        getitem_124: "f32[4, 512, 1280]" = split_11[1]
        getitem_125: "f32[4, 512, 1280]" = split_11[2];  split_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_135: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_124, [4, 512, -1, 64]);  getitem_124 = None
        permute_44: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_135, [0, 2, 1, 3]);  view_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_136: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_125, [4, 512, -1, 64]);  getitem_125 = None
        permute_45: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_137: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_123, [4, 512, -1, 64]);  getitem_123 = None
        permute_46: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_46, permute_44, permute_45, expand_2, True, 0.1, scale = 0.125)
        getitem_126: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_11[0]
        getitem_127: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention_11[1]
        getitem_128: "i64[]" = _scaled_dot_product_efficient_attention_11[2]
        getitem_129: "i64[]" = _scaled_dot_product_efficient_attention_11[3];  _scaled_dot_product_efficient_attention_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_47: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_138: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_47, [4, 512, -1]);  permute_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_139: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_138, [-1, 1280]);  view_138 = None
        addmm_45: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_140, view_139, primals_141);  primals_140 = view_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_140: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_45, [4, 512, 1280]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_23: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 23)
        inductor_random_default_49: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_23, 'rand');  inductor_lookup_seed_default_23 = None
        gt_23: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_49, 0.1);  inductor_random_default_49 = None
        mul_136: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_23, view_140);  view_140 = None
        mul_137: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_136, 1.1111111111111112);  mul_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_94: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_137, add_91);  mul_137 = add_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_23 = torch.ops.aten.var_mean.correction(add_94, [2], correction = 0, keepdim = True)
        getitem_130: "f32[4, 512, 1]" = var_mean_23[0]
        getitem_131: "f32[4, 512, 1]" = var_mean_23[1];  var_mean_23 = None
        add_95: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_23: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
        sub_25: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_94, getitem_131);  getitem_131 = None
        mul_138: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_23);  sub_25 = None
        mul_139: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_138, primals_142)
        add_96: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_139, primals_143);  mul_139 = primals_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_141: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_96, [-1, 1280]);  add_96 = None
        addmm_46: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_144, view_141, primals_145);  primals_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_142: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_46, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_140: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_142, 0.5)
        pow_12: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_142, 3.0)
        mul_141: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_97: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_142, mul_141);  view_142 = mul_141 = None
        mul_142: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_97, 0.7978845608028654);  add_97 = None
        tanh_11: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_142);  mul_142 = None
        add_98: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_143: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_140, add_98);  mul_140 = add_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_143: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_143, [-1, 5120]);  mul_143 = None
        addmm_47: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_146, view_143, primals_147);  primals_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_144: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_47, [4, 512, 1280]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_24: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 24)
        inductor_random_default_48: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_24, 'rand');  inductor_lookup_seed_default_24 = None
        gt_24: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_48, 0.1);  inductor_random_default_48 = None
        mul_144: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_24, view_144);  view_144 = None
        mul_145: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_144, 1.1111111111111112);  mul_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_99: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_94, mul_145);  add_94 = mul_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_24 = torch.ops.aten.var_mean.correction(add_99, [2], correction = 0, keepdim = True)
        getitem_132: "f32[4, 512, 1]" = var_mean_24[0]
        getitem_133: "f32[4, 512, 1]" = var_mean_24[1];  var_mean_24 = None
        add_100: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-05);  getitem_132 = None
        rsqrt_24: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        sub_26: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_99, getitem_133);  getitem_133 = None
        mul_146: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_24);  sub_26 = None
        mul_147: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_146, primals_148)
        add_101: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_147, primals_149);  mul_147 = primals_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_145: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_101, [-1, 1280]);  add_101 = None
        addmm_48: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_150, view_145, primals_151);  primals_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_146: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_48, [4, 512, 3840]);  addmm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_12 = torch.ops.aten.split.Tensor(view_146, 1280, 2);  view_146 = None
        getitem_134: "f32[4, 512, 1280]" = split_12[0]
        getitem_135: "f32[4, 512, 1280]" = split_12[1]
        getitem_136: "f32[4, 512, 1280]" = split_12[2];  split_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_147: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_135, [4, 512, -1, 64]);  getitem_135 = None
        permute_48: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_147, [0, 2, 1, 3]);  view_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_148: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_136, [4, 512, -1, 64]);  getitem_136 = None
        permute_49: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_149: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_134, [4, 512, -1, 64]);  getitem_134 = None
        permute_50: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_12 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_50, permute_48, permute_49, expand_2, True, 0.1, scale = 0.125)
        getitem_137: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_12[0]
        getitem_138: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention_12[1]
        getitem_139: "i64[]" = _scaled_dot_product_efficient_attention_12[2]
        getitem_140: "i64[]" = _scaled_dot_product_efficient_attention_12[3];  _scaled_dot_product_efficient_attention_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_51: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_137, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_150: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_51, [4, 512, -1]);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_151: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_150, [-1, 1280]);  view_150 = None
        addmm_49: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_152, view_151, primals_153);  primals_152 = view_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_152: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_49, [4, 512, 1280]);  addmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_25: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 25)
        inductor_random_default_47: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_25, 'rand');  inductor_lookup_seed_default_25 = None
        gt_25: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_47, 0.1);  inductor_random_default_47 = None
        mul_148: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_25, view_152);  view_152 = None
        mul_149: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_148, 1.1111111111111112);  mul_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_102: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_149, add_99);  mul_149 = add_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_25 = torch.ops.aten.var_mean.correction(add_102, [2], correction = 0, keepdim = True)
        getitem_141: "f32[4, 512, 1]" = var_mean_25[0]
        getitem_142: "f32[4, 512, 1]" = var_mean_25[1];  var_mean_25 = None
        add_103: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_141, 1e-05);  getitem_141 = None
        rsqrt_25: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
        sub_27: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_102, getitem_142);  getitem_142 = None
        mul_150: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_25);  sub_27 = None
        mul_151: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_150, primals_154)
        add_104: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_151, primals_155);  mul_151 = primals_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_153: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_104, [-1, 1280]);  add_104 = None
        addmm_50: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_156, view_153, primals_157);  primals_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_154: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_50, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_152: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_154, 0.5)
        pow_13: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_154, 3.0)
        mul_153: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
        add_105: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_154, mul_153);  view_154 = mul_153 = None
        mul_154: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_105, 0.7978845608028654);  add_105 = None
        tanh_12: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_154);  mul_154 = None
        add_106: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_12, 1.0);  tanh_12 = None
        mul_155: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_152, add_106);  mul_152 = add_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_155: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_155, [-1, 5120]);  mul_155 = None
        addmm_51: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_158, view_155, primals_159);  primals_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_156: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_51, [4, 512, 1280]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_26: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 26)
        inductor_random_default_46: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_26, 'rand');  inductor_lookup_seed_default_26 = None
        gt_26: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_46, 0.1);  inductor_random_default_46 = None
        mul_156: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_26, view_156);  view_156 = None
        mul_157: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_156, 1.1111111111111112);  mul_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_107: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_102, mul_157);  add_102 = mul_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_26 = torch.ops.aten.var_mean.correction(add_107, [2], correction = 0, keepdim = True)
        getitem_143: "f32[4, 512, 1]" = var_mean_26[0]
        getitem_144: "f32[4, 512, 1]" = var_mean_26[1];  var_mean_26 = None
        add_108: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_143, 1e-05);  getitem_143 = None
        rsqrt_26: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
        sub_28: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_107, getitem_144);  getitem_144 = None
        mul_158: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_26);  sub_28 = None
        mul_159: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_158, primals_160)
        add_109: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_159, primals_161);  mul_159 = primals_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_157: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_109, [-1, 1280]);  add_109 = None
        addmm_52: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_162, view_157, primals_163);  primals_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_158: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_52, [4, 512, 3840]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_13 = torch.ops.aten.split.Tensor(view_158, 1280, 2);  view_158 = None
        getitem_145: "f32[4, 512, 1280]" = split_13[0]
        getitem_146: "f32[4, 512, 1280]" = split_13[1]
        getitem_147: "f32[4, 512, 1280]" = split_13[2];  split_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_159: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_146, [4, 512, -1, 64]);  getitem_146 = None
        permute_52: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_159, [0, 2, 1, 3]);  view_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_160: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_147, [4, 512, -1, 64]);  getitem_147 = None
        permute_53: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_161: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_145, [4, 512, -1, 64]);  getitem_145 = None
        permute_54: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_161, [0, 2, 1, 3]);  view_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_13 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_54, permute_52, permute_53, expand_2, True, 0.1, scale = 0.125)
        getitem_148: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_13[0]
        getitem_149: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention_13[1]
        getitem_150: "i64[]" = _scaled_dot_product_efficient_attention_13[2]
        getitem_151: "i64[]" = _scaled_dot_product_efficient_attention_13[3];  _scaled_dot_product_efficient_attention_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_55: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_148, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_162: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_55, [4, 512, -1]);  permute_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_163: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_162, [-1, 1280]);  view_162 = None
        addmm_53: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_164, view_163, primals_165);  primals_164 = view_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_164: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_53, [4, 512, 1280]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_27: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 27)
        inductor_random_default_45: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_27, 'rand');  inductor_lookup_seed_default_27 = None
        gt_27: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_45, 0.1);  inductor_random_default_45 = None
        mul_160: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_27, view_164);  view_164 = None
        mul_161: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_160, 1.1111111111111112);  mul_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_110: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_161, add_107);  mul_161 = add_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_27 = torch.ops.aten.var_mean.correction(add_110, [2], correction = 0, keepdim = True)
        getitem_152: "f32[4, 512, 1]" = var_mean_27[0]
        getitem_153: "f32[4, 512, 1]" = var_mean_27[1];  var_mean_27 = None
        add_111: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_152, 1e-05);  getitem_152 = None
        rsqrt_27: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
        sub_29: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_110, getitem_153);  getitem_153 = None
        mul_162: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_27);  sub_29 = None
        mul_163: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_162, primals_166)
        add_112: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_163, primals_167);  mul_163 = primals_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_165: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_112, [-1, 1280]);  add_112 = None
        addmm_54: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_168, view_165, primals_169);  primals_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_166: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_54, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_164: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_166, 0.5)
        pow_14: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_166, 3.0)
        mul_165: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_14, 0.044715);  pow_14 = None
        add_113: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_166, mul_165);  view_166 = mul_165 = None
        mul_166: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_113, 0.7978845608028654);  add_113 = None
        tanh_13: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_166);  mul_166 = None
        add_114: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_13, 1.0);  tanh_13 = None
        mul_167: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_164, add_114);  mul_164 = add_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_167: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_167, [-1, 5120]);  mul_167 = None
        addmm_55: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_170, view_167, primals_171);  primals_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_168: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_55, [4, 512, 1280]);  addmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_28: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 28)
        inductor_random_default_44: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_28, 'rand');  inductor_lookup_seed_default_28 = None
        gt_28: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_44, 0.1);  inductor_random_default_44 = None
        mul_168: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_28, view_168);  view_168 = None
        mul_169: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_168, 1.1111111111111112);  mul_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_115: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_110, mul_169);  add_110 = mul_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_28 = torch.ops.aten.var_mean.correction(add_115, [2], correction = 0, keepdim = True)
        getitem_154: "f32[4, 512, 1]" = var_mean_28[0]
        getitem_155: "f32[4, 512, 1]" = var_mean_28[1];  var_mean_28 = None
        add_116: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_154, 1e-05);  getitem_154 = None
        rsqrt_28: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        sub_30: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_115, getitem_155);  getitem_155 = None
        mul_170: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_28);  sub_30 = None
        mul_171: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_170, primals_172)
        add_117: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_171, primals_173);  mul_171 = primals_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_169: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_117, [-1, 1280]);  add_117 = None
        addmm_56: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_174, view_169, primals_175);  primals_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_170: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_56, [4, 512, 3840]);  addmm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_14 = torch.ops.aten.split.Tensor(view_170, 1280, 2);  view_170 = None
        getitem_156: "f32[4, 512, 1280]" = split_14[0]
        getitem_157: "f32[4, 512, 1280]" = split_14[1]
        getitem_158: "f32[4, 512, 1280]" = split_14[2];  split_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_171: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_157, [4, 512, -1, 64]);  getitem_157 = None
        permute_56: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_171, [0, 2, 1, 3]);  view_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_172: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_158, [4, 512, -1, 64]);  getitem_158 = None
        permute_57: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_172, [0, 2, 1, 3]);  view_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_173: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_156, [4, 512, -1, 64]);  getitem_156 = None
        permute_58: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_173, [0, 2, 1, 3]);  view_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_14 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_58, permute_56, permute_57, expand_2, True, 0.1, scale = 0.125)
        getitem_159: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_14[0]
        getitem_160: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention_14[1]
        getitem_161: "i64[]" = _scaled_dot_product_efficient_attention_14[2]
        getitem_162: "i64[]" = _scaled_dot_product_efficient_attention_14[3];  _scaled_dot_product_efficient_attention_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_59: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_159, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_174: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_59, [4, 512, -1]);  permute_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_175: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_174, [-1, 1280]);  view_174 = None
        addmm_57: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_176, view_175, primals_177);  primals_176 = view_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_176: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_57, [4, 512, 1280]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_29: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 29)
        inductor_random_default_43: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_29, 'rand');  inductor_lookup_seed_default_29 = None
        gt_29: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_43, 0.1);  inductor_random_default_43 = None
        mul_172: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_29, view_176);  view_176 = None
        mul_173: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_172, 1.1111111111111112);  mul_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_118: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_173, add_115);  mul_173 = add_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_29 = torch.ops.aten.var_mean.correction(add_118, [2], correction = 0, keepdim = True)
        getitem_163: "f32[4, 512, 1]" = var_mean_29[0]
        getitem_164: "f32[4, 512, 1]" = var_mean_29[1];  var_mean_29 = None
        add_119: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_163, 1e-05);  getitem_163 = None
        rsqrt_29: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
        sub_31: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_118, getitem_164);  getitem_164 = None
        mul_174: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_29);  sub_31 = None
        mul_175: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_174, primals_178)
        add_120: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_175, primals_179);  mul_175 = primals_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_177: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_120, [-1, 1280]);  add_120 = None
        addmm_58: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_180, view_177, primals_181);  primals_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_178: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_58, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_176: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_178, 0.5)
        pow_15: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_178, 3.0)
        mul_177: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
        add_121: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_178, mul_177);  view_178 = mul_177 = None
        mul_178: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_121, 0.7978845608028654);  add_121 = None
        tanh_14: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_178);  mul_178 = None
        add_122: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_14, 1.0);  tanh_14 = None
        mul_179: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_176, add_122);  mul_176 = add_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_179: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_179, [-1, 5120]);  mul_179 = None
        addmm_59: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_182, view_179, primals_183);  primals_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_180: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_59, [4, 512, 1280]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_30: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 30)
        inductor_random_default_42: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_30, 'rand');  inductor_lookup_seed_default_30 = None
        gt_30: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_42, 0.1);  inductor_random_default_42 = None
        mul_180: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_30, view_180);  view_180 = None
        mul_181: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_180, 1.1111111111111112);  mul_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_123: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_118, mul_181);  add_118 = mul_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_30 = torch.ops.aten.var_mean.correction(add_123, [2], correction = 0, keepdim = True)
        getitem_165: "f32[4, 512, 1]" = var_mean_30[0]
        getitem_166: "f32[4, 512, 1]" = var_mean_30[1];  var_mean_30 = None
        add_124: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_165, 1e-05);  getitem_165 = None
        rsqrt_30: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
        sub_32: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_123, getitem_166);  getitem_166 = None
        mul_182: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_30);  sub_32 = None
        mul_183: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_182, primals_184)
        add_125: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_183, primals_185);  mul_183 = primals_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_181: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_125, [-1, 1280]);  add_125 = None
        addmm_60: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_186, view_181, primals_187);  primals_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_182: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_60, [4, 512, 3840]);  addmm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_15 = torch.ops.aten.split.Tensor(view_182, 1280, 2);  view_182 = None
        getitem_167: "f32[4, 512, 1280]" = split_15[0]
        getitem_168: "f32[4, 512, 1280]" = split_15[1]
        getitem_169: "f32[4, 512, 1280]" = split_15[2];  split_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_183: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_168, [4, 512, -1, 64]);  getitem_168 = None
        permute_60: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_184: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_169, [4, 512, -1, 64]);  getitem_169 = None
        permute_61: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_185: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_167, [4, 512, -1, 64]);  getitem_167 = None
        permute_62: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_15 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_62, permute_60, permute_61, expand_2, True, 0.1, scale = 0.125)
        getitem_170: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_15[0]
        getitem_171: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention_15[1]
        getitem_172: "i64[]" = _scaled_dot_product_efficient_attention_15[2]
        getitem_173: "i64[]" = _scaled_dot_product_efficient_attention_15[3];  _scaled_dot_product_efficient_attention_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_63: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_170, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_186: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_63, [4, 512, -1]);  permute_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_187: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_186, [-1, 1280]);  view_186 = None
        addmm_61: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_188, view_187, primals_189);  primals_188 = view_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_188: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_61, [4, 512, 1280]);  addmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_31: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 31)
        inductor_random_default_41: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_31, 'rand');  inductor_lookup_seed_default_31 = None
        gt_31: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_41, 0.1);  inductor_random_default_41 = None
        mul_184: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_31, view_188);  view_188 = None
        mul_185: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_184, 1.1111111111111112);  mul_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_126: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_185, add_123);  mul_185 = add_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_31 = torch.ops.aten.var_mean.correction(add_126, [2], correction = 0, keepdim = True)
        getitem_174: "f32[4, 512, 1]" = var_mean_31[0]
        getitem_175: "f32[4, 512, 1]" = var_mean_31[1];  var_mean_31 = None
        add_127: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_174, 1e-05);  getitem_174 = None
        rsqrt_31: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
        sub_33: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_126, getitem_175);  getitem_175 = None
        mul_186: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_31);  sub_33 = None
        mul_187: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_186, primals_190)
        add_128: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_187, primals_191);  mul_187 = primals_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_189: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_128, [-1, 1280]);  add_128 = None
        addmm_62: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_192, view_189, primals_193);  primals_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_190: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_62, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_188: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_190, 0.5)
        pow_16: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_190, 3.0)
        mul_189: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_16, 0.044715);  pow_16 = None
        add_129: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_190, mul_189);  view_190 = mul_189 = None
        mul_190: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_129, 0.7978845608028654);  add_129 = None
        tanh_15: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_190);  mul_190 = None
        add_130: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_15, 1.0);  tanh_15 = None
        mul_191: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_188, add_130);  mul_188 = add_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_191: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_191, [-1, 5120]);  mul_191 = None
        addmm_63: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_194, view_191, primals_195);  primals_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_192: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_63, [4, 512, 1280]);  addmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_32: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 32)
        inductor_random_default_40: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_32, 'rand');  inductor_lookup_seed_default_32 = None
        gt_32: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_40, 0.1);  inductor_random_default_40 = None
        mul_192: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_32, view_192);  view_192 = None
        mul_193: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_192, 1.1111111111111112);  mul_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_131: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_126, mul_193);  add_126 = mul_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_32 = torch.ops.aten.var_mean.correction(add_131, [2], correction = 0, keepdim = True)
        getitem_176: "f32[4, 512, 1]" = var_mean_32[0]
        getitem_177: "f32[4, 512, 1]" = var_mean_32[1];  var_mean_32 = None
        add_132: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-05);  getitem_176 = None
        rsqrt_32: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
        sub_34: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_131, getitem_177);  getitem_177 = None
        mul_194: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_32);  sub_34 = None
        mul_195: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_194, primals_196)
        add_133: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_195, primals_197);  mul_195 = primals_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_193: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_133, [-1, 1280]);  add_133 = None
        addmm_64: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_198, view_193, primals_199);  primals_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_194: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_64, [4, 512, 3840]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_16 = torch.ops.aten.split.Tensor(view_194, 1280, 2);  view_194 = None
        getitem_178: "f32[4, 512, 1280]" = split_16[0]
        getitem_179: "f32[4, 512, 1280]" = split_16[1]
        getitem_180: "f32[4, 512, 1280]" = split_16[2];  split_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_195: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_179, [4, 512, -1, 64]);  getitem_179 = None
        permute_64: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_195, [0, 2, 1, 3]);  view_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_196: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_180, [4, 512, -1, 64]);  getitem_180 = None
        permute_65: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_196, [0, 2, 1, 3]);  view_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_197: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_178, [4, 512, -1, 64]);  getitem_178 = None
        permute_66: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_197, [0, 2, 1, 3]);  view_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_16 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_66, permute_64, permute_65, expand_2, True, 0.1, scale = 0.125)
        getitem_181: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_16[0]
        getitem_182: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention_16[1]
        getitem_183: "i64[]" = _scaled_dot_product_efficient_attention_16[2]
        getitem_184: "i64[]" = _scaled_dot_product_efficient_attention_16[3];  _scaled_dot_product_efficient_attention_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_67: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_181, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_198: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_67, [4, 512, -1]);  permute_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_199: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_198, [-1, 1280]);  view_198 = None
        addmm_65: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_200, view_199, primals_201);  primals_200 = view_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_200: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_65, [4, 512, 1280]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_33: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 33)
        inductor_random_default_39: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_33, 'rand');  inductor_lookup_seed_default_33 = None
        gt_33: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_39, 0.1);  inductor_random_default_39 = None
        mul_196: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_33, view_200);  view_200 = None
        mul_197: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_196, 1.1111111111111112);  mul_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_134: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_197, add_131);  mul_197 = add_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_33 = torch.ops.aten.var_mean.correction(add_134, [2], correction = 0, keepdim = True)
        getitem_185: "f32[4, 512, 1]" = var_mean_33[0]
        getitem_186: "f32[4, 512, 1]" = var_mean_33[1];  var_mean_33 = None
        add_135: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_185, 1e-05);  getitem_185 = None
        rsqrt_33: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
        sub_35: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_134, getitem_186);  getitem_186 = None
        mul_198: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_33);  sub_35 = None
        mul_199: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_198, primals_202)
        add_136: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_199, primals_203);  mul_199 = primals_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_201: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_136, [-1, 1280]);  add_136 = None
        addmm_66: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_204, view_201, primals_205);  primals_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_202: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_66, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_200: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_202, 0.5)
        pow_17: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_202, 3.0)
        mul_201: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_17, 0.044715);  pow_17 = None
        add_137: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_202, mul_201);  view_202 = mul_201 = None
        mul_202: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_137, 0.7978845608028654);  add_137 = None
        tanh_16: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_202);  mul_202 = None
        add_138: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_16, 1.0);  tanh_16 = None
        mul_203: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_200, add_138);  mul_200 = add_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_203: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_203, [-1, 5120]);  mul_203 = None
        addmm_67: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_206, view_203, primals_207);  primals_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_204: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_67, [4, 512, 1280]);  addmm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_34: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 34)
        inductor_random_default_38: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_34, 'rand');  inductor_lookup_seed_default_34 = None
        gt_34: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_38, 0.1);  inductor_random_default_38 = None
        mul_204: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_34, view_204);  view_204 = None
        mul_205: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_204, 1.1111111111111112);  mul_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_139: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_134, mul_205);  add_134 = mul_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_34 = torch.ops.aten.var_mean.correction(add_139, [2], correction = 0, keepdim = True)
        getitem_187: "f32[4, 512, 1]" = var_mean_34[0]
        getitem_188: "f32[4, 512, 1]" = var_mean_34[1];  var_mean_34 = None
        add_140: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_187, 1e-05);  getitem_187 = None
        rsqrt_34: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
        sub_36: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_139, getitem_188);  getitem_188 = None
        mul_206: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_34);  sub_36 = None
        mul_207: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_206, primals_208)
        add_141: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_207, primals_209);  mul_207 = primals_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_205: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_141, [-1, 1280]);  add_141 = None
        addmm_68: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_210, view_205, primals_211);  primals_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_206: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_68, [4, 512, 3840]);  addmm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_17 = torch.ops.aten.split.Tensor(view_206, 1280, 2);  view_206 = None
        getitem_189: "f32[4, 512, 1280]" = split_17[0]
        getitem_190: "f32[4, 512, 1280]" = split_17[1]
        getitem_191: "f32[4, 512, 1280]" = split_17[2];  split_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_207: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_190, [4, 512, -1, 64]);  getitem_190 = None
        permute_68: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_208: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_191, [4, 512, -1, 64]);  getitem_191 = None
        permute_69: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_209: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_189, [4, 512, -1, 64]);  getitem_189 = None
        permute_70: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_209, [0, 2, 1, 3]);  view_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_17 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_70, permute_68, permute_69, expand_2, True, 0.1, scale = 0.125)
        getitem_192: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_17[0]
        getitem_193: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention_17[1]
        getitem_194: "i64[]" = _scaled_dot_product_efficient_attention_17[2]
        getitem_195: "i64[]" = _scaled_dot_product_efficient_attention_17[3];  _scaled_dot_product_efficient_attention_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_71: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_192, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_210: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_71, [4, 512, -1]);  permute_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_211: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_210, [-1, 1280]);  view_210 = None
        addmm_69: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_212, view_211, primals_213);  primals_212 = view_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_212: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_69, [4, 512, 1280]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_35: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 35)
        inductor_random_default_37: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_35, 'rand');  inductor_lookup_seed_default_35 = None
        gt_35: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_37, 0.1);  inductor_random_default_37 = None
        mul_208: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_35, view_212);  view_212 = None
        mul_209: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_208, 1.1111111111111112);  mul_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_142: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_209, add_139);  mul_209 = add_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_35 = torch.ops.aten.var_mean.correction(add_142, [2], correction = 0, keepdim = True)
        getitem_196: "f32[4, 512, 1]" = var_mean_35[0]
        getitem_197: "f32[4, 512, 1]" = var_mean_35[1];  var_mean_35 = None
        add_143: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_196, 1e-05);  getitem_196 = None
        rsqrt_35: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
        sub_37: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_142, getitem_197);  getitem_197 = None
        mul_210: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_35);  sub_37 = None
        mul_211: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_210, primals_214)
        add_144: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_211, primals_215);  mul_211 = primals_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_213: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_144, [-1, 1280]);  add_144 = None
        addmm_70: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_216, view_213, primals_217);  primals_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_214: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_70, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_212: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_214, 0.5)
        pow_18: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_214, 3.0)
        mul_213: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_18, 0.044715);  pow_18 = None
        add_145: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_214, mul_213);  view_214 = mul_213 = None
        mul_214: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_145, 0.7978845608028654);  add_145 = None
        tanh_17: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_214);  mul_214 = None
        add_146: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_17, 1.0);  tanh_17 = None
        mul_215: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_212, add_146);  mul_212 = add_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_215: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_215, [-1, 5120]);  mul_215 = None
        addmm_71: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_218, view_215, primals_219);  primals_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_216: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_71, [4, 512, 1280]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_36: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 36)
        inductor_random_default_36: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_36, 'rand');  inductor_lookup_seed_default_36 = None
        gt_36: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_36, 0.1);  inductor_random_default_36 = None
        mul_216: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_36, view_216);  view_216 = None
        mul_217: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_216, 1.1111111111111112);  mul_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_147: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_142, mul_217);  add_142 = mul_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_36 = torch.ops.aten.var_mean.correction(add_147, [2], correction = 0, keepdim = True)
        getitem_198: "f32[4, 512, 1]" = var_mean_36[0]
        getitem_199: "f32[4, 512, 1]" = var_mean_36[1];  var_mean_36 = None
        add_148: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_198, 1e-05);  getitem_198 = None
        rsqrt_36: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
        sub_38: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_147, getitem_199);  getitem_199 = None
        mul_218: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_36);  sub_38 = None
        mul_219: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_218, primals_220)
        add_149: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_219, primals_221);  mul_219 = primals_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_217: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_149, [-1, 1280]);  add_149 = None
        addmm_72: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_222, view_217, primals_223);  primals_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_218: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_72, [4, 512, 3840]);  addmm_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_18 = torch.ops.aten.split.Tensor(view_218, 1280, 2);  view_218 = None
        getitem_200: "f32[4, 512, 1280]" = split_18[0]
        getitem_201: "f32[4, 512, 1280]" = split_18[1]
        getitem_202: "f32[4, 512, 1280]" = split_18[2];  split_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_219: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_201, [4, 512, -1, 64]);  getitem_201 = None
        permute_72: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_219, [0, 2, 1, 3]);  view_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_220: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_202, [4, 512, -1, 64]);  getitem_202 = None
        permute_73: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_220, [0, 2, 1, 3]);  view_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_221: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_200, [4, 512, -1, 64]);  getitem_200 = None
        permute_74: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_221, [0, 2, 1, 3]);  view_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_18 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_74, permute_72, permute_73, expand_2, True, 0.1, scale = 0.125)
        getitem_203: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_18[0]
        getitem_204: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention_18[1]
        getitem_205: "i64[]" = _scaled_dot_product_efficient_attention_18[2]
        getitem_206: "i64[]" = _scaled_dot_product_efficient_attention_18[3];  _scaled_dot_product_efficient_attention_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_75: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_203, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_222: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_75, [4, 512, -1]);  permute_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_223: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_222, [-1, 1280]);  view_222 = None
        addmm_73: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_224, view_223, primals_225);  primals_224 = view_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_224: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_73, [4, 512, 1280]);  addmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_37: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 37)
        inductor_random_default_35: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_37, 'rand');  inductor_lookup_seed_default_37 = None
        gt_37: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_35, 0.1);  inductor_random_default_35 = None
        mul_220: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_37, view_224);  view_224 = None
        mul_221: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_220, 1.1111111111111112);  mul_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_150: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_221, add_147);  mul_221 = add_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_37 = torch.ops.aten.var_mean.correction(add_150, [2], correction = 0, keepdim = True)
        getitem_207: "f32[4, 512, 1]" = var_mean_37[0]
        getitem_208: "f32[4, 512, 1]" = var_mean_37[1];  var_mean_37 = None
        add_151: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_207, 1e-05);  getitem_207 = None
        rsqrt_37: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
        sub_39: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_150, getitem_208);  getitem_208 = None
        mul_222: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_37);  sub_39 = None
        mul_223: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_222, primals_226)
        add_152: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_223, primals_227);  mul_223 = primals_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_225: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_152, [-1, 1280]);  add_152 = None
        addmm_74: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_228, view_225, primals_229);  primals_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_226: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_74, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_224: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_226, 0.5)
        pow_19: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_226, 3.0)
        mul_225: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_19, 0.044715);  pow_19 = None
        add_153: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_226, mul_225);  view_226 = mul_225 = None
        mul_226: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_153, 0.7978845608028654);  add_153 = None
        tanh_18: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_226);  mul_226 = None
        add_154: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_18, 1.0);  tanh_18 = None
        mul_227: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_224, add_154);  mul_224 = add_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_227: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_227, [-1, 5120]);  mul_227 = None
        addmm_75: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_230, view_227, primals_231);  primals_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_228: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_75, [4, 512, 1280]);  addmm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_38: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 38)
        inductor_random_default_34: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_38, 'rand');  inductor_lookup_seed_default_38 = None
        gt_38: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_34, 0.1);  inductor_random_default_34 = None
        mul_228: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_38, view_228);  view_228 = None
        mul_229: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_228, 1.1111111111111112);  mul_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_155: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_150, mul_229);  add_150 = mul_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_38 = torch.ops.aten.var_mean.correction(add_155, [2], correction = 0, keepdim = True)
        getitem_209: "f32[4, 512, 1]" = var_mean_38[0]
        getitem_210: "f32[4, 512, 1]" = var_mean_38[1];  var_mean_38 = None
        add_156: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_209, 1e-05);  getitem_209 = None
        rsqrt_38: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
        sub_40: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_155, getitem_210);  getitem_210 = None
        mul_230: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_38);  sub_40 = None
        mul_231: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_230, primals_232)
        add_157: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_231, primals_233);  mul_231 = primals_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_229: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_157, [-1, 1280]);  add_157 = None
        addmm_76: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_234, view_229, primals_235);  primals_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_230: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_76, [4, 512, 3840]);  addmm_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_19 = torch.ops.aten.split.Tensor(view_230, 1280, 2);  view_230 = None
        getitem_211: "f32[4, 512, 1280]" = split_19[0]
        getitem_212: "f32[4, 512, 1280]" = split_19[1]
        getitem_213: "f32[4, 512, 1280]" = split_19[2];  split_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_231: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_212, [4, 512, -1, 64]);  getitem_212 = None
        permute_76: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_231, [0, 2, 1, 3]);  view_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_232: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_213, [4, 512, -1, 64]);  getitem_213 = None
        permute_77: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_232, [0, 2, 1, 3]);  view_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_233: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_211, [4, 512, -1, 64]);  getitem_211 = None
        permute_78: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_233, [0, 2, 1, 3]);  view_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_19 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_78, permute_76, permute_77, expand_2, True, 0.1, scale = 0.125)
        getitem_214: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_19[0]
        getitem_215: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention_19[1]
        getitem_216: "i64[]" = _scaled_dot_product_efficient_attention_19[2]
        getitem_217: "i64[]" = _scaled_dot_product_efficient_attention_19[3];  _scaled_dot_product_efficient_attention_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_79: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_214, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_234: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_79, [4, 512, -1]);  permute_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_235: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_234, [-1, 1280]);  view_234 = None
        addmm_77: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_236, view_235, primals_237);  primals_236 = view_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_236: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_77, [4, 512, 1280]);  addmm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_39: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 39)
        inductor_random_default_33: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_39, 'rand');  inductor_lookup_seed_default_39 = None
        gt_39: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_33, 0.1);  inductor_random_default_33 = None
        mul_232: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_39, view_236);  view_236 = None
        mul_233: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_232, 1.1111111111111112);  mul_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_158: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_233, add_155);  mul_233 = add_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_39 = torch.ops.aten.var_mean.correction(add_158, [2], correction = 0, keepdim = True)
        getitem_218: "f32[4, 512, 1]" = var_mean_39[0]
        getitem_219: "f32[4, 512, 1]" = var_mean_39[1];  var_mean_39 = None
        add_159: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_218, 1e-05);  getitem_218 = None
        rsqrt_39: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
        sub_41: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_158, getitem_219);  getitem_219 = None
        mul_234: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_39);  sub_41 = None
        mul_235: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_234, primals_238)
        add_160: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_235, primals_239);  mul_235 = primals_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_237: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_160, [-1, 1280]);  add_160 = None
        addmm_78: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_240, view_237, primals_241);  primals_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_238: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_78, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_236: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_238, 0.5)
        pow_20: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_238, 3.0)
        mul_237: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_20, 0.044715);  pow_20 = None
        add_161: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_238, mul_237);  view_238 = mul_237 = None
        mul_238: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_161, 0.7978845608028654);  add_161 = None
        tanh_19: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_238);  mul_238 = None
        add_162: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_19, 1.0);  tanh_19 = None
        mul_239: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_236, add_162);  mul_236 = add_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_239: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_239, [-1, 5120]);  mul_239 = None
        addmm_79: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_242, view_239, primals_243);  primals_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_240: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_79, [4, 512, 1280]);  addmm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_40: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 40)
        inductor_random_default_32: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_40, 'rand');  inductor_lookup_seed_default_40 = None
        gt_40: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_32, 0.1);  inductor_random_default_32 = None
        mul_240: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_40, view_240);  view_240 = None
        mul_241: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_240, 1.1111111111111112);  mul_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_163: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_158, mul_241);  add_158 = mul_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_40 = torch.ops.aten.var_mean.correction(add_163, [2], correction = 0, keepdim = True)
        getitem_220: "f32[4, 512, 1]" = var_mean_40[0]
        getitem_221: "f32[4, 512, 1]" = var_mean_40[1];  var_mean_40 = None
        add_164: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_220, 1e-05);  getitem_220 = None
        rsqrt_40: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
        sub_42: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_163, getitem_221);  getitem_221 = None
        mul_242: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_40);  sub_42 = None
        mul_243: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_242, primals_244)
        add_165: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_243, primals_245);  mul_243 = primals_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_241: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_165, [-1, 1280]);  add_165 = None
        addmm_80: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_246, view_241, primals_247);  primals_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_242: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_80, [4, 512, 3840]);  addmm_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_20 = torch.ops.aten.split.Tensor(view_242, 1280, 2);  view_242 = None
        getitem_222: "f32[4, 512, 1280]" = split_20[0]
        getitem_223: "f32[4, 512, 1280]" = split_20[1]
        getitem_224: "f32[4, 512, 1280]" = split_20[2];  split_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_243: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_223, [4, 512, -1, 64]);  getitem_223 = None
        permute_80: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_243, [0, 2, 1, 3]);  view_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_244: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_224, [4, 512, -1, 64]);  getitem_224 = None
        permute_81: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_245: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_222, [4, 512, -1, 64]);  getitem_222 = None
        permute_82: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_245, [0, 2, 1, 3]);  view_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_20 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_82, permute_80, permute_81, expand_2, True, 0.1, scale = 0.125)
        getitem_225: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_20[0]
        getitem_226: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention_20[1]
        getitem_227: "i64[]" = _scaled_dot_product_efficient_attention_20[2]
        getitem_228: "i64[]" = _scaled_dot_product_efficient_attention_20[3];  _scaled_dot_product_efficient_attention_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_83: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_225, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_246: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_83, [4, 512, -1]);  permute_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_247: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_246, [-1, 1280]);  view_246 = None
        addmm_81: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_248, view_247, primals_249);  primals_248 = view_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_248: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_81, [4, 512, 1280]);  addmm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_41: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 41)
        inductor_random_default_31: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_41, 'rand');  inductor_lookup_seed_default_41 = None
        gt_41: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_31, 0.1);  inductor_random_default_31 = None
        mul_244: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_41, view_248);  view_248 = None
        mul_245: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_244, 1.1111111111111112);  mul_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_166: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_245, add_163);  mul_245 = add_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_41 = torch.ops.aten.var_mean.correction(add_166, [2], correction = 0, keepdim = True)
        getitem_229: "f32[4, 512, 1]" = var_mean_41[0]
        getitem_230: "f32[4, 512, 1]" = var_mean_41[1];  var_mean_41 = None
        add_167: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_229, 1e-05);  getitem_229 = None
        rsqrt_41: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_167);  add_167 = None
        sub_43: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_166, getitem_230);  getitem_230 = None
        mul_246: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_41);  sub_43 = None
        mul_247: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_246, primals_250)
        add_168: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_247, primals_251);  mul_247 = primals_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_249: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_168, [-1, 1280]);  add_168 = None
        addmm_82: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_252, view_249, primals_253);  primals_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_250: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_82, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_248: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_250, 0.5)
        pow_21: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_250, 3.0)
        mul_249: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
        add_169: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_250, mul_249);  view_250 = mul_249 = None
        mul_250: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_169, 0.7978845608028654);  add_169 = None
        tanh_20: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_250);  mul_250 = None
        add_170: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_20, 1.0);  tanh_20 = None
        mul_251: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_248, add_170);  mul_248 = add_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_251: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_251, [-1, 5120]);  mul_251 = None
        addmm_83: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_254, view_251, primals_255);  primals_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_252: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_83, [4, 512, 1280]);  addmm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_42: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 42)
        inductor_random_default_30: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_42, 'rand');  inductor_lookup_seed_default_42 = None
        gt_42: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_30, 0.1);  inductor_random_default_30 = None
        mul_252: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_42, view_252);  view_252 = None
        mul_253: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_252, 1.1111111111111112);  mul_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_171: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_166, mul_253);  add_166 = mul_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_42 = torch.ops.aten.var_mean.correction(add_171, [2], correction = 0, keepdim = True)
        getitem_231: "f32[4, 512, 1]" = var_mean_42[0]
        getitem_232: "f32[4, 512, 1]" = var_mean_42[1];  var_mean_42 = None
        add_172: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_231, 1e-05);  getitem_231 = None
        rsqrt_42: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
        sub_44: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_171, getitem_232);  getitem_232 = None
        mul_254: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_42);  sub_44 = None
        mul_255: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_254, primals_256)
        add_173: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_255, primals_257);  mul_255 = primals_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_253: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_173, [-1, 1280]);  add_173 = None
        addmm_84: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_258, view_253, primals_259);  primals_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_254: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_84, [4, 512, 3840]);  addmm_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_21 = torch.ops.aten.split.Tensor(view_254, 1280, 2);  view_254 = None
        getitem_233: "f32[4, 512, 1280]" = split_21[0]
        getitem_234: "f32[4, 512, 1280]" = split_21[1]
        getitem_235: "f32[4, 512, 1280]" = split_21[2];  split_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_255: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_234, [4, 512, -1, 64]);  getitem_234 = None
        permute_84: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_255, [0, 2, 1, 3]);  view_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_256: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_235, [4, 512, -1, 64]);  getitem_235 = None
        permute_85: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_257: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_233, [4, 512, -1, 64]);  getitem_233 = None
        permute_86: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_257, [0, 2, 1, 3]);  view_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_21 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_86, permute_84, permute_85, expand_2, True, 0.1, scale = 0.125)
        getitem_236: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_21[0]
        getitem_237: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention_21[1]
        getitem_238: "i64[]" = _scaled_dot_product_efficient_attention_21[2]
        getitem_239: "i64[]" = _scaled_dot_product_efficient_attention_21[3];  _scaled_dot_product_efficient_attention_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_87: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_236, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_258: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_87, [4, 512, -1]);  permute_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_259: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_258, [-1, 1280]);  view_258 = None
        addmm_85: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_260, view_259, primals_261);  primals_260 = view_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_260: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_85, [4, 512, 1280]);  addmm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_43: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 43)
        inductor_random_default_29: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_43, 'rand');  inductor_lookup_seed_default_43 = None
        gt_43: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_29, 0.1);  inductor_random_default_29 = None
        mul_256: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_43, view_260);  view_260 = None
        mul_257: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_256, 1.1111111111111112);  mul_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_174: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_257, add_171);  mul_257 = add_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_43 = torch.ops.aten.var_mean.correction(add_174, [2], correction = 0, keepdim = True)
        getitem_240: "f32[4, 512, 1]" = var_mean_43[0]
        getitem_241: "f32[4, 512, 1]" = var_mean_43[1];  var_mean_43 = None
        add_175: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_240, 1e-05);  getitem_240 = None
        rsqrt_43: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
        sub_45: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_174, getitem_241);  getitem_241 = None
        mul_258: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_43);  sub_45 = None
        mul_259: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_258, primals_262)
        add_176: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_259, primals_263);  mul_259 = primals_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_261: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_176, [-1, 1280]);  add_176 = None
        addmm_86: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_264, view_261, primals_265);  primals_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_262: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_86, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_260: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_262, 0.5)
        pow_22: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_262, 3.0)
        mul_261: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_22, 0.044715);  pow_22 = None
        add_177: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_262, mul_261);  view_262 = mul_261 = None
        mul_262: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_177, 0.7978845608028654);  add_177 = None
        tanh_21: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_262);  mul_262 = None
        add_178: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_21, 1.0);  tanh_21 = None
        mul_263: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_260, add_178);  mul_260 = add_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_263: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_263, [-1, 5120]);  mul_263 = None
        addmm_87: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_266, view_263, primals_267);  primals_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_264: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_87, [4, 512, 1280]);  addmm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_44: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 44)
        inductor_random_default_28: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_44, 'rand');  inductor_lookup_seed_default_44 = None
        gt_44: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_28, 0.1);  inductor_random_default_28 = None
        mul_264: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_44, view_264);  view_264 = None
        mul_265: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_264, 1.1111111111111112);  mul_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_179: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_174, mul_265);  add_174 = mul_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_44 = torch.ops.aten.var_mean.correction(add_179, [2], correction = 0, keepdim = True)
        getitem_242: "f32[4, 512, 1]" = var_mean_44[0]
        getitem_243: "f32[4, 512, 1]" = var_mean_44[1];  var_mean_44 = None
        add_180: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_242, 1e-05);  getitem_242 = None
        rsqrt_44: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_180);  add_180 = None
        sub_46: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_179, getitem_243);  getitem_243 = None
        mul_266: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_44);  sub_46 = None
        mul_267: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_266, primals_268)
        add_181: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_267, primals_269);  mul_267 = primals_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_265: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_181, [-1, 1280]);  add_181 = None
        addmm_88: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_270, view_265, primals_271);  primals_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_266: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_88, [4, 512, 3840]);  addmm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_22 = torch.ops.aten.split.Tensor(view_266, 1280, 2);  view_266 = None
        getitem_244: "f32[4, 512, 1280]" = split_22[0]
        getitem_245: "f32[4, 512, 1280]" = split_22[1]
        getitem_246: "f32[4, 512, 1280]" = split_22[2];  split_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_267: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_245, [4, 512, -1, 64]);  getitem_245 = None
        permute_88: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_267, [0, 2, 1, 3]);  view_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_268: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_246, [4, 512, -1, 64]);  getitem_246 = None
        permute_89: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_268, [0, 2, 1, 3]);  view_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_269: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_244, [4, 512, -1, 64]);  getitem_244 = None
        permute_90: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_269, [0, 2, 1, 3]);  view_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_22 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_90, permute_88, permute_89, expand_2, True, 0.1, scale = 0.125)
        getitem_247: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_22[0]
        getitem_248: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention_22[1]
        getitem_249: "i64[]" = _scaled_dot_product_efficient_attention_22[2]
        getitem_250: "i64[]" = _scaled_dot_product_efficient_attention_22[3];  _scaled_dot_product_efficient_attention_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_91: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_247, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_270: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_91, [4, 512, -1]);  permute_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_271: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_270, [-1, 1280]);  view_270 = None
        addmm_89: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_272, view_271, primals_273);  primals_272 = view_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_272: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_89, [4, 512, 1280]);  addmm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_45: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 45)
        inductor_random_default_27: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_45, 'rand');  inductor_lookup_seed_default_45 = None
        gt_45: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_27, 0.1);  inductor_random_default_27 = None
        mul_268: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_45, view_272);  view_272 = None
        mul_269: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_268, 1.1111111111111112);  mul_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_182: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_269, add_179);  mul_269 = add_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_45 = torch.ops.aten.var_mean.correction(add_182, [2], correction = 0, keepdim = True)
        getitem_251: "f32[4, 512, 1]" = var_mean_45[0]
        getitem_252: "f32[4, 512, 1]" = var_mean_45[1];  var_mean_45 = None
        add_183: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_251, 1e-05);  getitem_251 = None
        rsqrt_45: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_183);  add_183 = None
        sub_47: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_182, getitem_252);  getitem_252 = None
        mul_270: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_45);  sub_47 = None
        mul_271: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_270, primals_274)
        add_184: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_271, primals_275);  mul_271 = primals_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_273: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_184, [-1, 1280]);  add_184 = None
        addmm_90: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_276, view_273, primals_277);  primals_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_274: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_90, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_272: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_274, 0.5)
        pow_23: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_274, 3.0)
        mul_273: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_23, 0.044715);  pow_23 = None
        add_185: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_274, mul_273);  view_274 = mul_273 = None
        mul_274: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_185, 0.7978845608028654);  add_185 = None
        tanh_22: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_274);  mul_274 = None
        add_186: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_22, 1.0);  tanh_22 = None
        mul_275: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_272, add_186);  mul_272 = add_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_275: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_275, [-1, 5120]);  mul_275 = None
        addmm_91: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_278, view_275, primals_279);  primals_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_276: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_91, [4, 512, 1280]);  addmm_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_46: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 46)
        inductor_random_default_26: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_46, 'rand');  inductor_lookup_seed_default_46 = None
        gt_46: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_26, 0.1);  inductor_random_default_26 = None
        mul_276: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_46, view_276);  view_276 = None
        mul_277: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_276, 1.1111111111111112);  mul_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_187: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_182, mul_277);  add_182 = mul_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_46 = torch.ops.aten.var_mean.correction(add_187, [2], correction = 0, keepdim = True)
        getitem_253: "f32[4, 512, 1]" = var_mean_46[0]
        getitem_254: "f32[4, 512, 1]" = var_mean_46[1];  var_mean_46 = None
        add_188: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_253, 1e-05);  getitem_253 = None
        rsqrt_46: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
        sub_48: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_187, getitem_254);  getitem_254 = None
        mul_278: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_46);  sub_48 = None
        mul_279: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_278, primals_280)
        add_189: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_279, primals_281);  mul_279 = primals_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_277: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_189, [-1, 1280]);  add_189 = None
        addmm_92: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_282, view_277, primals_283);  primals_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_278: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_92, [4, 512, 3840]);  addmm_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_23 = torch.ops.aten.split.Tensor(view_278, 1280, 2);  view_278 = None
        getitem_255: "f32[4, 512, 1280]" = split_23[0]
        getitem_256: "f32[4, 512, 1280]" = split_23[1]
        getitem_257: "f32[4, 512, 1280]" = split_23[2];  split_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_279: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_256, [4, 512, -1, 64]);  getitem_256 = None
        permute_92: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_279, [0, 2, 1, 3]);  view_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_280: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_257, [4, 512, -1, 64]);  getitem_257 = None
        permute_93: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_280, [0, 2, 1, 3]);  view_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_281: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_255, [4, 512, -1, 64]);  getitem_255 = None
        permute_94: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_281, [0, 2, 1, 3]);  view_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_23 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_94, permute_92, permute_93, expand_2, True, 0.1, scale = 0.125)
        getitem_258: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_23[0]
        getitem_259: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention_23[1]
        getitem_260: "i64[]" = _scaled_dot_product_efficient_attention_23[2]
        getitem_261: "i64[]" = _scaled_dot_product_efficient_attention_23[3];  _scaled_dot_product_efficient_attention_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_95: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_258, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_282: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_95, [4, 512, -1]);  permute_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_283: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_282, [-1, 1280]);  view_282 = None
        addmm_93: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_284, view_283, primals_285);  primals_284 = view_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_284: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_93, [4, 512, 1280]);  addmm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_47: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 47)
        inductor_random_default_25: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_47, 'rand');  inductor_lookup_seed_default_47 = None
        gt_47: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_25, 0.1);  inductor_random_default_25 = None
        mul_280: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_47, view_284);  view_284 = None
        mul_281: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_280, 1.1111111111111112);  mul_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_190: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_281, add_187);  mul_281 = add_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_47 = torch.ops.aten.var_mean.correction(add_190, [2], correction = 0, keepdim = True)
        getitem_262: "f32[4, 512, 1]" = var_mean_47[0]
        getitem_263: "f32[4, 512, 1]" = var_mean_47[1];  var_mean_47 = None
        add_191: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_262, 1e-05);  getitem_262 = None
        rsqrt_47: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
        sub_49: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_190, getitem_263);  getitem_263 = None
        mul_282: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_47);  sub_49 = None
        mul_283: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_282, primals_286)
        add_192: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_283, primals_287);  mul_283 = primals_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_285: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_192, [-1, 1280]);  add_192 = None
        addmm_94: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_288, view_285, primals_289);  primals_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_286: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_94, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_284: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_286, 0.5)
        pow_24: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_286, 3.0)
        mul_285: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_24, 0.044715);  pow_24 = None
        add_193: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_286, mul_285);  view_286 = mul_285 = None
        mul_286: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_193, 0.7978845608028654);  add_193 = None
        tanh_23: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_286);  mul_286 = None
        add_194: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_23, 1.0);  tanh_23 = None
        mul_287: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_284, add_194);  mul_284 = add_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_287: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_287, [-1, 5120]);  mul_287 = None
        addmm_95: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_290, view_287, primals_291);  primals_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_288: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_95, [4, 512, 1280]);  addmm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_48: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 48)
        inductor_random_default_24: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_48, 'rand');  inductor_lookup_seed_default_48 = None
        gt_48: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_24, 0.1);  inductor_random_default_24 = None
        mul_288: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_48, view_288);  view_288 = None
        mul_289: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_288, 1.1111111111111112);  mul_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_195: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_190, mul_289);  add_190 = mul_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_48 = torch.ops.aten.var_mean.correction(add_195, [2], correction = 0, keepdim = True)
        getitem_264: "f32[4, 512, 1]" = var_mean_48[0]
        getitem_265: "f32[4, 512, 1]" = var_mean_48[1];  var_mean_48 = None
        add_196: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_264, 1e-05);  getitem_264 = None
        rsqrt_48: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_196);  add_196 = None
        sub_50: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_195, getitem_265);  getitem_265 = None
        mul_290: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_48);  sub_50 = None
        mul_291: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_290, primals_292)
        add_197: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_291, primals_293);  mul_291 = primals_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_289: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_197, [-1, 1280]);  add_197 = None
        addmm_96: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_294, view_289, primals_295);  primals_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_290: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_96, [4, 512, 3840]);  addmm_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_24 = torch.ops.aten.split.Tensor(view_290, 1280, 2);  view_290 = None
        getitem_266: "f32[4, 512, 1280]" = split_24[0]
        getitem_267: "f32[4, 512, 1280]" = split_24[1]
        getitem_268: "f32[4, 512, 1280]" = split_24[2];  split_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_291: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_267, [4, 512, -1, 64]);  getitem_267 = None
        permute_96: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_291, [0, 2, 1, 3]);  view_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_292: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_268, [4, 512, -1, 64]);  getitem_268 = None
        permute_97: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_293: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_266, [4, 512, -1, 64]);  getitem_266 = None
        permute_98: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_293, [0, 2, 1, 3]);  view_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_24 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_98, permute_96, permute_97, expand_2, True, 0.1, scale = 0.125)
        getitem_269: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_24[0]
        getitem_270: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention_24[1]
        getitem_271: "i64[]" = _scaled_dot_product_efficient_attention_24[2]
        getitem_272: "i64[]" = _scaled_dot_product_efficient_attention_24[3];  _scaled_dot_product_efficient_attention_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_99: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_269, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_294: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_99, [4, 512, -1]);  permute_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_295: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_294, [-1, 1280]);  view_294 = None
        addmm_97: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_296, view_295, primals_297);  primals_296 = view_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_296: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_97, [4, 512, 1280]);  addmm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_49: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 49)
        inductor_random_default_23: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_49, 'rand');  inductor_lookup_seed_default_49 = None
        gt_49: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_23, 0.1);  inductor_random_default_23 = None
        mul_292: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_49, view_296);  view_296 = None
        mul_293: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_292, 1.1111111111111112);  mul_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_198: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_293, add_195);  mul_293 = add_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_49 = torch.ops.aten.var_mean.correction(add_198, [2], correction = 0, keepdim = True)
        getitem_273: "f32[4, 512, 1]" = var_mean_49[0]
        getitem_274: "f32[4, 512, 1]" = var_mean_49[1];  var_mean_49 = None
        add_199: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_273, 1e-05);  getitem_273 = None
        rsqrt_49: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
        sub_51: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_198, getitem_274);  getitem_274 = None
        mul_294: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_49);  sub_51 = None
        mul_295: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_294, primals_298)
        add_200: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_295, primals_299);  mul_295 = primals_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_297: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_200, [-1, 1280]);  add_200 = None
        addmm_98: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_300, view_297, primals_301);  primals_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_298: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_98, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_296: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_298, 0.5)
        pow_25: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_298, 3.0)
        mul_297: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_25, 0.044715);  pow_25 = None
        add_201: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_298, mul_297);  view_298 = mul_297 = None
        mul_298: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_201, 0.7978845608028654);  add_201 = None
        tanh_24: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_298);  mul_298 = None
        add_202: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_24, 1.0);  tanh_24 = None
        mul_299: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_296, add_202);  mul_296 = add_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_299: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_299, [-1, 5120]);  mul_299 = None
        addmm_99: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_302, view_299, primals_303);  primals_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_300: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_99, [4, 512, 1280]);  addmm_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_50: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 50)
        inductor_random_default_22: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_50, 'rand');  inductor_lookup_seed_default_50 = None
        gt_50: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_22, 0.1);  inductor_random_default_22 = None
        mul_300: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_50, view_300);  view_300 = None
        mul_301: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_300, 1.1111111111111112);  mul_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_203: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_198, mul_301);  add_198 = mul_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_50 = torch.ops.aten.var_mean.correction(add_203, [2], correction = 0, keepdim = True)
        getitem_275: "f32[4, 512, 1]" = var_mean_50[0]
        getitem_276: "f32[4, 512, 1]" = var_mean_50[1];  var_mean_50 = None
        add_204: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_275, 1e-05);  getitem_275 = None
        rsqrt_50: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_204);  add_204 = None
        sub_52: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_203, getitem_276);  getitem_276 = None
        mul_302: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_50);  sub_52 = None
        mul_303: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_302, primals_304)
        add_205: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_303, primals_305);  mul_303 = primals_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_301: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_205, [-1, 1280]);  add_205 = None
        addmm_100: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_306, view_301, primals_307);  primals_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_302: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_100, [4, 512, 3840]);  addmm_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_25 = torch.ops.aten.split.Tensor(view_302, 1280, 2);  view_302 = None
        getitem_277: "f32[4, 512, 1280]" = split_25[0]
        getitem_278: "f32[4, 512, 1280]" = split_25[1]
        getitem_279: "f32[4, 512, 1280]" = split_25[2];  split_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_303: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_278, [4, 512, -1, 64]);  getitem_278 = None
        permute_100: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_303, [0, 2, 1, 3]);  view_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_304: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_279, [4, 512, -1, 64]);  getitem_279 = None
        permute_101: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_304, [0, 2, 1, 3]);  view_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_305: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_277, [4, 512, -1, 64]);  getitem_277 = None
        permute_102: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_305, [0, 2, 1, 3]);  view_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_25 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_102, permute_100, permute_101, expand_2, True, 0.1, scale = 0.125)
        getitem_280: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_25[0]
        getitem_281: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention_25[1]
        getitem_282: "i64[]" = _scaled_dot_product_efficient_attention_25[2]
        getitem_283: "i64[]" = _scaled_dot_product_efficient_attention_25[3];  _scaled_dot_product_efficient_attention_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_103: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_280, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_306: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_103, [4, 512, -1]);  permute_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_307: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_306, [-1, 1280]);  view_306 = None
        addmm_101: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_308, view_307, primals_309);  primals_308 = view_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_308: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_101, [4, 512, 1280]);  addmm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_51: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 51)
        inductor_random_default_21: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_51, 'rand');  inductor_lookup_seed_default_51 = None
        gt_51: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_21, 0.1);  inductor_random_default_21 = None
        mul_304: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_51, view_308);  view_308 = None
        mul_305: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_304, 1.1111111111111112);  mul_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_206: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_305, add_203);  mul_305 = add_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_51 = torch.ops.aten.var_mean.correction(add_206, [2], correction = 0, keepdim = True)
        getitem_284: "f32[4, 512, 1]" = var_mean_51[0]
        getitem_285: "f32[4, 512, 1]" = var_mean_51[1];  var_mean_51 = None
        add_207: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_284, 1e-05);  getitem_284 = None
        rsqrt_51: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_207);  add_207 = None
        sub_53: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_206, getitem_285);  getitem_285 = None
        mul_306: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_51);  sub_53 = None
        mul_307: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_306, primals_310)
        add_208: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_307, primals_311);  mul_307 = primals_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_309: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_208, [-1, 1280]);  add_208 = None
        addmm_102: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_312, view_309, primals_313);  primals_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_310: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_102, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_308: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_310, 0.5)
        pow_26: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_310, 3.0)
        mul_309: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_26, 0.044715);  pow_26 = None
        add_209: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_310, mul_309);  view_310 = mul_309 = None
        mul_310: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_209, 0.7978845608028654);  add_209 = None
        tanh_25: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_310);  mul_310 = None
        add_210: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_25, 1.0);  tanh_25 = None
        mul_311: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_308, add_210);  mul_308 = add_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_311: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_311, [-1, 5120]);  mul_311 = None
        addmm_103: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_314, view_311, primals_315);  primals_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_312: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_103, [4, 512, 1280]);  addmm_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_52: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 52)
        inductor_random_default_20: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_52, 'rand');  inductor_lookup_seed_default_52 = None
        gt_52: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_20, 0.1);  inductor_random_default_20 = None
        mul_312: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_52, view_312);  view_312 = None
        mul_313: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_312, 1.1111111111111112);  mul_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_211: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_206, mul_313);  add_206 = mul_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_52 = torch.ops.aten.var_mean.correction(add_211, [2], correction = 0, keepdim = True)
        getitem_286: "f32[4, 512, 1]" = var_mean_52[0]
        getitem_287: "f32[4, 512, 1]" = var_mean_52[1];  var_mean_52 = None
        add_212: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_286, 1e-05);  getitem_286 = None
        rsqrt_52: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_212);  add_212 = None
        sub_54: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_211, getitem_287);  getitem_287 = None
        mul_314: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_52);  sub_54 = None
        mul_315: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_314, primals_316)
        add_213: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_315, primals_317);  mul_315 = primals_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_313: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_213, [-1, 1280]);  add_213 = None
        addmm_104: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_318, view_313, primals_319);  primals_318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_314: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_104, [4, 512, 3840]);  addmm_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_26 = torch.ops.aten.split.Tensor(view_314, 1280, 2);  view_314 = None
        getitem_288: "f32[4, 512, 1280]" = split_26[0]
        getitem_289: "f32[4, 512, 1280]" = split_26[1]
        getitem_290: "f32[4, 512, 1280]" = split_26[2];  split_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_315: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_289, [4, 512, -1, 64]);  getitem_289 = None
        permute_104: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_315, [0, 2, 1, 3]);  view_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_316: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_290, [4, 512, -1, 64]);  getitem_290 = None
        permute_105: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_317: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_288, [4, 512, -1, 64]);  getitem_288 = None
        permute_106: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_317, [0, 2, 1, 3]);  view_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_26 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_106, permute_104, permute_105, expand_2, True, 0.1, scale = 0.125)
        getitem_291: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_26[0]
        getitem_292: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention_26[1]
        getitem_293: "i64[]" = _scaled_dot_product_efficient_attention_26[2]
        getitem_294: "i64[]" = _scaled_dot_product_efficient_attention_26[3];  _scaled_dot_product_efficient_attention_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_107: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_291, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_318: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_107, [4, 512, -1]);  permute_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_319: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_318, [-1, 1280]);  view_318 = None
        addmm_105: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_320, view_319, primals_321);  primals_320 = view_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_320: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_105, [4, 512, 1280]);  addmm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_53: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 53)
        inductor_random_default_19: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_53, 'rand');  inductor_lookup_seed_default_53 = None
        gt_53: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_19, 0.1);  inductor_random_default_19 = None
        mul_316: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_53, view_320);  view_320 = None
        mul_317: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_316, 1.1111111111111112);  mul_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_214: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_317, add_211);  mul_317 = add_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_53 = torch.ops.aten.var_mean.correction(add_214, [2], correction = 0, keepdim = True)
        getitem_295: "f32[4, 512, 1]" = var_mean_53[0]
        getitem_296: "f32[4, 512, 1]" = var_mean_53[1];  var_mean_53 = None
        add_215: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_295, 1e-05);  getitem_295 = None
        rsqrt_53: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_215);  add_215 = None
        sub_55: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_214, getitem_296);  getitem_296 = None
        mul_318: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_53);  sub_55 = None
        mul_319: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_318, primals_322)
        add_216: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_319, primals_323);  mul_319 = primals_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_321: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_216, [-1, 1280]);  add_216 = None
        addmm_106: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_324, view_321, primals_325);  primals_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_322: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_106, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_320: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_322, 0.5)
        pow_27: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_322, 3.0)
        mul_321: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_27, 0.044715);  pow_27 = None
        add_217: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_322, mul_321);  view_322 = mul_321 = None
        mul_322: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_217, 0.7978845608028654);  add_217 = None
        tanh_26: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_322);  mul_322 = None
        add_218: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_26, 1.0);  tanh_26 = None
        mul_323: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_320, add_218);  mul_320 = add_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_323: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_323, [-1, 5120]);  mul_323 = None
        addmm_107: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_326, view_323, primals_327);  primals_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_324: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_107, [4, 512, 1280]);  addmm_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_54: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 54)
        inductor_random_default_18: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_54, 'rand');  inductor_lookup_seed_default_54 = None
        gt_54: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_18, 0.1);  inductor_random_default_18 = None
        mul_324: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_54, view_324);  view_324 = None
        mul_325: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_324, 1.1111111111111112);  mul_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_219: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_214, mul_325);  add_214 = mul_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_54 = torch.ops.aten.var_mean.correction(add_219, [2], correction = 0, keepdim = True)
        getitem_297: "f32[4, 512, 1]" = var_mean_54[0]
        getitem_298: "f32[4, 512, 1]" = var_mean_54[1];  var_mean_54 = None
        add_220: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_297, 1e-05);  getitem_297 = None
        rsqrt_54: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_220);  add_220 = None
        sub_56: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_219, getitem_298);  getitem_298 = None
        mul_326: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_54);  sub_56 = None
        mul_327: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_326, primals_328)
        add_221: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_327, primals_329);  mul_327 = primals_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_325: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_221, [-1, 1280]);  add_221 = None
        addmm_108: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_330, view_325, primals_331);  primals_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_326: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_108, [4, 512, 3840]);  addmm_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_27 = torch.ops.aten.split.Tensor(view_326, 1280, 2);  view_326 = None
        getitem_299: "f32[4, 512, 1280]" = split_27[0]
        getitem_300: "f32[4, 512, 1280]" = split_27[1]
        getitem_301: "f32[4, 512, 1280]" = split_27[2];  split_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_327: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_300, [4, 512, -1, 64]);  getitem_300 = None
        permute_108: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_327, [0, 2, 1, 3]);  view_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_328: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_301, [4, 512, -1, 64]);  getitem_301 = None
        permute_109: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_328, [0, 2, 1, 3]);  view_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_329: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_299, [4, 512, -1, 64]);  getitem_299 = None
        permute_110: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_329, [0, 2, 1, 3]);  view_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_27 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_110, permute_108, permute_109, expand_2, True, 0.1, scale = 0.125)
        getitem_302: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_27[0]
        getitem_303: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention_27[1]
        getitem_304: "i64[]" = _scaled_dot_product_efficient_attention_27[2]
        getitem_305: "i64[]" = _scaled_dot_product_efficient_attention_27[3];  _scaled_dot_product_efficient_attention_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_111: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_302, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_330: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_111, [4, 512, -1]);  permute_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_331: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_330, [-1, 1280]);  view_330 = None
        addmm_109: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_332, view_331, primals_333);  primals_332 = view_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_332: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_109, [4, 512, 1280]);  addmm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_55: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 55)
        inductor_random_default_17: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_55, 'rand');  inductor_lookup_seed_default_55 = None
        gt_55: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_17, 0.1);  inductor_random_default_17 = None
        mul_328: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_55, view_332);  view_332 = None
        mul_329: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_328, 1.1111111111111112);  mul_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_222: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_329, add_219);  mul_329 = add_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_55 = torch.ops.aten.var_mean.correction(add_222, [2], correction = 0, keepdim = True)
        getitem_306: "f32[4, 512, 1]" = var_mean_55[0]
        getitem_307: "f32[4, 512, 1]" = var_mean_55[1];  var_mean_55 = None
        add_223: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_306, 1e-05);  getitem_306 = None
        rsqrt_55: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_223);  add_223 = None
        sub_57: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_222, getitem_307);  getitem_307 = None
        mul_330: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_55);  sub_57 = None
        mul_331: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_330, primals_334)
        add_224: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_331, primals_335);  mul_331 = primals_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_333: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_224, [-1, 1280]);  add_224 = None
        addmm_110: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_336, view_333, primals_337);  primals_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_334: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_110, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_332: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_334, 0.5)
        pow_28: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_334, 3.0)
        mul_333: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_28, 0.044715);  pow_28 = None
        add_225: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_334, mul_333);  view_334 = mul_333 = None
        mul_334: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_225, 0.7978845608028654);  add_225 = None
        tanh_27: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_334);  mul_334 = None
        add_226: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_27, 1.0);  tanh_27 = None
        mul_335: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_332, add_226);  mul_332 = add_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_335: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_335, [-1, 5120]);  mul_335 = None
        addmm_111: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_338, view_335, primals_339);  primals_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_336: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_111, [4, 512, 1280]);  addmm_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_56: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 56)
        inductor_random_default_16: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_56, 'rand');  inductor_lookup_seed_default_56 = None
        gt_56: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_16, 0.1);  inductor_random_default_16 = None
        mul_336: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_56, view_336);  view_336 = None
        mul_337: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_336, 1.1111111111111112);  mul_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_227: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_222, mul_337);  add_222 = mul_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_56 = torch.ops.aten.var_mean.correction(add_227, [2], correction = 0, keepdim = True)
        getitem_308: "f32[4, 512, 1]" = var_mean_56[0]
        getitem_309: "f32[4, 512, 1]" = var_mean_56[1];  var_mean_56 = None
        add_228: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_308, 1e-05);  getitem_308 = None
        rsqrt_56: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_228);  add_228 = None
        sub_58: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_227, getitem_309);  getitem_309 = None
        mul_338: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_56);  sub_58 = None
        mul_339: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_338, primals_340)
        add_229: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_339, primals_341);  mul_339 = primals_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_337: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_229, [-1, 1280]);  add_229 = None
        addmm_112: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_342, view_337, primals_343);  primals_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_338: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_112, [4, 512, 3840]);  addmm_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_28 = torch.ops.aten.split.Tensor(view_338, 1280, 2);  view_338 = None
        getitem_310: "f32[4, 512, 1280]" = split_28[0]
        getitem_311: "f32[4, 512, 1280]" = split_28[1]
        getitem_312: "f32[4, 512, 1280]" = split_28[2];  split_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_339: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_311, [4, 512, -1, 64]);  getitem_311 = None
        permute_112: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_339, [0, 2, 1, 3]);  view_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_340: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_312, [4, 512, -1, 64]);  getitem_312 = None
        permute_113: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_340, [0, 2, 1, 3]);  view_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_341: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_310, [4, 512, -1, 64]);  getitem_310 = None
        permute_114: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_341, [0, 2, 1, 3]);  view_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_28 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_114, permute_112, permute_113, expand_2, True, 0.1, scale = 0.125)
        getitem_313: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_28[0]
        getitem_314: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention_28[1]
        getitem_315: "i64[]" = _scaled_dot_product_efficient_attention_28[2]
        getitem_316: "i64[]" = _scaled_dot_product_efficient_attention_28[3];  _scaled_dot_product_efficient_attention_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_115: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_313, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_342: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_115, [4, 512, -1]);  permute_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_343: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_342, [-1, 1280]);  view_342 = None
        addmm_113: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_344, view_343, primals_345);  primals_344 = view_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_344: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_113, [4, 512, 1280]);  addmm_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_57: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 57)
        inductor_random_default_15: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_57, 'rand');  inductor_lookup_seed_default_57 = None
        gt_57: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_15, 0.1);  inductor_random_default_15 = None
        mul_340: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_57, view_344);  view_344 = None
        mul_341: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_340, 1.1111111111111112);  mul_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_230: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_341, add_227);  mul_341 = add_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_57 = torch.ops.aten.var_mean.correction(add_230, [2], correction = 0, keepdim = True)
        getitem_317: "f32[4, 512, 1]" = var_mean_57[0]
        getitem_318: "f32[4, 512, 1]" = var_mean_57[1];  var_mean_57 = None
        add_231: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_317, 1e-05);  getitem_317 = None
        rsqrt_57: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_231);  add_231 = None
        sub_59: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_230, getitem_318);  getitem_318 = None
        mul_342: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_57);  sub_59 = None
        mul_343: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_342, primals_346)
        add_232: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_343, primals_347);  mul_343 = primals_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_345: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_232, [-1, 1280]);  add_232 = None
        addmm_114: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_348, view_345, primals_349);  primals_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_346: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_114, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_344: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_346, 0.5)
        pow_29: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_346, 3.0)
        mul_345: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_29, 0.044715);  pow_29 = None
        add_233: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_346, mul_345);  view_346 = mul_345 = None
        mul_346: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_233, 0.7978845608028654);  add_233 = None
        tanh_28: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_346);  mul_346 = None
        add_234: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_28, 1.0);  tanh_28 = None
        mul_347: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_344, add_234);  mul_344 = add_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_347: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_347, [-1, 5120]);  mul_347 = None
        addmm_115: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_350, view_347, primals_351);  primals_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_348: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_115, [4, 512, 1280]);  addmm_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_58: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 58)
        inductor_random_default_14: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_58, 'rand');  inductor_lookup_seed_default_58 = None
        gt_58: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_14, 0.1);  inductor_random_default_14 = None
        mul_348: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_58, view_348);  view_348 = None
        mul_349: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_348, 1.1111111111111112);  mul_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_235: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_230, mul_349);  add_230 = mul_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_58 = torch.ops.aten.var_mean.correction(add_235, [2], correction = 0, keepdim = True)
        getitem_319: "f32[4, 512, 1]" = var_mean_58[0]
        getitem_320: "f32[4, 512, 1]" = var_mean_58[1];  var_mean_58 = None
        add_236: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_319, 1e-05);  getitem_319 = None
        rsqrt_58: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_236);  add_236 = None
        sub_60: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_235, getitem_320);  getitem_320 = None
        mul_350: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_58);  sub_60 = None
        mul_351: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_350, primals_352)
        add_237: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_351, primals_353);  mul_351 = primals_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_349: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_237, [-1, 1280]);  add_237 = None
        addmm_116: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_354, view_349, primals_355);  primals_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_350: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_116, [4, 512, 3840]);  addmm_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_29 = torch.ops.aten.split.Tensor(view_350, 1280, 2);  view_350 = None
        getitem_321: "f32[4, 512, 1280]" = split_29[0]
        getitem_322: "f32[4, 512, 1280]" = split_29[1]
        getitem_323: "f32[4, 512, 1280]" = split_29[2];  split_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_351: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_322, [4, 512, -1, 64]);  getitem_322 = None
        permute_116: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_351, [0, 2, 1, 3]);  view_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_352: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_323, [4, 512, -1, 64]);  getitem_323 = None
        permute_117: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_352, [0, 2, 1, 3]);  view_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_353: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_321, [4, 512, -1, 64]);  getitem_321 = None
        permute_118: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_353, [0, 2, 1, 3]);  view_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_29 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_118, permute_116, permute_117, expand_2, True, 0.1, scale = 0.125)
        getitem_324: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_29[0]
        getitem_325: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention_29[1]
        getitem_326: "i64[]" = _scaled_dot_product_efficient_attention_29[2]
        getitem_327: "i64[]" = _scaled_dot_product_efficient_attention_29[3];  _scaled_dot_product_efficient_attention_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_119: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_324, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_354: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_119, [4, 512, -1]);  permute_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_355: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_354, [-1, 1280]);  view_354 = None
        addmm_117: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_356, view_355, primals_357);  primals_356 = view_355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_356: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_117, [4, 512, 1280]);  addmm_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_59: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 59)
        inductor_random_default_13: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_59, 'rand');  inductor_lookup_seed_default_59 = None
        gt_59: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_13, 0.1);  inductor_random_default_13 = None
        mul_352: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_59, view_356);  view_356 = None
        mul_353: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_352, 1.1111111111111112);  mul_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_238: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_353, add_235);  mul_353 = add_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_59 = torch.ops.aten.var_mean.correction(add_238, [2], correction = 0, keepdim = True)
        getitem_328: "f32[4, 512, 1]" = var_mean_59[0]
        getitem_329: "f32[4, 512, 1]" = var_mean_59[1];  var_mean_59 = None
        add_239: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_328, 1e-05);  getitem_328 = None
        rsqrt_59: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_239);  add_239 = None
        sub_61: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_238, getitem_329);  getitem_329 = None
        mul_354: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_59);  sub_61 = None
        mul_355: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_354, primals_358)
        add_240: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_355, primals_359);  mul_355 = primals_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_357: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_240, [-1, 1280]);  add_240 = None
        addmm_118: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_360, view_357, primals_361);  primals_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_358: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_118, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_356: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_358, 0.5)
        pow_30: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_358, 3.0)
        mul_357: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_30, 0.044715);  pow_30 = None
        add_241: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_358, mul_357);  view_358 = mul_357 = None
        mul_358: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_241, 0.7978845608028654);  add_241 = None
        tanh_29: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_358);  mul_358 = None
        add_242: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_29, 1.0);  tanh_29 = None
        mul_359: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_356, add_242);  mul_356 = add_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_359: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_359, [-1, 5120]);  mul_359 = None
        addmm_119: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_362, view_359, primals_363);  primals_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_360: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_119, [4, 512, 1280]);  addmm_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_60: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 60)
        inductor_random_default_12: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_60, 'rand');  inductor_lookup_seed_default_60 = None
        gt_60: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_12, 0.1);  inductor_random_default_12 = None
        mul_360: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_60, view_360);  view_360 = None
        mul_361: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_360, 1.1111111111111112);  mul_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_243: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_238, mul_361);  add_238 = mul_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_60 = torch.ops.aten.var_mean.correction(add_243, [2], correction = 0, keepdim = True)
        getitem_330: "f32[4, 512, 1]" = var_mean_60[0]
        getitem_331: "f32[4, 512, 1]" = var_mean_60[1];  var_mean_60 = None
        add_244: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_330, 1e-05);  getitem_330 = None
        rsqrt_60: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_244);  add_244 = None
        sub_62: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_243, getitem_331);  getitem_331 = None
        mul_362: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_60);  sub_62 = None
        mul_363: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_362, primals_364)
        add_245: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_363, primals_365);  mul_363 = primals_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_361: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_245, [-1, 1280]);  add_245 = None
        addmm_120: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_366, view_361, primals_367);  primals_366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_362: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_120, [4, 512, 3840]);  addmm_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_30 = torch.ops.aten.split.Tensor(view_362, 1280, 2);  view_362 = None
        getitem_332: "f32[4, 512, 1280]" = split_30[0]
        getitem_333: "f32[4, 512, 1280]" = split_30[1]
        getitem_334: "f32[4, 512, 1280]" = split_30[2];  split_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_363: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_333, [4, 512, -1, 64]);  getitem_333 = None
        permute_120: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_363, [0, 2, 1, 3]);  view_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_364: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_334, [4, 512, -1, 64]);  getitem_334 = None
        permute_121: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_364, [0, 2, 1, 3]);  view_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_365: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_332, [4, 512, -1, 64]);  getitem_332 = None
        permute_122: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_365, [0, 2, 1, 3]);  view_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_30 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_122, permute_120, permute_121, expand_2, True, 0.1, scale = 0.125)
        getitem_335: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_30[0]
        getitem_336: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention_30[1]
        getitem_337: "i64[]" = _scaled_dot_product_efficient_attention_30[2]
        getitem_338: "i64[]" = _scaled_dot_product_efficient_attention_30[3];  _scaled_dot_product_efficient_attention_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_123: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_335, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_366: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_123, [4, 512, -1]);  permute_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_367: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_366, [-1, 1280]);  view_366 = None
        addmm_121: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_368, view_367, primals_369);  primals_368 = view_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_368: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_121, [4, 512, 1280]);  addmm_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_61: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 61)
        inductor_random_default_11: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_61, 'rand');  inductor_lookup_seed_default_61 = None
        gt_61: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_11, 0.1);  inductor_random_default_11 = None
        mul_364: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_61, view_368);  view_368 = None
        mul_365: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_364, 1.1111111111111112);  mul_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_246: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_365, add_243);  mul_365 = add_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_61 = torch.ops.aten.var_mean.correction(add_246, [2], correction = 0, keepdim = True)
        getitem_339: "f32[4, 512, 1]" = var_mean_61[0]
        getitem_340: "f32[4, 512, 1]" = var_mean_61[1];  var_mean_61 = None
        add_247: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_339, 1e-05);  getitem_339 = None
        rsqrt_61: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_247);  add_247 = None
        sub_63: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_246, getitem_340);  getitem_340 = None
        mul_366: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_61);  sub_63 = None
        mul_367: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_366, primals_370)
        add_248: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_367, primals_371);  mul_367 = primals_371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_369: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_248, [-1, 1280]);  add_248 = None
        addmm_122: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_372, view_369, primals_373);  primals_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_370: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_122, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_368: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_370, 0.5)
        pow_31: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_370, 3.0)
        mul_369: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_31, 0.044715);  pow_31 = None
        add_249: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_370, mul_369);  view_370 = mul_369 = None
        mul_370: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_249, 0.7978845608028654);  add_249 = None
        tanh_30: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_370);  mul_370 = None
        add_250: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_30, 1.0);  tanh_30 = None
        mul_371: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_368, add_250);  mul_368 = add_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_371: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_371, [-1, 5120]);  mul_371 = None
        addmm_123: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_374, view_371, primals_375);  primals_374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_372: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_123, [4, 512, 1280]);  addmm_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_62: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 62)
        inductor_random_default_10: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_62, 'rand');  inductor_lookup_seed_default_62 = None
        gt_62: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_10, 0.1);  inductor_random_default_10 = None
        mul_372: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_62, view_372);  view_372 = None
        mul_373: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_372, 1.1111111111111112);  mul_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_251: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_246, mul_373);  add_246 = mul_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_62 = torch.ops.aten.var_mean.correction(add_251, [2], correction = 0, keepdim = True)
        getitem_341: "f32[4, 512, 1]" = var_mean_62[0]
        getitem_342: "f32[4, 512, 1]" = var_mean_62[1];  var_mean_62 = None
        add_252: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_341, 1e-05);  getitem_341 = None
        rsqrt_62: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_252);  add_252 = None
        sub_64: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_251, getitem_342);  getitem_342 = None
        mul_374: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_62);  sub_64 = None
        mul_375: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_374, primals_376)
        add_253: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_375, primals_377);  mul_375 = primals_377 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_373: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_253, [-1, 1280]);  add_253 = None
        addmm_124: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_378, view_373, primals_379);  primals_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_374: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_124, [4, 512, 3840]);  addmm_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_31 = torch.ops.aten.split.Tensor(view_374, 1280, 2);  view_374 = None
        getitem_343: "f32[4, 512, 1280]" = split_31[0]
        getitem_344: "f32[4, 512, 1280]" = split_31[1]
        getitem_345: "f32[4, 512, 1280]" = split_31[2];  split_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_375: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_344, [4, 512, -1, 64]);  getitem_344 = None
        permute_124: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_375, [0, 2, 1, 3]);  view_375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_376: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_345, [4, 512, -1, 64]);  getitem_345 = None
        permute_125: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_376, [0, 2, 1, 3]);  view_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_377: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_343, [4, 512, -1, 64]);  getitem_343 = None
        permute_126: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_377, [0, 2, 1, 3]);  view_377 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_31 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_126, permute_124, permute_125, expand_2, True, 0.1, scale = 0.125)
        getitem_346: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_31[0]
        getitem_347: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention_31[1]
        getitem_348: "i64[]" = _scaled_dot_product_efficient_attention_31[2]
        getitem_349: "i64[]" = _scaled_dot_product_efficient_attention_31[3];  _scaled_dot_product_efficient_attention_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_127: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_346, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_378: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_127, [4, 512, -1]);  permute_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_379: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_378, [-1, 1280]);  view_378 = None
        addmm_125: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_380, view_379, primals_381);  primals_380 = view_379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_380: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_125, [4, 512, 1280]);  addmm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_63: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 63)
        inductor_random_default_9: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_63, 'rand');  inductor_lookup_seed_default_63 = None
        gt_63: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_9, 0.1);  inductor_random_default_9 = None
        mul_376: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_63, view_380);  view_380 = None
        mul_377: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_376, 1.1111111111111112);  mul_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_254: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_377, add_251);  mul_377 = add_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_63 = torch.ops.aten.var_mean.correction(add_254, [2], correction = 0, keepdim = True)
        getitem_350: "f32[4, 512, 1]" = var_mean_63[0]
        getitem_351: "f32[4, 512, 1]" = var_mean_63[1];  var_mean_63 = None
        add_255: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_350, 1e-05);  getitem_350 = None
        rsqrt_63: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_255);  add_255 = None
        sub_65: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_254, getitem_351);  getitem_351 = None
        mul_378: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_63);  sub_65 = None
        mul_379: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_378, primals_382)
        add_256: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_379, primals_383);  mul_379 = primals_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_381: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_256, [-1, 1280]);  add_256 = None
        addmm_126: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_384, view_381, primals_385);  primals_384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_382: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_126, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_380: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_382, 0.5)
        pow_32: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_382, 3.0)
        mul_381: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_32, 0.044715);  pow_32 = None
        add_257: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_382, mul_381);  view_382 = mul_381 = None
        mul_382: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_257, 0.7978845608028654);  add_257 = None
        tanh_31: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_382);  mul_382 = None
        add_258: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_31, 1.0);  tanh_31 = None
        mul_383: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_380, add_258);  mul_380 = add_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_383: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_383, [-1, 5120]);  mul_383 = None
        addmm_127: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_386, view_383, primals_387);  primals_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_384: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_127, [4, 512, 1280]);  addmm_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_64: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 64)
        inductor_random_default_8: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_64, 'rand');  inductor_lookup_seed_default_64 = None
        gt_64: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_8, 0.1);  inductor_random_default_8 = None
        mul_384: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_64, view_384);  view_384 = None
        mul_385: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_384, 1.1111111111111112);  mul_384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_259: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_254, mul_385);  add_254 = mul_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_64 = torch.ops.aten.var_mean.correction(add_259, [2], correction = 0, keepdim = True)
        getitem_352: "f32[4, 512, 1]" = var_mean_64[0]
        getitem_353: "f32[4, 512, 1]" = var_mean_64[1];  var_mean_64 = None
        add_260: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_352, 1e-05);  getitem_352 = None
        rsqrt_64: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_260);  add_260 = None
        sub_66: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_259, getitem_353);  getitem_353 = None
        mul_386: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_64);  sub_66 = None
        mul_387: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_386, primals_388)
        add_261: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_387, primals_389);  mul_387 = primals_389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_385: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_261, [-1, 1280]);  add_261 = None
        addmm_128: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_390, view_385, primals_391);  primals_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_386: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_128, [4, 512, 3840]);  addmm_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_32 = torch.ops.aten.split.Tensor(view_386, 1280, 2);  view_386 = None
        getitem_354: "f32[4, 512, 1280]" = split_32[0]
        getitem_355: "f32[4, 512, 1280]" = split_32[1]
        getitem_356: "f32[4, 512, 1280]" = split_32[2];  split_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_387: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_355, [4, 512, -1, 64]);  getitem_355 = None
        permute_128: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_387, [0, 2, 1, 3]);  view_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_388: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_356, [4, 512, -1, 64]);  getitem_356 = None
        permute_129: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_388, [0, 2, 1, 3]);  view_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_389: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_354, [4, 512, -1, 64]);  getitem_354 = None
        permute_130: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_389, [0, 2, 1, 3]);  view_389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_32 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_130, permute_128, permute_129, expand_2, True, 0.1, scale = 0.125)
        getitem_357: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_32[0]
        getitem_358: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention_32[1]
        getitem_359: "i64[]" = _scaled_dot_product_efficient_attention_32[2]
        getitem_360: "i64[]" = _scaled_dot_product_efficient_attention_32[3];  _scaled_dot_product_efficient_attention_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_131: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_357, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_390: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_131, [4, 512, -1]);  permute_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_391: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_390, [-1, 1280]);  view_390 = None
        addmm_129: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_392, view_391, primals_393);  primals_392 = view_391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_392: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_129, [4, 512, 1280]);  addmm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_65: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 65)
        inductor_random_default_7: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_65, 'rand');  inductor_lookup_seed_default_65 = None
        gt_65: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_7, 0.1);  inductor_random_default_7 = None
        mul_388: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_65, view_392);  view_392 = None
        mul_389: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_388, 1.1111111111111112);  mul_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_262: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_389, add_259);  mul_389 = add_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_65 = torch.ops.aten.var_mean.correction(add_262, [2], correction = 0, keepdim = True)
        getitem_361: "f32[4, 512, 1]" = var_mean_65[0]
        getitem_362: "f32[4, 512, 1]" = var_mean_65[1];  var_mean_65 = None
        add_263: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_361, 1e-05);  getitem_361 = None
        rsqrt_65: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_263);  add_263 = None
        sub_67: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_262, getitem_362);  getitem_362 = None
        mul_390: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_65);  sub_67 = None
        mul_391: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_390, primals_394)
        add_264: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_391, primals_395);  mul_391 = primals_395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_393: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_264, [-1, 1280]);  add_264 = None
        addmm_130: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_396, view_393, primals_397);  primals_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_394: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_130, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_392: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_394, 0.5)
        pow_33: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_394, 3.0)
        mul_393: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_33, 0.044715);  pow_33 = None
        add_265: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_394, mul_393);  view_394 = mul_393 = None
        mul_394: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_265, 0.7978845608028654);  add_265 = None
        tanh_32: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_394);  mul_394 = None
        add_266: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_32, 1.0);  tanh_32 = None
        mul_395: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_392, add_266);  mul_392 = add_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_395: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_395, [-1, 5120]);  mul_395 = None
        addmm_131: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_398, view_395, primals_399);  primals_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_396: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_131, [4, 512, 1280]);  addmm_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_66: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 66)
        inductor_random_default_6: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_66, 'rand');  inductor_lookup_seed_default_66 = None
        gt_66: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_6, 0.1);  inductor_random_default_6 = None
        mul_396: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_66, view_396);  view_396 = None
        mul_397: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_396, 1.1111111111111112);  mul_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_267: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_262, mul_397);  add_262 = mul_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_66 = torch.ops.aten.var_mean.correction(add_267, [2], correction = 0, keepdim = True)
        getitem_363: "f32[4, 512, 1]" = var_mean_66[0]
        getitem_364: "f32[4, 512, 1]" = var_mean_66[1];  var_mean_66 = None
        add_268: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_363, 1e-05);  getitem_363 = None
        rsqrt_66: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_268);  add_268 = None
        sub_68: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_267, getitem_364);  getitem_364 = None
        mul_398: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_66);  sub_68 = None
        mul_399: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_398, primals_400)
        add_269: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_399, primals_401);  mul_399 = primals_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_397: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_269, [-1, 1280]);  add_269 = None
        addmm_132: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_402, view_397, primals_403);  primals_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_398: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_132, [4, 512, 3840]);  addmm_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_33 = torch.ops.aten.split.Tensor(view_398, 1280, 2);  view_398 = None
        getitem_365: "f32[4, 512, 1280]" = split_33[0]
        getitem_366: "f32[4, 512, 1280]" = split_33[1]
        getitem_367: "f32[4, 512, 1280]" = split_33[2];  split_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_399: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_366, [4, 512, -1, 64]);  getitem_366 = None
        permute_132: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_399, [0, 2, 1, 3]);  view_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_400: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_367, [4, 512, -1, 64]);  getitem_367 = None
        permute_133: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_400, [0, 2, 1, 3]);  view_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_401: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_365, [4, 512, -1, 64]);  getitem_365 = None
        permute_134: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_401, [0, 2, 1, 3]);  view_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_33 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_134, permute_132, permute_133, expand_2, True, 0.1, scale = 0.125)
        getitem_368: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_33[0]
        getitem_369: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention_33[1]
        getitem_370: "i64[]" = _scaled_dot_product_efficient_attention_33[2]
        getitem_371: "i64[]" = _scaled_dot_product_efficient_attention_33[3];  _scaled_dot_product_efficient_attention_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_135: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_368, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_402: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_135, [4, 512, -1]);  permute_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_403: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_402, [-1, 1280]);  view_402 = None
        addmm_133: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_404, view_403, primals_405);  primals_404 = view_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_404: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_133, [4, 512, 1280]);  addmm_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_67: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 67)
        inductor_random_default_5: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_67, 'rand');  inductor_lookup_seed_default_67 = None
        gt_67: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_5, 0.1);  inductor_random_default_5 = None
        mul_400: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_67, view_404);  view_404 = None
        mul_401: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_400, 1.1111111111111112);  mul_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_270: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_401, add_267);  mul_401 = add_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_67 = torch.ops.aten.var_mean.correction(add_270, [2], correction = 0, keepdim = True)
        getitem_372: "f32[4, 512, 1]" = var_mean_67[0]
        getitem_373: "f32[4, 512, 1]" = var_mean_67[1];  var_mean_67 = None
        add_271: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_372, 1e-05);  getitem_372 = None
        rsqrt_67: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_271);  add_271 = None
        sub_69: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_270, getitem_373);  getitem_373 = None
        mul_402: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_67);  sub_69 = None
        mul_403: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_402, primals_406)
        add_272: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_403, primals_407);  mul_403 = primals_407 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_405: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_272, [-1, 1280]);  add_272 = None
        addmm_134: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_408, view_405, primals_409);  primals_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_406: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_134, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_404: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_406, 0.5)
        pow_34: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_406, 3.0)
        mul_405: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_34, 0.044715);  pow_34 = None
        add_273: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_406, mul_405);  view_406 = mul_405 = None
        mul_406: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_273, 0.7978845608028654);  add_273 = None
        tanh_33: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_406);  mul_406 = None
        add_274: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_33, 1.0);  tanh_33 = None
        mul_407: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_404, add_274);  mul_404 = add_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_407: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_407, [-1, 5120]);  mul_407 = None
        addmm_135: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_410, view_407, primals_411);  primals_410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_408: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_135, [4, 512, 1280]);  addmm_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_68: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 68)
        inductor_random_default_4: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_68, 'rand');  inductor_lookup_seed_default_68 = None
        gt_68: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_4, 0.1);  inductor_random_default_4 = None
        mul_408: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_68, view_408);  view_408 = None
        mul_409: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_408, 1.1111111111111112);  mul_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_275: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_270, mul_409);  add_270 = mul_409 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_68 = torch.ops.aten.var_mean.correction(add_275, [2], correction = 0, keepdim = True)
        getitem_374: "f32[4, 512, 1]" = var_mean_68[0]
        getitem_375: "f32[4, 512, 1]" = var_mean_68[1];  var_mean_68 = None
        add_276: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_374, 1e-05);  getitem_374 = None
        rsqrt_68: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_276);  add_276 = None
        sub_70: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_275, getitem_375);  getitem_375 = None
        mul_410: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_68);  sub_70 = None
        mul_411: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_410, primals_412)
        add_277: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_411, primals_413);  mul_411 = primals_413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_409: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_277, [-1, 1280]);  add_277 = None
        addmm_136: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_414, view_409, primals_415);  primals_414 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_410: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_136, [4, 512, 3840]);  addmm_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_34 = torch.ops.aten.split.Tensor(view_410, 1280, 2);  view_410 = None
        getitem_376: "f32[4, 512, 1280]" = split_34[0]
        getitem_377: "f32[4, 512, 1280]" = split_34[1]
        getitem_378: "f32[4, 512, 1280]" = split_34[2];  split_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_411: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_377, [4, 512, -1, 64]);  getitem_377 = None
        permute_136: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_411, [0, 2, 1, 3]);  view_411 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_412: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_378, [4, 512, -1, 64]);  getitem_378 = None
        permute_137: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_412, [0, 2, 1, 3]);  view_412 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_413: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_376, [4, 512, -1, 64]);  getitem_376 = None
        permute_138: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_413, [0, 2, 1, 3]);  view_413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_34 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_138, permute_136, permute_137, expand_2, True, 0.1, scale = 0.125)
        getitem_379: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_34[0]
        getitem_380: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention_34[1]
        getitem_381: "i64[]" = _scaled_dot_product_efficient_attention_34[2]
        getitem_382: "i64[]" = _scaled_dot_product_efficient_attention_34[3];  _scaled_dot_product_efficient_attention_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_139: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_379, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_414: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_139, [4, 512, -1]);  permute_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_415: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_414, [-1, 1280]);  view_414 = None
        addmm_137: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_416, view_415, primals_417);  primals_416 = view_415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_416: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_137, [4, 512, 1280]);  addmm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_69: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 69)
        inductor_random_default_3: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_69, 'rand');  inductor_lookup_seed_default_69 = None
        gt_69: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_3, 0.1);  inductor_random_default_3 = None
        mul_412: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_69, view_416);  view_416 = None
        mul_413: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_412, 1.1111111111111112);  mul_412 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_278: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_413, add_275);  mul_413 = add_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_69 = torch.ops.aten.var_mean.correction(add_278, [2], correction = 0, keepdim = True)
        getitem_383: "f32[4, 512, 1]" = var_mean_69[0]
        getitem_384: "f32[4, 512, 1]" = var_mean_69[1];  var_mean_69 = None
        add_279: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_383, 1e-05);  getitem_383 = None
        rsqrt_69: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_279);  add_279 = None
        sub_71: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_278, getitem_384);  getitem_384 = None
        mul_414: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_69);  sub_71 = None
        mul_415: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_414, primals_418)
        add_280: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_415, primals_419);  mul_415 = primals_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_417: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_280, [-1, 1280]);  add_280 = None
        addmm_138: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_420, view_417, primals_421);  primals_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_418: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_138, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_416: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_418, 0.5)
        pow_35: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_418, 3.0)
        mul_417: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_35, 0.044715);  pow_35 = None
        add_281: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_418, mul_417);  view_418 = mul_417 = None
        mul_418: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_281, 0.7978845608028654);  add_281 = None
        tanh_34: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_418);  mul_418 = None
        add_282: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_34, 1.0);  tanh_34 = None
        mul_419: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_416, add_282);  mul_416 = add_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_419: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_419, [-1, 5120]);  mul_419 = None
        addmm_139: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_422, view_419, primals_423);  primals_422 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_420: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_139, [4, 512, 1280]);  addmm_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_70: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 70)
        inductor_random_default_2: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_70, 'rand');  inductor_lookup_seed_default_70 = None
        gt_70: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_2, 0.1);  inductor_random_default_2 = None
        mul_420: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_70, view_420);  view_420 = None
        mul_421: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_420, 1.1111111111111112);  mul_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_283: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_278, mul_421);  add_278 = mul_421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_70 = torch.ops.aten.var_mean.correction(add_283, [2], correction = 0, keepdim = True)
        getitem_385: "f32[4, 512, 1]" = var_mean_70[0]
        getitem_386: "f32[4, 512, 1]" = var_mean_70[1];  var_mean_70 = None
        add_284: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_385, 1e-05);  getitem_385 = None
        rsqrt_70: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_284);  add_284 = None
        sub_72: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_283, getitem_386);  getitem_386 = None
        mul_422: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_70);  sub_72 = None
        mul_423: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_422, primals_424)
        add_285: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_423, primals_425);  mul_423 = primals_425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_421: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_285, [-1, 1280]);  add_285 = None
        addmm_140: "f32[2048, 3840]" = torch.ops.aten.addmm.default(primals_426, view_421, primals_427);  primals_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_422: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_140, [4, 512, 3840]);  addmm_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_35 = torch.ops.aten.split.Tensor(view_422, 1280, 2);  view_422 = None
        getitem_387: "f32[4, 512, 1280]" = split_35[0]
        getitem_388: "f32[4, 512, 1280]" = split_35[1]
        getitem_389: "f32[4, 512, 1280]" = split_35[2];  split_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_423: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_388, [4, 512, -1, 64]);  getitem_388 = None
        permute_140: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_423, [0, 2, 1, 3]);  view_423 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_424: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_389, [4, 512, -1, 64]);  getitem_389 = None
        permute_141: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_424, [0, 2, 1, 3]);  view_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_425: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_387, [4, 512, -1, 64]);  getitem_387 = None
        permute_142: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_425, [0, 2, 1, 3]);  view_425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_35 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_142, permute_140, permute_141, expand_2, True, 0.1, scale = 0.125);  expand_2 = None
        getitem_390: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_35[0]
        getitem_391: "f32[4, 20, 512]" = _scaled_dot_product_efficient_attention_35[1]
        getitem_392: "i64[]" = _scaled_dot_product_efficient_attention_35[2]
        getitem_393: "i64[]" = _scaled_dot_product_efficient_attention_35[3];  _scaled_dot_product_efficient_attention_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_143: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_390, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_426: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_143, [4, 512, -1]);  permute_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_427: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_426, [-1, 1280]);  view_426 = None
        addmm_141: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_428, view_427, primals_429);  primals_428 = view_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_428: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_141, [4, 512, 1280]);  addmm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default_71: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 71)
        inductor_random_default_1: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_71, 'rand');  inductor_lookup_seed_default_71 = None
        gt_71: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.1);  inductor_random_default_1 = None
        mul_424: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_71, view_428);  view_428 = None
        mul_425: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_424, 1.1111111111111112);  mul_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_286: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_425, add_283);  mul_425 = add_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_71 = torch.ops.aten.var_mean.correction(add_286, [2], correction = 0, keepdim = True)
        getitem_394: "f32[4, 512, 1]" = var_mean_71[0]
        getitem_395: "f32[4, 512, 1]" = var_mean_71[1];  var_mean_71 = None
        add_287: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_394, 1e-05);  getitem_394 = None
        rsqrt_71: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_287);  add_287 = None
        sub_73: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_286, getitem_395);  getitem_395 = None
        mul_426: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_71);  sub_73 = None
        mul_427: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_426, primals_430)
        add_288: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_427, primals_431);  mul_427 = primals_431 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_429: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_288, [-1, 1280]);  add_288 = None
        addmm_142: "f32[2048, 5120]" = torch.ops.aten.addmm.default(primals_432, view_429, primals_433);  primals_432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_430: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_142, [4, 512, 5120])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_428: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_430, 0.5)
        pow_36: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_430, 3.0)
        mul_429: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_36, 0.044715);  pow_36 = None
        add_289: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_430, mul_429);  view_430 = mul_429 = None
        mul_430: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_289, 0.7978845608028654);  add_289 = None
        tanh_35: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_430);  mul_430 = None
        add_290: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_35, 1.0);  tanh_35 = None
        mul_431: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_428, add_290);  mul_428 = add_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_431: "f32[2048, 5120]" = torch.ops.aten.reshape.default(mul_431, [-1, 5120]);  mul_431 = None
        addmm_143: "f32[2048, 1280]" = torch.ops.aten.addmm.default(primals_434, view_431, primals_435);  primals_434 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_432: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_143, [4, 512, 1280]);  addmm_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_72: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 72);  inductor_seeds_default = None
        inductor_random_default: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default_72, 'rand');  inductor_lookup_seed_default_72 = None
        gt_72: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_432: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_72, view_432);  view_432 = None
        mul_433: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_432, 1.1111111111111112);  mul_432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_291: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_286, mul_433);  add_286 = mul_433 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:628 in forward, code: hidden_states = self.ln_f(hidden_states)
        var_mean_72 = torch.ops.aten.var_mean.correction(add_291, [2], correction = 0, keepdim = True)
        getitem_396: "f32[4, 512, 1]" = var_mean_72[0]
        getitem_397: "f32[4, 512, 1]" = var_mean_72[1];  var_mean_72 = None
        add_292: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_396, 1e-05);  getitem_396 = None
        rsqrt_72: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_292);  add_292 = None
        sub_74: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_291, getitem_397);  add_291 = getitem_397 = None
        mul_434: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_72);  sub_74 = None
        mul_435: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_434, primals_436)
        add_293: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_435, primals_437);  mul_435 = primals_437 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:706 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        permute_144: "f32[1280, 50257]" = torch.ops.aten.permute.default(primals_2, [1, 0])
        view_434: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_293, [2048, 1280])
        mm: "f32[2048, 50257]" = torch.ops.aten.mm.default(view_434, permute_144);  permute_144 = None
        view_435: "f32[4, 512, 50257]" = torch.ops.aten.reshape.default(mm, [4, 512, 50257]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:628 in forward, code: hidden_states = self.ln_f(hidden_states)
        div: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_72, 1280);  rsqrt_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_150: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_431, [1, 0]);  view_431 = None
        permute_152: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_429, [1, 0]);  view_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_1: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_71, 1280);  rsqrt_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_160: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_421, [1, 0]);  view_421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_2: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_70, 1280);  rsqrt_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_162: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_419, [1, 0]);  view_419 = None
        permute_164: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_417, [1, 0]);  view_417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_3: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_69, 1280);  rsqrt_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_172: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_409, [1, 0]);  view_409 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_4: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_68, 1280);  rsqrt_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_174: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_407, [1, 0]);  view_407 = None
        permute_176: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_405, [1, 0]);  view_405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_5: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_67, 1280);  rsqrt_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_184: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_397, [1, 0]);  view_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_6: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_66, 1280);  rsqrt_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_186: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_395, [1, 0]);  view_395 = None
        permute_188: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_393, [1, 0]);  view_393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_7: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_65, 1280);  rsqrt_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_196: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_385, [1, 0]);  view_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_8: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_64, 1280);  rsqrt_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_198: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_383, [1, 0]);  view_383 = None
        permute_200: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_381, [1, 0]);  view_381 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_9: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_63, 1280);  rsqrt_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_208: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_373, [1, 0]);  view_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_10: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_62, 1280);  rsqrt_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_210: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_371, [1, 0]);  view_371 = None
        permute_212: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_369, [1, 0]);  view_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_11: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_61, 1280);  rsqrt_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_220: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_361, [1, 0]);  view_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_12: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_60, 1280);  rsqrt_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_222: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_359, [1, 0]);  view_359 = None
        permute_224: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_357, [1, 0]);  view_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_13: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_59, 1280);  rsqrt_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_232: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_349, [1, 0]);  view_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_14: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_58, 1280);  rsqrt_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_234: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_347, [1, 0]);  view_347 = None
        permute_236: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_345, [1, 0]);  view_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_15: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_57, 1280);  rsqrt_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_244: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_337, [1, 0]);  view_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_16: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_56, 1280);  rsqrt_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_246: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_335, [1, 0]);  view_335 = None
        permute_248: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_333, [1, 0]);  view_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_17: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_55, 1280);  rsqrt_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_256: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_325, [1, 0]);  view_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_18: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_54, 1280);  rsqrt_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_258: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_323, [1, 0]);  view_323 = None
        permute_260: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_321, [1, 0]);  view_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_19: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_53, 1280);  rsqrt_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_268: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_313, [1, 0]);  view_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_20: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_52, 1280);  rsqrt_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_270: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_311, [1, 0]);  view_311 = None
        permute_272: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_309, [1, 0]);  view_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_21: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_51, 1280);  rsqrt_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_280: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_301, [1, 0]);  view_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_22: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_50, 1280);  rsqrt_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_282: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_299, [1, 0]);  view_299 = None
        permute_284: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_297, [1, 0]);  view_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_23: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_49, 1280);  rsqrt_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_292: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_289, [1, 0]);  view_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_24: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_48, 1280);  rsqrt_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_294: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_287, [1, 0]);  view_287 = None
        permute_296: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_285, [1, 0]);  view_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_25: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_47, 1280);  rsqrt_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_304: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_277, [1, 0]);  view_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_26: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 1280);  rsqrt_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_306: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_275, [1, 0]);  view_275 = None
        permute_308: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_273, [1, 0]);  view_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_27: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 1280);  rsqrt_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_316: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_265, [1, 0]);  view_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_28: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 1280);  rsqrt_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_318: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_263, [1, 0]);  view_263 = None
        permute_320: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_261, [1, 0]);  view_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_29: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 1280);  rsqrt_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_328: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_253, [1, 0]);  view_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_30: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_42, 1280);  rsqrt_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_330: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_251, [1, 0]);  view_251 = None
        permute_332: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_249, [1, 0]);  view_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_31: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_41, 1280);  rsqrt_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_340: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_241, [1, 0]);  view_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_32: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 1280);  rsqrt_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_342: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_239, [1, 0]);  view_239 = None
        permute_344: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_237, [1, 0]);  view_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_33: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 1280);  rsqrt_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_352: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_229, [1, 0]);  view_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_34: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 1280);  rsqrt_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_354: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_227, [1, 0]);  view_227 = None
        permute_356: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_225, [1, 0]);  view_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_35: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 1280);  rsqrt_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_364: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_217, [1, 0]);  view_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_36: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 1280);  rsqrt_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_366: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_215, [1, 0]);  view_215 = None
        permute_368: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_213, [1, 0]);  view_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_37: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 1280);  rsqrt_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_376: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_205, [1, 0]);  view_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_38: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 1280);  rsqrt_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_378: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_203, [1, 0]);  view_203 = None
        permute_380: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_201, [1, 0]);  view_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_39: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 1280);  rsqrt_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_388: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_193, [1, 0]);  view_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_40: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 1280);  rsqrt_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_390: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_191, [1, 0]);  view_191 = None
        permute_392: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_189, [1, 0]);  view_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_41: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 1280);  rsqrt_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_400: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_181, [1, 0]);  view_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_42: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 1280);  rsqrt_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_402: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_179, [1, 0]);  view_179 = None
        permute_404: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_177, [1, 0]);  view_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_43: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 1280);  rsqrt_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_412: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_169, [1, 0]);  view_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_44: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 1280);  rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_414: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_167, [1, 0]);  view_167 = None
        permute_416: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_165, [1, 0]);  view_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_45: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 1280);  rsqrt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_424: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_157, [1, 0]);  view_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_46: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 1280);  rsqrt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_426: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_155, [1, 0]);  view_155 = None
        permute_428: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_153, [1, 0]);  view_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_47: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 1280);  rsqrt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_436: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_145, [1, 0]);  view_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_48: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 1280);  rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_438: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_143, [1, 0]);  view_143 = None
        permute_440: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_141, [1, 0]);  view_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_49: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 1280);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_448: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_133, [1, 0]);  view_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_50: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 1280);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_450: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_131, [1, 0]);  view_131 = None
        permute_452: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_129, [1, 0]);  view_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_51: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 1280);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_460: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_121, [1, 0]);  view_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_52: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 1280);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_462: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_119, [1, 0]);  view_119 = None
        permute_464: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_117, [1, 0]);  view_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_53: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 1280);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_472: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_109, [1, 0]);  view_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_54: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 1280);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_474: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_107, [1, 0]);  view_107 = None
        permute_476: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_105, [1, 0]);  view_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_55: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 1280);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_484: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_97, [1, 0]);  view_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_56: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 1280);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_486: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_95, [1, 0]);  view_95 = None
        permute_488: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_93, [1, 0]);  view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_57: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 1280);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_496: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_85, [1, 0]);  view_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_58: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 1280);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_498: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_83, [1, 0]);  view_83 = None
        permute_500: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_81, [1, 0]);  view_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_59: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 1280);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_508: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_73, [1, 0]);  view_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_60: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 1280);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_510: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_71, [1, 0]);  view_71 = None
        permute_512: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_69, [1, 0]);  view_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_61: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 1280);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_520: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_61, [1, 0]);  view_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_62: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 1280);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_522: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_59, [1, 0]);  view_59 = None
        permute_524: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_57, [1, 0]);  view_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_63: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 1280);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_532: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_49, [1, 0]);  view_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_64: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 1280);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_534: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_47, [1, 0]);  view_47 = None
        permute_536: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_45, [1, 0]);  view_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_65: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 1280);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_544: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_37, [1, 0]);  view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_66: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 1280);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_546: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_35, [1, 0]);  view_35 = None
        permute_548: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_33, [1, 0]);  view_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_67: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 1280);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_556: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_25, [1, 0]);  view_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_68: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 1280);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_558: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_23, [1, 0]);  view_23 = None
        permute_560: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_21, [1, 0]);  view_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_69: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 1280);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_568: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_13, [1, 0]);  view_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_70: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 1280);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_570: "f32[5120, 2048]" = torch.ops.aten.permute.default(view_11, [1, 0]);  view_11 = None
        permute_572: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_9, [1, 0]);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_71: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 1280);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_580: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_1, [1, 0]);  view_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_72: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 1280);  rsqrt = None
        return (view_435, add_293, primals_1, primals_2, primals_4, primals_7, primals_9, primals_10, primals_13, primals_15, primals_16, primals_19, primals_21, primals_22, primals_25, primals_27, primals_28, primals_31, primals_33, primals_34, primals_37, primals_39, primals_40, primals_43, primals_45, primals_46, primals_49, primals_51, primals_52, primals_55, primals_57, primals_58, primals_61, primals_63, primals_64, primals_67, primals_69, primals_70, primals_73, primals_75, primals_76, primals_79, primals_81, primals_82, primals_85, primals_87, primals_88, primals_91, primals_93, primals_94, primals_97, primals_99, primals_100, primals_103, primals_105, primals_106, primals_109, primals_111, primals_112, primals_115, primals_117, primals_118, primals_121, primals_123, primals_124, primals_127, primals_129, primals_130, primals_133, primals_135, primals_136, primals_139, primals_141, primals_142, primals_145, primals_147, primals_148, primals_151, primals_153, primals_154, primals_157, primals_159, primals_160, primals_163, primals_165, primals_166, primals_169, primals_171, primals_172, primals_175, primals_177, primals_178, primals_181, primals_183, primals_184, primals_187, primals_189, primals_190, primals_193, primals_195, primals_196, primals_199, primals_201, primals_202, primals_205, primals_207, primals_208, primals_211, primals_213, primals_214, primals_217, primals_219, primals_220, primals_223, primals_225, primals_226, primals_229, primals_231, primals_232, primals_235, primals_237, primals_238, primals_241, primals_243, primals_244, primals_247, primals_249, primals_250, primals_253, primals_255, primals_256, primals_259, primals_261, primals_262, primals_265, primals_267, primals_268, primals_271, primals_273, primals_274, primals_277, primals_279, primals_280, primals_283, primals_285, primals_286, primals_289, primals_291, primals_292, primals_295, primals_297, primals_298, primals_301, primals_303, primals_304, primals_307, primals_309, primals_310, primals_313, primals_315, primals_316, primals_319, primals_321, primals_322, primals_325, primals_327, primals_328, primals_331, primals_333, primals_334, primals_337, primals_339, primals_340, primals_343, primals_345, primals_346, primals_349, primals_351, primals_352, primals_355, primals_357, primals_358, primals_361, primals_363, primals_364, primals_367, primals_369, primals_370, primals_373, primals_375, primals_376, primals_379, primals_381, primals_382, primals_385, primals_387, primals_388, primals_391, primals_393, primals_394, primals_397, primals_399, primals_400, primals_403, primals_405, primals_406, primals_409, primals_411, primals_412, primals_415, primals_417, primals_418, primals_421, primals_423, primals_424, primals_427, primals_429, primals_430, primals_433, primals_435, primals_436, unsqueeze, gt, mul_2, permute, permute_1, permute_2, where, getitem_5, getitem_6, getitem_7, getitem_8, gt_1, mul_6, addmm_2, gt_2, mul_14, permute_4, permute_5, permute_6, getitem_16, getitem_17, getitem_18, getitem_19, gt_3, mul_18, addmm_6, gt_4, mul_26, permute_8, permute_9, permute_10, getitem_27, getitem_28, getitem_29, getitem_30, gt_5, mul_30, addmm_10, gt_6, mul_38, permute_12, permute_13, permute_14, getitem_38, getitem_39, getitem_40, getitem_41, gt_7, mul_42, addmm_14, gt_8, mul_50, permute_16, permute_17, permute_18, getitem_49, getitem_50, getitem_51, getitem_52, gt_9, mul_54, addmm_18, gt_10, mul_62, permute_20, permute_21, permute_22, getitem_60, getitem_61, getitem_62, getitem_63, gt_11, mul_66, addmm_22, gt_12, mul_74, permute_24, permute_25, permute_26, getitem_71, getitem_72, getitem_73, getitem_74, gt_13, mul_78, addmm_26, gt_14, mul_86, permute_28, permute_29, permute_30, getitem_82, getitem_83, getitem_84, getitem_85, gt_15, mul_90, addmm_30, gt_16, mul_98, permute_32, permute_33, permute_34, getitem_93, getitem_94, getitem_95, getitem_96, gt_17, mul_102, addmm_34, gt_18, mul_110, permute_36, permute_37, permute_38, getitem_104, getitem_105, getitem_106, getitem_107, gt_19, mul_114, addmm_38, gt_20, mul_122, permute_40, permute_41, permute_42, getitem_115, getitem_116, getitem_117, getitem_118, gt_21, mul_126, addmm_42, gt_22, mul_134, permute_44, permute_45, permute_46, getitem_126, getitem_127, getitem_128, getitem_129, gt_23, mul_138, addmm_46, gt_24, mul_146, permute_48, permute_49, permute_50, getitem_137, getitem_138, getitem_139, getitem_140, gt_25, mul_150, addmm_50, gt_26, mul_158, permute_52, permute_53, permute_54, getitem_148, getitem_149, getitem_150, getitem_151, gt_27, mul_162, addmm_54, gt_28, mul_170, permute_56, permute_57, permute_58, getitem_159, getitem_160, getitem_161, getitem_162, gt_29, mul_174, addmm_58, gt_30, mul_182, permute_60, permute_61, permute_62, getitem_170, getitem_171, getitem_172, getitem_173, gt_31, mul_186, addmm_62, gt_32, mul_194, permute_64, permute_65, permute_66, getitem_181, getitem_182, getitem_183, getitem_184, gt_33, mul_198, addmm_66, gt_34, mul_206, permute_68, permute_69, permute_70, getitem_192, getitem_193, getitem_194, getitem_195, gt_35, mul_210, addmm_70, gt_36, mul_218, permute_72, permute_73, permute_74, getitem_203, getitem_204, getitem_205, getitem_206, gt_37, mul_222, addmm_74, gt_38, mul_230, permute_76, permute_77, permute_78, getitem_214, getitem_215, getitem_216, getitem_217, gt_39, mul_234, addmm_78, gt_40, mul_242, permute_80, permute_81, permute_82, getitem_225, getitem_226, getitem_227, getitem_228, gt_41, mul_246, addmm_82, gt_42, mul_254, permute_84, permute_85, permute_86, getitem_236, getitem_237, getitem_238, getitem_239, gt_43, mul_258, addmm_86, gt_44, mul_266, permute_88, permute_89, permute_90, getitem_247, getitem_248, getitem_249, getitem_250, gt_45, mul_270, addmm_90, gt_46, mul_278, permute_92, permute_93, permute_94, getitem_258, getitem_259, getitem_260, getitem_261, gt_47, mul_282, addmm_94, gt_48, mul_290, permute_96, permute_97, permute_98, getitem_269, getitem_270, getitem_271, getitem_272, gt_49, mul_294, addmm_98, gt_50, mul_302, permute_100, permute_101, permute_102, getitem_280, getitem_281, getitem_282, getitem_283, gt_51, mul_306, addmm_102, gt_52, mul_314, permute_104, permute_105, permute_106, getitem_291, getitem_292, getitem_293, getitem_294, gt_53, mul_318, addmm_106, gt_54, mul_326, permute_108, permute_109, permute_110, getitem_302, getitem_303, getitem_304, getitem_305, gt_55, mul_330, addmm_110, gt_56, mul_338, permute_112, permute_113, permute_114, getitem_313, getitem_314, getitem_315, getitem_316, gt_57, mul_342, addmm_114, gt_58, mul_350, permute_116, permute_117, permute_118, getitem_324, getitem_325, getitem_326, getitem_327, gt_59, mul_354, addmm_118, gt_60, mul_362, permute_120, permute_121, permute_122, getitem_335, getitem_336, getitem_337, getitem_338, gt_61, mul_366, addmm_122, gt_62, mul_374, permute_124, permute_125, permute_126, getitem_346, getitem_347, getitem_348, getitem_349, gt_63, mul_378, addmm_126, gt_64, mul_386, permute_128, permute_129, permute_130, getitem_357, getitem_358, getitem_359, getitem_360, gt_65, mul_390, addmm_130, gt_66, mul_398, permute_132, permute_133, permute_134, getitem_368, getitem_369, getitem_370, getitem_371, gt_67, mul_402, addmm_134, gt_68, mul_410, permute_136, permute_137, permute_138, getitem_379, getitem_380, getitem_381, getitem_382, gt_69, mul_414, addmm_138, gt_70, mul_422, permute_140, permute_141, permute_142, getitem_390, getitem_391, getitem_392, getitem_393, gt_71, mul_426, addmm_142, gt_72, mul_434, view_434, div, permute_150, permute_152, div_1, permute_160, div_2, permute_162, permute_164, div_3, permute_172, div_4, permute_174, permute_176, div_5, permute_184, div_6, permute_186, permute_188, div_7, permute_196, div_8, permute_198, permute_200, div_9, permute_208, div_10, permute_210, permute_212, div_11, permute_220, div_12, permute_222, permute_224, div_13, permute_232, div_14, permute_234, permute_236, div_15, permute_244, div_16, permute_246, permute_248, div_17, permute_256, div_18, permute_258, permute_260, div_19, permute_268, div_20, permute_270, permute_272, div_21, permute_280, div_22, permute_282, permute_284, div_23, permute_292, div_24, permute_294, permute_296, div_25, permute_304, div_26, permute_306, permute_308, div_27, permute_316, div_28, permute_318, permute_320, div_29, permute_328, div_30, permute_330, permute_332, div_31, permute_340, div_32, permute_342, permute_344, div_33, permute_352, div_34, permute_354, permute_356, div_35, permute_364, div_36, permute_366, permute_368, div_37, permute_376, div_38, permute_378, permute_380, div_39, permute_388, div_40, permute_390, permute_392, div_41, permute_400, div_42, permute_402, permute_404, div_43, permute_412, div_44, permute_414, permute_416, div_45, permute_424, div_46, permute_426, permute_428, div_47, permute_436, div_48, permute_438, permute_440, div_49, permute_448, div_50, permute_450, permute_452, div_51, permute_460, div_52, permute_462, permute_464, div_53, permute_472, div_54, permute_474, permute_476, div_55, permute_484, div_56, permute_486, permute_488, div_57, permute_496, div_58, permute_498, permute_500, div_59, permute_508, div_60, permute_510, permute_512, div_61, permute_520, div_62, permute_522, permute_524, div_63, permute_532, div_64, permute_534, permute_536, div_65, permute_544, div_66, permute_546, permute_548, div_67, permute_556, div_68, permute_558, permute_560, div_69, permute_568, div_70, permute_570, permute_572, div_71, permute_580, div_72)
