class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[32, 128]", primals_2: "f32[50257, 2048]", primals_3: "f32[2048, 2048]", primals_4: "f32[2048]", primals_5: "f32[2048]", primals_6: "f32[2048, 2048]", primals_7: "f32[2048, 2048]", primals_8: "f32[2048, 2048]", primals_9: "b8[1, 1, 2048, 2048]", primals_10: "f32[2048, 2048]", primals_11: "f32[2048]", primals_12: "f32[2048]", primals_13: "f32[2048]", primals_14: "f32[8192, 2048]", primals_15: "f32[8192]", primals_16: "f32[2048, 8192]", primals_17: "f32[2048]", primals_18: "f32[2048]", primals_19: "f32[2048]", primals_20: "f32[2048, 2048]", primals_21: "f32[2048, 2048]", primals_22: "f32[2048, 2048]", primals_23: "b8[1, 1, 2048, 2048]", primals_24: "f32[2048, 2048]", primals_25: "f32[2048]", primals_26: "f32[2048]", primals_27: "f32[2048]", primals_28: "f32[8192, 2048]", primals_29: "f32[8192]", primals_30: "f32[2048, 8192]", primals_31: "f32[2048]", primals_32: "f32[2048]", primals_33: "f32[2048]", primals_34: "f32[2048, 2048]", primals_35: "f32[2048, 2048]", primals_36: "f32[2048, 2048]", primals_37: "b8[1, 1, 2048, 2048]", primals_38: "f32[2048, 2048]", primals_39: "f32[2048]", primals_40: "f32[2048]", primals_41: "f32[2048]", primals_42: "f32[8192, 2048]", primals_43: "f32[8192]", primals_44: "f32[2048, 8192]", primals_45: "f32[2048]", primals_46: "f32[2048]", primals_47: "f32[2048]", primals_48: "f32[2048, 2048]", primals_49: "f32[2048, 2048]", primals_50: "f32[2048, 2048]", primals_51: "b8[1, 1, 2048, 2048]", primals_52: "f32[2048, 2048]", primals_53: "f32[2048]", primals_54: "f32[2048]", primals_55: "f32[2048]", primals_56: "f32[8192, 2048]", primals_57: "f32[8192]", primals_58: "f32[2048, 8192]", primals_59: "f32[2048]", primals_60: "f32[2048]", primals_61: "f32[2048]", primals_62: "f32[2048, 2048]", primals_63: "f32[2048, 2048]", primals_64: "f32[2048, 2048]", primals_65: "b8[1, 1, 2048, 2048]", primals_66: "f32[2048, 2048]", primals_67: "f32[2048]", primals_68: "f32[2048]", primals_69: "f32[2048]", primals_70: "f32[8192, 2048]", primals_71: "f32[8192]", primals_72: "f32[2048, 8192]", primals_73: "f32[2048]", primals_74: "f32[2048]", primals_75: "f32[2048]", primals_76: "f32[2048, 2048]", primals_77: "f32[2048, 2048]", primals_78: "f32[2048, 2048]", primals_79: "b8[1, 1, 2048, 2048]", primals_80: "f32[2048, 2048]", primals_81: "f32[2048]", primals_82: "f32[2048]", primals_83: "f32[2048]", primals_84: "f32[8192, 2048]", primals_85: "f32[8192]", primals_86: "f32[2048, 8192]", primals_87: "f32[2048]", primals_88: "f32[2048]", primals_89: "f32[2048]", primals_90: "f32[2048, 2048]", primals_91: "f32[2048, 2048]", primals_92: "f32[2048, 2048]", primals_93: "b8[1, 1, 2048, 2048]", primals_94: "f32[2048, 2048]", primals_95: "f32[2048]", primals_96: "f32[2048]", primals_97: "f32[2048]", primals_98: "f32[8192, 2048]", primals_99: "f32[8192]", primals_100: "f32[2048, 8192]", primals_101: "f32[2048]", primals_102: "f32[2048]", primals_103: "f32[2048]", primals_104: "f32[2048, 2048]", primals_105: "f32[2048, 2048]", primals_106: "f32[2048, 2048]", primals_107: "b8[1, 1, 2048, 2048]", primals_108: "f32[2048, 2048]", primals_109: "f32[2048]", primals_110: "f32[2048]", primals_111: "f32[2048]", primals_112: "f32[8192, 2048]", primals_113: "f32[8192]", primals_114: "f32[2048, 8192]", primals_115: "f32[2048]", primals_116: "f32[2048]", primals_117: "f32[2048]", primals_118: "f32[2048, 2048]", primals_119: "f32[2048, 2048]", primals_120: "f32[2048, 2048]", primals_121: "b8[1, 1, 2048, 2048]", primals_122: "f32[2048, 2048]", primals_123: "f32[2048]", primals_124: "f32[2048]", primals_125: "f32[2048]", primals_126: "f32[8192, 2048]", primals_127: "f32[8192]", primals_128: "f32[2048, 8192]", primals_129: "f32[2048]", primals_130: "f32[2048]", primals_131: "f32[2048]", primals_132: "f32[2048, 2048]", primals_133: "f32[2048, 2048]", primals_134: "f32[2048, 2048]", primals_135: "b8[1, 1, 2048, 2048]", primals_136: "f32[2048, 2048]", primals_137: "f32[2048]", primals_138: "f32[2048]", primals_139: "f32[2048]", primals_140: "f32[8192, 2048]", primals_141: "f32[8192]", primals_142: "f32[2048, 8192]", primals_143: "f32[2048]", primals_144: "f32[2048]", primals_145: "f32[2048]", primals_146: "f32[2048, 2048]", primals_147: "f32[2048, 2048]", primals_148: "f32[2048, 2048]", primals_149: "b8[1, 1, 2048, 2048]", primals_150: "f32[2048, 2048]", primals_151: "f32[2048]", primals_152: "f32[2048]", primals_153: "f32[2048]", primals_154: "f32[8192, 2048]", primals_155: "f32[8192]", primals_156: "f32[2048, 8192]", primals_157: "f32[2048]", primals_158: "f32[2048]", primals_159: "f32[2048]", primals_160: "f32[2048, 2048]", primals_161: "f32[2048, 2048]", primals_162: "f32[2048, 2048]", primals_163: "b8[1, 1, 2048, 2048]", primals_164: "f32[2048, 2048]", primals_165: "f32[2048]", primals_166: "f32[2048]", primals_167: "f32[2048]", primals_168: "f32[8192, 2048]", primals_169: "f32[8192]", primals_170: "f32[2048, 8192]", primals_171: "f32[2048]", primals_172: "f32[2048]", primals_173: "f32[2048]", primals_174: "f32[2048, 2048]", primals_175: "f32[2048, 2048]", primals_176: "f32[2048, 2048]", primals_177: "b8[1, 1, 2048, 2048]", primals_178: "f32[2048, 2048]", primals_179: "f32[2048]", primals_180: "f32[2048]", primals_181: "f32[2048]", primals_182: "f32[8192, 2048]", primals_183: "f32[8192]", primals_184: "f32[2048, 8192]", primals_185: "f32[2048]", primals_186: "f32[2048]", primals_187: "f32[2048]", primals_188: "f32[2048, 2048]", primals_189: "f32[2048, 2048]", primals_190: "f32[2048, 2048]", primals_191: "b8[1, 1, 2048, 2048]", primals_192: "f32[2048, 2048]", primals_193: "f32[2048]", primals_194: "f32[2048]", primals_195: "f32[2048]", primals_196: "f32[8192, 2048]", primals_197: "f32[8192]", primals_198: "f32[2048, 8192]", primals_199: "f32[2048]", primals_200: "f32[2048]", primals_201: "f32[2048]", primals_202: "f32[2048, 2048]", primals_203: "f32[2048, 2048]", primals_204: "f32[2048, 2048]", primals_205: "b8[1, 1, 2048, 2048]", primals_206: "f32[2048, 2048]", primals_207: "f32[2048]", primals_208: "f32[2048]", primals_209: "f32[2048]", primals_210: "f32[8192, 2048]", primals_211: "f32[8192]", primals_212: "f32[2048, 8192]", primals_213: "f32[2048]", primals_214: "f32[2048]", primals_215: "f32[2048]", primals_216: "f32[2048, 2048]", primals_217: "f32[2048, 2048]", primals_218: "f32[2048, 2048]", primals_219: "b8[1, 1, 2048, 2048]", primals_220: "f32[2048, 2048]", primals_221: "f32[2048]", primals_222: "f32[2048]", primals_223: "f32[2048]", primals_224: "f32[8192, 2048]", primals_225: "f32[8192]", primals_226: "f32[2048, 8192]", primals_227: "f32[2048]", primals_228: "f32[2048]", primals_229: "f32[2048]", primals_230: "f32[2048, 2048]", primals_231: "f32[2048, 2048]", primals_232: "f32[2048, 2048]", primals_233: "b8[1, 1, 2048, 2048]", primals_234: "f32[2048, 2048]", primals_235: "f32[2048]", primals_236: "f32[2048]", primals_237: "f32[2048]", primals_238: "f32[8192, 2048]", primals_239: "f32[8192]", primals_240: "f32[2048, 8192]", primals_241: "f32[2048]", primals_242: "f32[2048]", primals_243: "f32[2048]", primals_244: "f32[2048, 2048]", primals_245: "f32[2048, 2048]", primals_246: "f32[2048, 2048]", primals_247: "b8[1, 1, 2048, 2048]", primals_248: "f32[2048, 2048]", primals_249: "f32[2048]", primals_250: "f32[2048]", primals_251: "f32[2048]", primals_252: "f32[8192, 2048]", primals_253: "f32[8192]", primals_254: "f32[2048, 8192]", primals_255: "f32[2048]", primals_256: "f32[2048]", primals_257: "f32[2048]", primals_258: "f32[2048, 2048]", primals_259: "f32[2048, 2048]", primals_260: "f32[2048, 2048]", primals_261: "b8[1, 1, 2048, 2048]", primals_262: "f32[2048, 2048]", primals_263: "f32[2048]", primals_264: "f32[2048]", primals_265: "f32[2048]", primals_266: "f32[8192, 2048]", primals_267: "f32[8192]", primals_268: "f32[2048, 8192]", primals_269: "f32[2048]", primals_270: "f32[2048]", primals_271: "f32[2048]", primals_272: "f32[2048, 2048]", primals_273: "f32[2048, 2048]", primals_274: "f32[2048, 2048]", primals_275: "b8[1, 1, 2048, 2048]", primals_276: "f32[2048, 2048]", primals_277: "f32[2048]", primals_278: "f32[2048]", primals_279: "f32[2048]", primals_280: "f32[8192, 2048]", primals_281: "f32[8192]", primals_282: "f32[2048, 8192]", primals_283: "f32[2048]", primals_284: "f32[2048]", primals_285: "f32[2048]", primals_286: "f32[2048, 2048]", primals_287: "f32[2048, 2048]", primals_288: "f32[2048, 2048]", primals_289: "b8[1, 1, 2048, 2048]", primals_290: "f32[2048, 2048]", primals_291: "f32[2048]", primals_292: "f32[2048]", primals_293: "f32[2048]", primals_294: "f32[8192, 2048]", primals_295: "f32[8192]", primals_296: "f32[2048, 8192]", primals_297: "f32[2048]", primals_298: "f32[2048]", primals_299: "f32[2048]", primals_300: "f32[2048, 2048]", primals_301: "f32[2048, 2048]", primals_302: "f32[2048, 2048]", primals_303: "b8[1, 1, 2048, 2048]", primals_304: "f32[2048, 2048]", primals_305: "f32[2048]", primals_306: "f32[2048]", primals_307: "f32[2048]", primals_308: "f32[8192, 2048]", primals_309: "f32[8192]", primals_310: "f32[2048, 8192]", primals_311: "f32[2048]", primals_312: "f32[2048]", primals_313: "f32[2048]", primals_314: "f32[2048, 2048]", primals_315: "f32[2048, 2048]", primals_316: "f32[2048, 2048]", primals_317: "b8[1, 1, 2048, 2048]", primals_318: "f32[2048, 2048]", primals_319: "f32[2048]", primals_320: "f32[2048]", primals_321: "f32[2048]", primals_322: "f32[8192, 2048]", primals_323: "f32[8192]", primals_324: "f32[2048, 8192]", primals_325: "f32[2048]", primals_326: "f32[2048]", primals_327: "f32[2048]", primals_328: "f32[2048, 2048]", primals_329: "f32[2048, 2048]", primals_330: "f32[2048, 2048]", primals_331: "b8[1, 1, 2048, 2048]", primals_332: "f32[2048, 2048]", primals_333: "f32[2048]", primals_334: "f32[2048]", primals_335: "f32[2048]", primals_336: "f32[8192, 2048]", primals_337: "f32[8192]", primals_338: "f32[2048, 8192]", primals_339: "f32[2048]", primals_340: "f32[2048]", primals_341: "f32[2048]", primals_342: "i64[32, 128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:444 in forward, code: inputs_embeds = self.wte(input_ids)
        embedding: "f32[32, 128, 2048]" = torch.ops.aten.embedding.default(primals_2, primals_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:451 in forward, code: position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[128]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:452 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:875 in _preprocess_mask_arguments, code: position_ids = position_ids.expand(batch_size, -1)
        expand: "i64[32, 128]" = torch.ops.aten.expand.default(unsqueeze, [32, -1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:765 in find_packed_sequence_indices, code: first_dummy_value = position_ids[:, :1] - 1  # We just need the diff on this first value to be 1
        slice_1: "i64[32, 1]" = torch.ops.aten.slice.Tensor(expand, 1, 0, 1)
        sub: "i64[32, 1]" = torch.ops.aten.sub.Tensor(slice_1, 1);  slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:766 in find_packed_sequence_indices, code: position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
        cat: "i64[32, 129]" = torch.ops.aten.cat.default([sub, expand], -1);  sub = expand = None
        slice_2: "i64[32, 128]" = torch.ops.aten.slice.Tensor(cat, -1, 0, 128)
        slice_3: "i64[32, 128]" = torch.ops.aten.slice.Tensor(cat, -1, 1, 129);  cat = None
        sub_1: "i64[32, 128]" = torch.ops.aten.sub.Tensor(slice_3, slice_2);  slice_3 = slice_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:767 in find_packed_sequence_indices, code: packed_sequence_mask = (position_diff != 1).cumsum(-1)
        ne: "b8[32, 128]" = torch.ops.aten.ne.Scalar(sub_1, 1);  sub_1 = None
        cumsum: "i64[32, 128]" = torch.ops.aten.cumsum.default(ne, -1);  ne = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:509 in sdpa_mask, code: batch_arange = torch.arange(batch_size, device=device)
        iota_1: "i64[32]" = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:360 in _non_vmap_expansion_sdpa, code: batch_indices = batch_indices[:, None, None, None]
        unsqueeze_1: "i64[32, 1]" = torch.ops.aten.unsqueeze.default(iota_1, 1);  iota_1 = None
        unsqueeze_2: "i64[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None
        unsqueeze_3: "i64[32, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 3);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_5: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1)
        unsqueeze_6: "i64[1, 1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 3)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_9: "i64[1, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 2);  unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:52 in and_mask, code: result = q_idx.new_ones((), dtype=torch.bool)
        full_default: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 128, 128]" = torch.ops.aten.le.Tensor(unsqueeze_9, unsqueeze_6)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and: "b8[1, 1, 128, 128]" = torch.ops.aten.bitwise_and.Tensor(full_default, le);  full_default = le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:168 in inner_mask, code: return packed_sequence_mask[batch_idx, q_idx] == packed_sequence_mask[batch_idx, kv_idx]
        index: "i64[32, 1, 128, 1]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_6]);  unsqueeze_6 = None
        index_1: "i64[32, 1, 1, 128]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_9]);  cumsum = unsqueeze_3 = unsqueeze_9 = None
        eq: "b8[32, 1, 128, 128]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and_1: "b8[32, 1, 128, 128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, eq);  bitwise_and = eq = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_1: "b8[32, 1, 128, 128]" = torch.ops.aten.expand.default(bitwise_and_1, [32, -1, 128, 128]);  bitwise_and_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_1, full_default_1, full_default_2);  expand_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:462 in forward, code: position_embeds = self.wpe(position_ids)
        embedding_1: "f32[1, 128, 2048]" = torch.ops.aten.embedding.default(primals_3, unsqueeze);  primals_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:463 in forward, code: hidden_states = inputs_embeds + position_embeds
        add_3: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(embedding, embedding_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 128, 1]" = var_mean[0]
        getitem_1: "f32[32, 128, 1]" = var_mean[1];  var_mean = None
        add_4: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_2: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_3, getitem_1)
        mul: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = None
        mul_1: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul, primals_4);  mul = None
        add_5: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_1, primals_5);  mul_1 = primals_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_6, [1, 0])
        view: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_5, [4096, 2048]);  add_5 = None
        mm: "f32[4096, 2048]" = torch.ops.aten.mm.default(view, permute);  permute = None
        view_1: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm, [32, 128, 2048]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_1: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_7, [1, 0])
        mm_1: "f32[4096, 2048]" = torch.ops.aten.mm.default(view, permute_1);  permute_1 = None
        view_3: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_1, [32, 128, 2048]);  mm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_2: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_8, [1, 0])
        mm_2: "f32[4096, 2048]" = torch.ops.aten.mm.default(view, permute_2);  permute_2 = None
        view_5: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_2, [32, 128, 2048]);  mm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_6: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_1, [32, 128, 16, 128]);  view_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_3: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_7: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_3, [32, 128, 16, 128]);  view_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_4: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_8: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_5, [32, 128, 16, 128]);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_5: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_6: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(permute_4, [0, 1, 3, 2]);  permute_4 = None
        expand_2: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_3, [32, 16, 128, 128]);  permute_3 = None
        clone_1: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_9: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_1, [512, 128, 128]);  clone_1 = None
        expand_3: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_6, [32, 16, 128, 128]);  permute_6 = None
        clone_2: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_10: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_2, [512, 128, 128]);  clone_2 = None
        bmm: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_9, view_10)
        view_11: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm, [32, 16, 128, 128]);  bmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_4: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_9, 2, 0, 128)
        slice_5: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_4, 3, 0, 128);  slice_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_1: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_5, view_11, full_default_2);  slice_5 = view_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_6: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_1, where);  where_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_6, [-1], True)
        sub_3: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_6, amax);  add_6 = amax = None
        exp: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        sum_1: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_4: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div, [32, 16, 128, 128])
        view_12: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_4, [512, 128, 128]);  expand_4 = None
        expand_5: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_5, [32, 16, 128, 128]);  permute_5 = None
        clone_4: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_13: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_4, [512, 128, 128]);  clone_4 = None
        bmm_1: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_12, view_13);  view_12 = None
        view_14: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_1, [32, 16, 128, 128]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_7: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        clone_5: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_15: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_5, [32, 128, 2048]);  clone_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_16: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_15, [4096, 2048]);  view_15 = None
        permute_8: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_10, [1, 0])
        addmm: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_11, view_16, permute_8);  primals_11 = permute_8 = None
        view_17: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm, [32, 128, 2048]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_7: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_17, add_3);  view_17 = add_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
        getitem_2: "f32[32, 128, 1]" = var_mean_1[0]
        getitem_3: "f32[32, 128, 1]" = var_mean_1[1];  var_mean_1 = None
        add_8: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_4: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_7, getitem_3);  getitem_3 = None
        mul_2: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_1);  sub_4 = None
        mul_3: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_2, primals_12)
        add_9: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_3, primals_13);  mul_3 = primals_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_18: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_9, [4096, 2048]);  add_9 = None
        permute_9: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_14, [1, 0])
        addmm_1: "f32[4096, 8192]" = torch.ops.aten.addmm.default(primals_15, view_18, permute_9);  primals_15 = permute_9 = None
        view_19: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_1, [32, 128, 8192])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_4: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_19, 0.5)
        pow_1: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_19, 3.0)
        mul_5: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_10: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_19, mul_5);  view_19 = mul_5 = None
        mul_6: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_10, 0.7978845608028654);  add_10 = None
        tanh: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
        add_11: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_7: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_4, add_11);  mul_4 = add_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_20: "f32[4096, 8192]" = torch.ops.aten.reshape.default(mul_7, [4096, 8192]);  mul_7 = None
        permute_10: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_16, [1, 0])
        addmm_2: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_17, view_20, permute_10);  primals_17 = permute_10 = None
        view_21: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_2, [32, 128, 2048]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_12: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_7, view_21);  add_7 = view_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_12, [2], correction = 0, keepdim = True)
        getitem_4: "f32[32, 128, 1]" = var_mean_2[0]
        getitem_5: "f32[32, 128, 1]" = var_mean_2[1];  var_mean_2 = None
        add_13: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_5: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_12, getitem_5);  getitem_5 = None
        mul_8: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_2);  sub_5 = None
        mul_9: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_8, primals_18)
        add_14: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_9, primals_19);  mul_9 = primals_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_11: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_20, [1, 0])
        view_22: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_14, [4096, 2048]);  add_14 = None
        mm_3: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_22, permute_11);  permute_11 = None
        view_23: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_3, [32, 128, 2048]);  mm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_12: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_21, [1, 0])
        mm_4: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_22, permute_12);  permute_12 = None
        view_25: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_4, [32, 128, 2048]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_13: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_22, [1, 0])
        mm_5: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_22, permute_13);  permute_13 = None
        view_27: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_5, [32, 128, 2048]);  mm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_28: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_23, [32, 128, 16, 128]);  view_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_14: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_29: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_25, [32, 128, 16, 128]);  view_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_15: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_30: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_27, [32, 128, 16, 128]);  view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_16: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_17: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(permute_15, [0, 1, 3, 2]);  permute_15 = None
        expand_6: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_14, [32, 16, 128, 128]);  permute_14 = None
        clone_8: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_31: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_8, [512, 128, 128]);  clone_8 = None
        expand_7: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_17, [32, 16, 128, 128]);  permute_17 = None
        clone_9: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_32: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_9, [512, 128, 128]);  clone_9 = None
        bmm_2: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_31, view_32)
        view_33: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_2, [32, 16, 128, 128]);  bmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_6: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_23, 2, 0, 128)
        slice_7: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_6, 3, 0, 128);  slice_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_2: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_7, view_33, full_default_2);  slice_7 = view_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_15: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_2, where);  where_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_1: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_15, [-1], True)
        sub_6: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_15, amax_1);  add_15 = amax_1 = None
        exp_1: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_2: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_8: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_1, [32, 16, 128, 128])
        view_34: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_8, [512, 128, 128]);  expand_8 = None
        expand_9: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_16, [32, 16, 128, 128]);  permute_16 = None
        clone_11: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_35: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_11, [512, 128, 128]);  clone_11 = None
        bmm_3: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_34, view_35);  view_34 = None
        view_36: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_3, [32, 16, 128, 128]);  bmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_18: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
        clone_12: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_37: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_12, [32, 128, 2048]);  clone_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_38: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_37, [4096, 2048]);  view_37 = None
        permute_19: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_24, [1, 0])
        addmm_3: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_25, view_38, permute_19);  primals_25 = permute_19 = None
        view_39: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_3, [32, 128, 2048]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_16: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_39, add_12);  view_39 = add_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_3 = torch.ops.aten.var_mean.correction(add_16, [2], correction = 0, keepdim = True)
        getitem_6: "f32[32, 128, 1]" = var_mean_3[0]
        getitem_7: "f32[32, 128, 1]" = var_mean_3[1];  var_mean_3 = None
        add_17: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        sub_7: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_16, getitem_7);  getitem_7 = None
        mul_10: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_3);  sub_7 = None
        mul_11: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_10, primals_26)
        add_18: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_11, primals_27);  mul_11 = primals_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_40: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_18, [4096, 2048]);  add_18 = None
        permute_20: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_28, [1, 0])
        addmm_4: "f32[4096, 8192]" = torch.ops.aten.addmm.default(primals_29, view_40, permute_20);  primals_29 = permute_20 = None
        view_41: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_4, [32, 128, 8192])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_12: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_41, 0.5)
        pow_2: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_41, 3.0)
        mul_13: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_19: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_41, mul_13);  view_41 = mul_13 = None
        mul_14: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_19, 0.7978845608028654);  add_19 = None
        tanh_1: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
        add_20: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_15: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_12, add_20);  mul_12 = add_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_42: "f32[4096, 8192]" = torch.ops.aten.reshape.default(mul_15, [4096, 8192]);  mul_15 = None
        permute_21: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_30, [1, 0])
        addmm_5: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_31, view_42, permute_21);  primals_31 = permute_21 = None
        view_43: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_5, [32, 128, 2048]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_21: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_16, view_43);  add_16 = view_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
        getitem_8: "f32[32, 128, 1]" = var_mean_4[0]
        getitem_9: "f32[32, 128, 1]" = var_mean_4[1];  var_mean_4 = None
        add_22: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_4: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        sub_8: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_21, getitem_9);  getitem_9 = None
        mul_16: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_4);  sub_8 = None
        mul_17: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_16, primals_32)
        add_23: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_17, primals_33);  mul_17 = primals_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_22: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_34, [1, 0])
        view_44: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_23, [4096, 2048]);  add_23 = None
        mm_6: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_44, permute_22);  permute_22 = None
        view_45: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_6, [32, 128, 2048]);  mm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_23: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_35, [1, 0])
        mm_7: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_44, permute_23);  permute_23 = None
        view_47: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_7, [32, 128, 2048]);  mm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_24: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_36, [1, 0])
        mm_8: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_44, permute_24);  permute_24 = None
        view_49: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_8, [32, 128, 2048]);  mm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_50: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_45, [32, 128, 16, 128]);  view_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_25: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_51: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_47, [32, 128, 16, 128]);  view_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_26: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_52: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_49, [32, 128, 16, 128]);  view_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_27: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_28: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(permute_26, [0, 1, 3, 2]);  permute_26 = None
        expand_10: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_25, [32, 16, 128, 128]);  permute_25 = None
        clone_15: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_53: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_15, [512, 128, 128]);  clone_15 = None
        expand_11: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_28, [32, 16, 128, 128]);  permute_28 = None
        clone_16: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_54: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_16, [512, 128, 128]);  clone_16 = None
        bmm_4: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_53, view_54)
        view_55: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_4, [32, 16, 128, 128]);  bmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_8: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_37, 2, 0, 128)
        slice_9: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_8, 3, 0, 128);  slice_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_3: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_9, view_55, full_default_2);  slice_9 = view_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_24: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_3, where);  where_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_2: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_24, [-1], True)
        sub_9: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_24, amax_2);  add_24 = amax_2 = None
        exp_2: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        sum_3: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_12: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_2, [32, 16, 128, 128])
        view_56: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_12, [512, 128, 128]);  expand_12 = None
        expand_13: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_27, [32, 16, 128, 128]);  permute_27 = None
        clone_18: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_57: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_18, [512, 128, 128]);  clone_18 = None
        bmm_5: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_56, view_57);  view_56 = None
        view_58: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_5, [32, 16, 128, 128]);  bmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_29: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        clone_19: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_59: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_19, [32, 128, 2048]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_60: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_59, [4096, 2048]);  view_59 = None
        permute_30: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_38, [1, 0])
        addmm_6: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_39, view_60, permute_30);  primals_39 = permute_30 = None
        view_61: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_6, [32, 128, 2048]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_25: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_61, add_21);  view_61 = add_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_5 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_10: "f32[32, 128, 1]" = var_mean_5[0]
        getitem_11: "f32[32, 128, 1]" = var_mean_5[1];  var_mean_5 = None
        add_26: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_5: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_10: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_25, getitem_11);  getitem_11 = None
        mul_18: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_5);  sub_10 = None
        mul_19: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_18, primals_40)
        add_27: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_19, primals_41);  mul_19 = primals_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_62: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_27, [4096, 2048]);  add_27 = None
        permute_31: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_42, [1, 0])
        addmm_7: "f32[4096, 8192]" = torch.ops.aten.addmm.default(primals_43, view_62, permute_31);  primals_43 = permute_31 = None
        view_63: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_7, [32, 128, 8192])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_20: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_63, 0.5)
        pow_3: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_63, 3.0)
        mul_21: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_28: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_63, mul_21);  view_63 = mul_21 = None
        mul_22: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_28, 0.7978845608028654);  add_28 = None
        tanh_2: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
        add_29: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_23: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_20, add_29);  mul_20 = add_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_64: "f32[4096, 8192]" = torch.ops.aten.reshape.default(mul_23, [4096, 8192]);  mul_23 = None
        permute_32: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_44, [1, 0])
        addmm_8: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_45, view_64, permute_32);  primals_45 = permute_32 = None
        view_65: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_8, [32, 128, 2048]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_30: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_25, view_65);  add_25 = view_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_6 = torch.ops.aten.var_mean.correction(add_30, [2], correction = 0, keepdim = True)
        getitem_12: "f32[32, 128, 1]" = var_mean_6[0]
        getitem_13: "f32[32, 128, 1]" = var_mean_6[1];  var_mean_6 = None
        add_31: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_6: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        sub_11: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_30, getitem_13);  getitem_13 = None
        mul_24: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_6);  sub_11 = None
        mul_25: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_24, primals_46)
        add_32: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_25, primals_47);  mul_25 = primals_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_33: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_48, [1, 0])
        view_66: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_32, [4096, 2048]);  add_32 = None
        mm_9: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_66, permute_33);  permute_33 = None
        view_67: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_9, [32, 128, 2048]);  mm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_34: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_49, [1, 0])
        mm_10: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_66, permute_34);  permute_34 = None
        view_69: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_10, [32, 128, 2048]);  mm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_35: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_50, [1, 0])
        mm_11: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_66, permute_35);  permute_35 = None
        view_71: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_11, [32, 128, 2048]);  mm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_72: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_67, [32, 128, 16, 128]);  view_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_36: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_73: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_69, [32, 128, 16, 128]);  view_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_37: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_74: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_71, [32, 128, 16, 128]);  view_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_38: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_39: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(permute_37, [0, 1, 3, 2]);  permute_37 = None
        expand_14: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_36, [32, 16, 128, 128]);  permute_36 = None
        clone_22: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_75: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_22, [512, 128, 128]);  clone_22 = None
        expand_15: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_39, [32, 16, 128, 128]);  permute_39 = None
        clone_23: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_76: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_23, [512, 128, 128]);  clone_23 = None
        bmm_6: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_75, view_76)
        view_77: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_6, [32, 16, 128, 128]);  bmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_10: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_51, 2, 0, 128)
        slice_11: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_10, 3, 0, 128);  slice_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_4: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_11, view_77, full_default_2);  slice_11 = view_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_33: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_4, where);  where_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_3: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_33, [-1], True)
        sub_12: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_33, amax_3);  add_33 = amax_3 = None
        exp_3: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_4: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_16: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_3, [32, 16, 128, 128])
        view_78: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_16, [512, 128, 128]);  expand_16 = None
        expand_17: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_38, [32, 16, 128, 128]);  permute_38 = None
        clone_25: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_79: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_25, [512, 128, 128]);  clone_25 = None
        bmm_7: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_78, view_79);  view_78 = None
        view_80: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_7, [32, 16, 128, 128]);  bmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_40: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
        clone_26: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_81: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_26, [32, 128, 2048]);  clone_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_82: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_81, [4096, 2048]);  view_81 = None
        permute_41: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_52, [1, 0])
        addmm_9: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_53, view_82, permute_41);  primals_53 = permute_41 = None
        view_83: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_9, [32, 128, 2048]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_34: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_83, add_30);  view_83 = add_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_7 = torch.ops.aten.var_mean.correction(add_34, [2], correction = 0, keepdim = True)
        getitem_14: "f32[32, 128, 1]" = var_mean_7[0]
        getitem_15: "f32[32, 128, 1]" = var_mean_7[1];  var_mean_7 = None
        add_35: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_7: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        sub_13: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_34, getitem_15);  getitem_15 = None
        mul_26: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_7);  sub_13 = None
        mul_27: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_26, primals_54)
        add_36: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_27, primals_55);  mul_27 = primals_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_84: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_36, [4096, 2048]);  add_36 = None
        permute_42: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_56, [1, 0])
        addmm_10: "f32[4096, 8192]" = torch.ops.aten.addmm.default(primals_57, view_84, permute_42);  primals_57 = permute_42 = None
        view_85: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_10, [32, 128, 8192])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_28: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_85, 0.5)
        pow_4: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_85, 3.0)
        mul_29: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_37: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_85, mul_29);  view_85 = mul_29 = None
        mul_30: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_37, 0.7978845608028654);  add_37 = None
        tanh_3: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
        add_38: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_31: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_28, add_38);  mul_28 = add_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_86: "f32[4096, 8192]" = torch.ops.aten.reshape.default(mul_31, [4096, 8192]);  mul_31 = None
        permute_43: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_58, [1, 0])
        addmm_11: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_59, view_86, permute_43);  primals_59 = permute_43 = None
        view_87: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_11, [32, 128, 2048]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_39: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_34, view_87);  add_34 = view_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_8 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
        getitem_16: "f32[32, 128, 1]" = var_mean_8[0]
        getitem_17: "f32[32, 128, 1]" = var_mean_8[1];  var_mean_8 = None
        add_40: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_8: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        sub_14: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_39, getitem_17);  getitem_17 = None
        mul_32: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_8);  sub_14 = None
        mul_33: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_32, primals_60)
        add_41: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_33, primals_61);  mul_33 = primals_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_44: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_62, [1, 0])
        view_88: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_41, [4096, 2048]);  add_41 = None
        mm_12: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_88, permute_44);  permute_44 = None
        view_89: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_12, [32, 128, 2048]);  mm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_45: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_63, [1, 0])
        mm_13: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_88, permute_45);  permute_45 = None
        view_91: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_13, [32, 128, 2048]);  mm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_46: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_64, [1, 0])
        mm_14: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_88, permute_46);  permute_46 = None
        view_93: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_14, [32, 128, 2048]);  mm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_94: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_89, [32, 128, 16, 128]);  view_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_47: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_95: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_91, [32, 128, 16, 128]);  view_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_48: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_96: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_93, [32, 128, 16, 128]);  view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_49: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_50: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(permute_48, [0, 1, 3, 2]);  permute_48 = None
        expand_18: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_47, [32, 16, 128, 128]);  permute_47 = None
        clone_29: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_97: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_29, [512, 128, 128]);  clone_29 = None
        expand_19: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_50, [32, 16, 128, 128]);  permute_50 = None
        clone_30: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_98: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_30, [512, 128, 128]);  clone_30 = None
        bmm_8: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_97, view_98)
        view_99: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_8, [32, 16, 128, 128]);  bmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_12: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_65, 2, 0, 128)
        slice_13: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_12, 3, 0, 128);  slice_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_5: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_13, view_99, full_default_2);  slice_13 = view_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_42: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_5, where);  where_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_4: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_42, [-1], True)
        sub_15: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_42, amax_4);  add_42 = amax_4 = None
        exp_4: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
        sum_5: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_20: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_4, [32, 16, 128, 128])
        view_100: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_20, [512, 128, 128]);  expand_20 = None
        expand_21: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_49, [32, 16, 128, 128]);  permute_49 = None
        clone_32: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_101: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_32, [512, 128, 128]);  clone_32 = None
        bmm_9: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_100, view_101);  view_100 = None
        view_102: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_9, [32, 16, 128, 128]);  bmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_51: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
        clone_33: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_103: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_33, [32, 128, 2048]);  clone_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_104: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_103, [4096, 2048]);  view_103 = None
        permute_52: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_66, [1, 0])
        addmm_12: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_67, view_104, permute_52);  primals_67 = permute_52 = None
        view_105: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_12, [32, 128, 2048]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_43: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_105, add_39);  view_105 = add_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_9 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_18: "f32[32, 128, 1]" = var_mean_9[0]
        getitem_19: "f32[32, 128, 1]" = var_mean_9[1];  var_mean_9 = None
        add_44: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_9: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_16: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_43, getitem_19);  getitem_19 = None
        mul_34: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_9);  sub_16 = None
        mul_35: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_34, primals_68)
        add_45: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_35, primals_69);  mul_35 = primals_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_106: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_45, [4096, 2048]);  add_45 = None
        permute_53: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_70, [1, 0])
        addmm_13: "f32[4096, 8192]" = torch.ops.aten.addmm.default(primals_71, view_106, permute_53);  primals_71 = permute_53 = None
        view_107: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_13, [32, 128, 8192])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_36: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_107, 0.5)
        pow_5: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_107, 3.0)
        mul_37: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_46: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_107, mul_37);  view_107 = mul_37 = None
        mul_38: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_46, 0.7978845608028654);  add_46 = None
        tanh_4: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_47: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_39: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_36, add_47);  mul_36 = add_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_108: "f32[4096, 8192]" = torch.ops.aten.reshape.default(mul_39, [4096, 8192]);  mul_39 = None
        permute_54: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_72, [1, 0])
        addmm_14: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_73, view_108, permute_54);  primals_73 = permute_54 = None
        view_109: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_14, [32, 128, 2048]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_48: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_43, view_109);  add_43 = view_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_10 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
        getitem_20: "f32[32, 128, 1]" = var_mean_10[0]
        getitem_21: "f32[32, 128, 1]" = var_mean_10[1];  var_mean_10 = None
        add_49: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_10: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        sub_17: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_48, getitem_21);  getitem_21 = None
        mul_40: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_10);  sub_17 = None
        mul_41: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_40, primals_74)
        add_50: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_41, primals_75);  mul_41 = primals_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_55: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_76, [1, 0])
        view_110: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_50, [4096, 2048]);  add_50 = None
        mm_15: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_110, permute_55);  permute_55 = None
        view_111: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_15, [32, 128, 2048]);  mm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_56: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_77, [1, 0])
        mm_16: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_110, permute_56);  permute_56 = None
        view_113: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_16, [32, 128, 2048]);  mm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_57: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_78, [1, 0])
        mm_17: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_110, permute_57);  permute_57 = None
        view_115: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_17, [32, 128, 2048]);  mm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_116: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_111, [32, 128, 16, 128]);  view_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_58: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_117: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_113, [32, 128, 16, 128]);  view_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_59: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_118: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_115, [32, 128, 16, 128]);  view_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_60: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_61: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(permute_59, [0, 1, 3, 2]);  permute_59 = None
        expand_22: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_58, [32, 16, 128, 128]);  permute_58 = None
        clone_36: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_119: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_36, [512, 128, 128]);  clone_36 = None
        expand_23: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_61, [32, 16, 128, 128]);  permute_61 = None
        clone_37: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_120: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_37, [512, 128, 128]);  clone_37 = None
        bmm_10: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_119, view_120)
        view_121: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_10, [32, 16, 128, 128]);  bmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_14: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_79, 2, 0, 128)
        slice_15: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_14, 3, 0, 128);  slice_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_6: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_15, view_121, full_default_2);  slice_15 = view_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_51: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_6, where);  where_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_5: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_51, [-1], True)
        sub_18: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_51, amax_5);  add_51 = amax_5 = None
        exp_5: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
        sum_6: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_24: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_5, [32, 16, 128, 128])
        view_122: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_24, [512, 128, 128]);  expand_24 = None
        expand_25: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_60, [32, 16, 128, 128]);  permute_60 = None
        clone_39: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_123: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_39, [512, 128, 128]);  clone_39 = None
        bmm_11: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_122, view_123);  view_122 = None
        view_124: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_11, [32, 16, 128, 128]);  bmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_62: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
        clone_40: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_125: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_40, [32, 128, 2048]);  clone_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_126: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_125, [4096, 2048]);  view_125 = None
        permute_63: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_80, [1, 0])
        addmm_15: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_81, view_126, permute_63);  primals_81 = permute_63 = None
        view_127: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_15, [32, 128, 2048]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_52: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_127, add_48);  view_127 = add_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_11 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
        getitem_22: "f32[32, 128, 1]" = var_mean_11[0]
        getitem_23: "f32[32, 128, 1]" = var_mean_11[1];  var_mean_11 = None
        add_53: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_11: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_19: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_52, getitem_23);  getitem_23 = None
        mul_42: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_11);  sub_19 = None
        mul_43: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_42, primals_82)
        add_54: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_43, primals_83);  mul_43 = primals_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_128: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_54, [4096, 2048]);  add_54 = None
        permute_64: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_84, [1, 0])
        addmm_16: "f32[4096, 8192]" = torch.ops.aten.addmm.default(primals_85, view_128, permute_64);  primals_85 = permute_64 = None
        view_129: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_16, [32, 128, 8192])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_44: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_129, 0.5)
        pow_6: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_129, 3.0)
        mul_45: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_55: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_129, mul_45);  view_129 = mul_45 = None
        mul_46: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_55, 0.7978845608028654);  add_55 = None
        tanh_5: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
        add_56: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_47: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_44, add_56);  mul_44 = add_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_130: "f32[4096, 8192]" = torch.ops.aten.reshape.default(mul_47, [4096, 8192]);  mul_47 = None
        permute_65: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_86, [1, 0])
        addmm_17: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_87, view_130, permute_65);  primals_87 = permute_65 = None
        view_131: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_17, [32, 128, 2048]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_57: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_52, view_131);  add_52 = view_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_12 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_24: "f32[32, 128, 1]" = var_mean_12[0]
        getitem_25: "f32[32, 128, 1]" = var_mean_12[1];  var_mean_12 = None
        add_58: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_12: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_20: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_57, getitem_25);  getitem_25 = None
        mul_48: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_12);  sub_20 = None
        mul_49: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_48, primals_88)
        add_59: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_49, primals_89);  mul_49 = primals_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_66: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_90, [1, 0])
        view_132: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_59, [4096, 2048]);  add_59 = None
        mm_18: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_132, permute_66);  permute_66 = None
        view_133: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_18, [32, 128, 2048]);  mm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_67: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_91, [1, 0])
        mm_19: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_132, permute_67);  permute_67 = None
        view_135: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_19, [32, 128, 2048]);  mm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_68: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_92, [1, 0])
        mm_20: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_132, permute_68);  permute_68 = None
        view_137: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_20, [32, 128, 2048]);  mm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_138: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_133, [32, 128, 16, 128]);  view_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_69: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_139: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_135, [32, 128, 16, 128]);  view_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_70: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_140: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_137, [32, 128, 16, 128]);  view_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_71: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_72: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(permute_70, [0, 1, 3, 2]);  permute_70 = None
        expand_26: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_69, [32, 16, 128, 128]);  permute_69 = None
        clone_43: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_141: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_43, [512, 128, 128]);  clone_43 = None
        expand_27: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_72, [32, 16, 128, 128]);  permute_72 = None
        clone_44: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_142: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_44, [512, 128, 128]);  clone_44 = None
        bmm_12: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_141, view_142)
        view_143: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_12, [32, 16, 128, 128]);  bmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_16: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_93, 2, 0, 128)
        slice_17: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_16, 3, 0, 128);  slice_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_7: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_17, view_143, full_default_2);  slice_17 = view_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_60: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_7, where);  where_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_6: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_60, [-1], True)
        sub_21: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_60, amax_6);  add_60 = amax_6 = None
        exp_6: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
        sum_7: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_28: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_6, [32, 16, 128, 128])
        view_144: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_28, [512, 128, 128]);  expand_28 = None
        expand_29: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_71, [32, 16, 128, 128]);  permute_71 = None
        clone_46: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_145: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_46, [512, 128, 128]);  clone_46 = None
        bmm_13: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_144, view_145);  view_144 = None
        view_146: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_13, [32, 16, 128, 128]);  bmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_73: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
        clone_47: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_147: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_47, [32, 128, 2048]);  clone_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_148: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_147, [4096, 2048]);  view_147 = None
        permute_74: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_94, [1, 0])
        addmm_18: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_95, view_148, permute_74);  primals_95 = permute_74 = None
        view_149: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_18, [32, 128, 2048]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_61: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_149, add_57);  view_149 = add_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_13 = torch.ops.aten.var_mean.correction(add_61, [2], correction = 0, keepdim = True)
        getitem_26: "f32[32, 128, 1]" = var_mean_13[0]
        getitem_27: "f32[32, 128, 1]" = var_mean_13[1];  var_mean_13 = None
        add_62: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_13: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        sub_22: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_61, getitem_27);  getitem_27 = None
        mul_50: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_13);  sub_22 = None
        mul_51: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_50, primals_96)
        add_63: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_51, primals_97);  mul_51 = primals_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_150: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_63, [4096, 2048]);  add_63 = None
        permute_75: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_98, [1, 0])
        addmm_19: "f32[4096, 8192]" = torch.ops.aten.addmm.default(primals_99, view_150, permute_75);  primals_99 = permute_75 = None
        view_151: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_19, [32, 128, 8192])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_52: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_151, 0.5)
        pow_7: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_151, 3.0)
        mul_53: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_64: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_151, mul_53);  view_151 = mul_53 = None
        mul_54: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_64, 0.7978845608028654);  add_64 = None
        tanh_6: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
        add_65: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_55: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_52, add_65);  mul_52 = add_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_152: "f32[4096, 8192]" = torch.ops.aten.reshape.default(mul_55, [4096, 8192]);  mul_55 = None
        permute_76: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_100, [1, 0])
        addmm_20: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_101, view_152, permute_76);  primals_101 = permute_76 = None
        view_153: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_20, [32, 128, 2048]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_66: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_61, view_153);  add_61 = view_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_14 = torch.ops.aten.var_mean.correction(add_66, [2], correction = 0, keepdim = True)
        getitem_28: "f32[32, 128, 1]" = var_mean_14[0]
        getitem_29: "f32[32, 128, 1]" = var_mean_14[1];  var_mean_14 = None
        add_67: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_14: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
        sub_23: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_66, getitem_29);  getitem_29 = None
        mul_56: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_14);  sub_23 = None
        mul_57: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_56, primals_102)
        add_68: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_57, primals_103);  mul_57 = primals_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_77: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_104, [1, 0])
        view_154: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_68, [4096, 2048]);  add_68 = None
        mm_21: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_154, permute_77);  permute_77 = None
        view_155: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_21, [32, 128, 2048]);  mm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_78: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_105, [1, 0])
        mm_22: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_154, permute_78);  permute_78 = None
        view_157: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_22, [32, 128, 2048]);  mm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_79: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_106, [1, 0])
        mm_23: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_154, permute_79);  permute_79 = None
        view_159: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_23, [32, 128, 2048]);  mm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_160: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_155, [32, 128, 16, 128]);  view_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_80: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_161: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_157, [32, 128, 16, 128]);  view_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_81: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_161, [0, 2, 1, 3]);  view_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_162: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_159, [32, 128, 16, 128]);  view_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_82: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_83: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(permute_81, [0, 1, 3, 2]);  permute_81 = None
        expand_30: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_80, [32, 16, 128, 128]);  permute_80 = None
        clone_50: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_163: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_50, [512, 128, 128]);  clone_50 = None
        expand_31: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_83, [32, 16, 128, 128]);  permute_83 = None
        clone_51: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_164: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_51, [512, 128, 128]);  clone_51 = None
        bmm_14: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_163, view_164)
        view_165: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_14, [32, 16, 128, 128]);  bmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_18: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_107, 2, 0, 128)
        slice_19: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_18, 3, 0, 128);  slice_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_8: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_19, view_165, full_default_2);  slice_19 = view_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_69: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_8, where);  where_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_7: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_69, [-1], True)
        sub_24: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_69, amax_7);  add_69 = amax_7 = None
        exp_7: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_24);  sub_24 = None
        sum_8: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_32: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_7, [32, 16, 128, 128])
        view_166: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_32, [512, 128, 128]);  expand_32 = None
        expand_33: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_82, [32, 16, 128, 128]);  permute_82 = None
        clone_53: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_167: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_53, [512, 128, 128]);  clone_53 = None
        bmm_15: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_166, view_167);  view_166 = None
        view_168: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_15, [32, 16, 128, 128]);  bmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_84: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        clone_54: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_169: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_54, [32, 128, 2048]);  clone_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_170: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_169, [4096, 2048]);  view_169 = None
        permute_85: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_108, [1, 0])
        addmm_21: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_109, view_170, permute_85);  primals_109 = permute_85 = None
        view_171: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_21, [32, 128, 2048]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_70: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_171, add_66);  view_171 = add_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_15 = torch.ops.aten.var_mean.correction(add_70, [2], correction = 0, keepdim = True)
        getitem_30: "f32[32, 128, 1]" = var_mean_15[0]
        getitem_31: "f32[32, 128, 1]" = var_mean_15[1];  var_mean_15 = None
        add_71: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_15: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        sub_25: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_70, getitem_31);  getitem_31 = None
        mul_58: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_15);  sub_25 = None
        mul_59: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_58, primals_110)
        add_72: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_59, primals_111);  mul_59 = primals_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_172: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_72, [4096, 2048]);  add_72 = None
        permute_86: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_112, [1, 0])
        addmm_22: "f32[4096, 8192]" = torch.ops.aten.addmm.default(primals_113, view_172, permute_86);  primals_113 = permute_86 = None
        view_173: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_22, [32, 128, 8192])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_60: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_173, 0.5)
        pow_8: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_173, 3.0)
        mul_61: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_73: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_173, mul_61);  view_173 = mul_61 = None
        mul_62: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_73, 0.7978845608028654);  add_73 = None
        tanh_7: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_62);  mul_62 = None
        add_74: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_63: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_60, add_74);  mul_60 = add_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_174: "f32[4096, 8192]" = torch.ops.aten.reshape.default(mul_63, [4096, 8192]);  mul_63 = None
        permute_87: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_114, [1, 0])
        addmm_23: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_115, view_174, permute_87);  primals_115 = permute_87 = None
        view_175: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_23, [32, 128, 2048]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_75: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_70, view_175);  add_70 = view_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_16 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
        getitem_32: "f32[32, 128, 1]" = var_mean_16[0]
        getitem_33: "f32[32, 128, 1]" = var_mean_16[1];  var_mean_16 = None
        add_76: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_16: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        sub_26: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_75, getitem_33);  getitem_33 = None
        mul_64: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_16);  sub_26 = None
        mul_65: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_64, primals_116)
        add_77: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_65, primals_117);  mul_65 = primals_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_88: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_118, [1, 0])
        view_176: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_77, [4096, 2048]);  add_77 = None
        mm_24: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_176, permute_88);  permute_88 = None
        view_177: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_24, [32, 128, 2048]);  mm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_89: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_119, [1, 0])
        mm_25: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_176, permute_89);  permute_89 = None
        view_179: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_25, [32, 128, 2048]);  mm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_90: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_120, [1, 0])
        mm_26: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_176, permute_90);  permute_90 = None
        view_181: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_26, [32, 128, 2048]);  mm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_182: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_177, [32, 128, 16, 128]);  view_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_91: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_183: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_179, [32, 128, 16, 128]);  view_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_92: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_184: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_181, [32, 128, 16, 128]);  view_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_93: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_94: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(permute_92, [0, 1, 3, 2]);  permute_92 = None
        expand_34: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_91, [32, 16, 128, 128]);  permute_91 = None
        clone_57: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_185: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_57, [512, 128, 128]);  clone_57 = None
        expand_35: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_94, [32, 16, 128, 128]);  permute_94 = None
        clone_58: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_186: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_58, [512, 128, 128]);  clone_58 = None
        bmm_16: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_185, view_186)
        view_187: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_16, [32, 16, 128, 128]);  bmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_20: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_121, 2, 0, 128)
        slice_21: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_20, 3, 0, 128);  slice_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_9: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_21, view_187, full_default_2);  slice_21 = view_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_78: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_9, where);  where_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_8: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_78, [-1], True)
        sub_27: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_78, amax_8);  add_78 = amax_8 = None
        exp_8: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
        sum_9: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_36: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_8, [32, 16, 128, 128])
        view_188: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_36, [512, 128, 128]);  expand_36 = None
        expand_37: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_93, [32, 16, 128, 128]);  permute_93 = None
        clone_60: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_189: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_60, [512, 128, 128]);  clone_60 = None
        bmm_17: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_188, view_189);  view_188 = None
        view_190: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_17, [32, 16, 128, 128]);  bmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_95: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
        clone_61: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_191: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_61, [32, 128, 2048]);  clone_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_192: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_191, [4096, 2048]);  view_191 = None
        permute_96: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_122, [1, 0])
        addmm_24: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_123, view_192, permute_96);  primals_123 = permute_96 = None
        view_193: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_24, [32, 128, 2048]);  addmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_79: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_193, add_75);  view_193 = add_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_17 = torch.ops.aten.var_mean.correction(add_79, [2], correction = 0, keepdim = True)
        getitem_34: "f32[32, 128, 1]" = var_mean_17[0]
        getitem_35: "f32[32, 128, 1]" = var_mean_17[1];  var_mean_17 = None
        add_80: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_17: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        sub_28: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_79, getitem_35);  getitem_35 = None
        mul_66: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_17);  sub_28 = None
        mul_67: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_66, primals_124)
        add_81: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_67, primals_125);  mul_67 = primals_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_194: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_81, [4096, 2048]);  add_81 = None
        permute_97: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_126, [1, 0])
        addmm_25: "f32[4096, 8192]" = torch.ops.aten.addmm.default(primals_127, view_194, permute_97);  primals_127 = permute_97 = None
        view_195: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_25, [32, 128, 8192])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_68: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_195, 0.5)
        pow_9: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_195, 3.0)
        mul_69: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_82: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_195, mul_69);  view_195 = mul_69 = None
        mul_70: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_82, 0.7978845608028654);  add_82 = None
        tanh_8: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
        add_83: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_71: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_68, add_83);  mul_68 = add_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_196: "f32[4096, 8192]" = torch.ops.aten.reshape.default(mul_71, [4096, 8192]);  mul_71 = None
        permute_98: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_128, [1, 0])
        addmm_26: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_129, view_196, permute_98);  primals_129 = permute_98 = None
        view_197: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_26, [32, 128, 2048]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_84: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_79, view_197);  add_79 = view_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_18 = torch.ops.aten.var_mean.correction(add_84, [2], correction = 0, keepdim = True)
        getitem_36: "f32[32, 128, 1]" = var_mean_18[0]
        getitem_37: "f32[32, 128, 1]" = var_mean_18[1];  var_mean_18 = None
        add_85: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_18: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        sub_29: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_84, getitem_37);  getitem_37 = None
        mul_72: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_18);  sub_29 = None
        mul_73: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_72, primals_130)
        add_86: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_73, primals_131);  mul_73 = primals_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_99: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_132, [1, 0])
        view_198: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_86, [4096, 2048]);  add_86 = None
        mm_27: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_198, permute_99);  permute_99 = None
        view_199: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_27, [32, 128, 2048]);  mm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_100: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_133, [1, 0])
        mm_28: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_198, permute_100);  permute_100 = None
        view_201: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_28, [32, 128, 2048]);  mm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_101: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_134, [1, 0])
        mm_29: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_198, permute_101);  permute_101 = None
        view_203: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_29, [32, 128, 2048]);  mm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_204: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_199, [32, 128, 16, 128]);  view_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_102: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_205: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_201, [32, 128, 16, 128]);  view_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_103: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_206: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_203, [32, 128, 16, 128]);  view_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_104: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_105: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(permute_103, [0, 1, 3, 2]);  permute_103 = None
        expand_38: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_102, [32, 16, 128, 128]);  permute_102 = None
        clone_64: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_207: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_64, [512, 128, 128]);  clone_64 = None
        expand_39: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_105, [32, 16, 128, 128]);  permute_105 = None
        clone_65: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_208: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_65, [512, 128, 128]);  clone_65 = None
        bmm_18: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_207, view_208)
        view_209: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_18, [32, 16, 128, 128]);  bmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_22: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_135, 2, 0, 128)
        slice_23: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_22, 3, 0, 128);  slice_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_10: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_23, view_209, full_default_2);  slice_23 = view_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_87: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_10, where);  where_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_9: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_87, [-1], True)
        sub_30: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_87, amax_9);  add_87 = amax_9 = None
        exp_9: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_30);  sub_30 = None
        sum_10: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_40: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_9, [32, 16, 128, 128])
        view_210: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_40, [512, 128, 128]);  expand_40 = None
        expand_41: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_104, [32, 16, 128, 128]);  permute_104 = None
        clone_67: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_211: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_67, [512, 128, 128]);  clone_67 = None
        bmm_19: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_210, view_211);  view_210 = None
        view_212: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_19, [32, 16, 128, 128]);  bmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_106: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
        clone_68: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_213: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_68, [32, 128, 2048]);  clone_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_214: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_213, [4096, 2048]);  view_213 = None
        permute_107: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_136, [1, 0])
        addmm_27: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_137, view_214, permute_107);  primals_137 = permute_107 = None
        view_215: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_27, [32, 128, 2048]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_88: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_215, add_84);  view_215 = add_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_19 = torch.ops.aten.var_mean.correction(add_88, [2], correction = 0, keepdim = True)
        getitem_38: "f32[32, 128, 1]" = var_mean_19[0]
        getitem_39: "f32[32, 128, 1]" = var_mean_19[1];  var_mean_19 = None
        add_89: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_19: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
        sub_31: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_88, getitem_39);  getitem_39 = None
        mul_74: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_19);  sub_31 = None
        mul_75: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_74, primals_138)
        add_90: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_75, primals_139);  mul_75 = primals_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_216: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_90, [4096, 2048]);  add_90 = None
        permute_108: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_140, [1, 0])
        addmm_28: "f32[4096, 8192]" = torch.ops.aten.addmm.default(primals_141, view_216, permute_108);  primals_141 = permute_108 = None
        view_217: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_28, [32, 128, 8192])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_76: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_217, 0.5)
        pow_10: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_217, 3.0)
        mul_77: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_91: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_217, mul_77);  view_217 = mul_77 = None
        mul_78: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_91, 0.7978845608028654);  add_91 = None
        tanh_9: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
        add_92: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_79: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_76, add_92);  mul_76 = add_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_218: "f32[4096, 8192]" = torch.ops.aten.reshape.default(mul_79, [4096, 8192]);  mul_79 = None
        permute_109: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_142, [1, 0])
        addmm_29: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_143, view_218, permute_109);  primals_143 = permute_109 = None
        view_219: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_29, [32, 128, 2048]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_93: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_88, view_219);  add_88 = view_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_20 = torch.ops.aten.var_mean.correction(add_93, [2], correction = 0, keepdim = True)
        getitem_40: "f32[32, 128, 1]" = var_mean_20[0]
        getitem_41: "f32[32, 128, 1]" = var_mean_20[1];  var_mean_20 = None
        add_94: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_20: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        sub_32: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_93, getitem_41);  getitem_41 = None
        mul_80: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_20);  sub_32 = None
        mul_81: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_80, primals_144)
        add_95: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_81, primals_145);  mul_81 = primals_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_110: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_146, [1, 0])
        view_220: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_95, [4096, 2048]);  add_95 = None
        mm_30: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_220, permute_110);  permute_110 = None
        view_221: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_30, [32, 128, 2048]);  mm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_111: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_147, [1, 0])
        mm_31: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_220, permute_111);  permute_111 = None
        view_223: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_31, [32, 128, 2048]);  mm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_112: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_148, [1, 0])
        mm_32: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_220, permute_112);  permute_112 = None
        view_225: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_32, [32, 128, 2048]);  mm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_226: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_221, [32, 128, 16, 128]);  view_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_113: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_227: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_223, [32, 128, 16, 128]);  view_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_114: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_228: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_225, [32, 128, 16, 128]);  view_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_115: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_116: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(permute_114, [0, 1, 3, 2]);  permute_114 = None
        expand_42: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_113, [32, 16, 128, 128]);  permute_113 = None
        clone_71: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_229: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_71, [512, 128, 128]);  clone_71 = None
        expand_43: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_116, [32, 16, 128, 128]);  permute_116 = None
        clone_72: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_230: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_72, [512, 128, 128]);  clone_72 = None
        bmm_20: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_229, view_230)
        view_231: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_20, [32, 16, 128, 128]);  bmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_24: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_149, 2, 0, 128)
        slice_25: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_24, 3, 0, 128);  slice_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_11: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_25, view_231, full_default_2);  slice_25 = view_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_96: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_11, where);  where_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_10: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_96, [-1], True)
        sub_33: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_96, amax_10);  add_96 = amax_10 = None
        exp_10: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_33);  sub_33 = None
        sum_11: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_44: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_10, [32, 16, 128, 128])
        view_232: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_44, [512, 128, 128]);  expand_44 = None
        expand_45: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_115, [32, 16, 128, 128]);  permute_115 = None
        clone_74: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_233: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_74, [512, 128, 128]);  clone_74 = None
        bmm_21: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_232, view_233);  view_232 = None
        view_234: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_21, [32, 16, 128, 128]);  bmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_117: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None
        clone_75: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_235: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_75, [32, 128, 2048]);  clone_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_236: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_235, [4096, 2048]);  view_235 = None
        permute_118: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_150, [1, 0])
        addmm_30: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_151, view_236, permute_118);  primals_151 = permute_118 = None
        view_237: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_30, [32, 128, 2048]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_97: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_237, add_93);  view_237 = add_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_21 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
        getitem_42: "f32[32, 128, 1]" = var_mean_21[0]
        getitem_43: "f32[32, 128, 1]" = var_mean_21[1];  var_mean_21 = None
        add_98: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_21: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        sub_34: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_97, getitem_43);  getitem_43 = None
        mul_82: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_21);  sub_34 = None
        mul_83: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_82, primals_152)
        add_99: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_83, primals_153);  mul_83 = primals_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_238: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_99, [4096, 2048]);  add_99 = None
        permute_119: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_154, [1, 0])
        addmm_31: "f32[4096, 8192]" = torch.ops.aten.addmm.default(primals_155, view_238, permute_119);  primals_155 = permute_119 = None
        view_239: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_31, [32, 128, 8192])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_84: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_239, 0.5)
        pow_11: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_239, 3.0)
        mul_85: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_100: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_239, mul_85);  view_239 = mul_85 = None
        mul_86: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_100, 0.7978845608028654);  add_100 = None
        tanh_10: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_86);  mul_86 = None
        add_101: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_87: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_84, add_101);  mul_84 = add_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_240: "f32[4096, 8192]" = torch.ops.aten.reshape.default(mul_87, [4096, 8192]);  mul_87 = None
        permute_120: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_156, [1, 0])
        addmm_32: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_157, view_240, permute_120);  primals_157 = permute_120 = None
        view_241: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_32, [32, 128, 2048]);  addmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_102: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_97, view_241);  add_97 = view_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_22 = torch.ops.aten.var_mean.correction(add_102, [2], correction = 0, keepdim = True)
        getitem_44: "f32[32, 128, 1]" = var_mean_22[0]
        getitem_45: "f32[32, 128, 1]" = var_mean_22[1];  var_mean_22 = None
        add_103: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_22: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
        sub_35: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_102, getitem_45);  getitem_45 = None
        mul_88: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_22);  sub_35 = None
        mul_89: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_88, primals_158)
        add_104: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_89, primals_159);  mul_89 = primals_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_121: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_160, [1, 0])
        view_242: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_104, [4096, 2048]);  add_104 = None
        mm_33: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_242, permute_121);  permute_121 = None
        view_243: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_33, [32, 128, 2048]);  mm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_122: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_161, [1, 0])
        mm_34: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_242, permute_122);  permute_122 = None
        view_245: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_34, [32, 128, 2048]);  mm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_123: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_162, [1, 0])
        mm_35: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_242, permute_123);  permute_123 = None
        view_247: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_35, [32, 128, 2048]);  mm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_248: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_243, [32, 128, 16, 128]);  view_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_124: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_248, [0, 2, 1, 3]);  view_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_249: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_245, [32, 128, 16, 128]);  view_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_125: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_250: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_247, [32, 128, 16, 128]);  view_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_126: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_127: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(permute_125, [0, 1, 3, 2]);  permute_125 = None
        expand_46: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_124, [32, 16, 128, 128]);  permute_124 = None
        clone_78: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_251: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_78, [512, 128, 128]);  clone_78 = None
        expand_47: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_127, [32, 16, 128, 128]);  permute_127 = None
        clone_79: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_252: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_79, [512, 128, 128]);  clone_79 = None
        bmm_22: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_251, view_252)
        view_253: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_22, [32, 16, 128, 128]);  bmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_26: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_163, 2, 0, 128)
        slice_27: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_26, 3, 0, 128);  slice_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_12: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_27, view_253, full_default_2);  slice_27 = view_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_105: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_12, where);  where_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_11: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_105, [-1], True)
        sub_36: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_105, amax_11);  add_105 = amax_11 = None
        exp_11: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_36);  sub_36 = None
        sum_12: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_48: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_11, [32, 16, 128, 128])
        view_254: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_48, [512, 128, 128]);  expand_48 = None
        expand_49: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_126, [32, 16, 128, 128]);  permute_126 = None
        clone_81: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_255: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_81, [512, 128, 128]);  clone_81 = None
        bmm_23: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_254, view_255);  view_254 = None
        view_256: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_23, [32, 16, 128, 128]);  bmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_128: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
        clone_82: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_257: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_82, [32, 128, 2048]);  clone_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_258: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_257, [4096, 2048]);  view_257 = None
        permute_129: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_164, [1, 0])
        addmm_33: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_165, view_258, permute_129);  primals_165 = permute_129 = None
        view_259: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_33, [32, 128, 2048]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_106: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_259, add_102);  view_259 = add_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_23 = torch.ops.aten.var_mean.correction(add_106, [2], correction = 0, keepdim = True)
        getitem_46: "f32[32, 128, 1]" = var_mean_23[0]
        getitem_47: "f32[32, 128, 1]" = var_mean_23[1];  var_mean_23 = None
        add_107: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_23: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_107);  add_107 = None
        sub_37: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_106, getitem_47);  getitem_47 = None
        mul_90: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_23);  sub_37 = None
        mul_91: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_90, primals_166)
        add_108: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_91, primals_167);  mul_91 = primals_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_260: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_108, [4096, 2048]);  add_108 = None
        permute_130: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_168, [1, 0])
        addmm_34: "f32[4096, 8192]" = torch.ops.aten.addmm.default(primals_169, view_260, permute_130);  primals_169 = permute_130 = None
        view_261: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_34, [32, 128, 8192])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_92: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_261, 0.5)
        pow_12: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_261, 3.0)
        mul_93: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_109: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_261, mul_93);  view_261 = mul_93 = None
        mul_94: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_109, 0.7978845608028654);  add_109 = None
        tanh_11: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
        add_110: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_95: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_92, add_110);  mul_92 = add_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_262: "f32[4096, 8192]" = torch.ops.aten.reshape.default(mul_95, [4096, 8192]);  mul_95 = None
        permute_131: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_170, [1, 0])
        addmm_35: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_171, view_262, permute_131);  primals_171 = permute_131 = None
        view_263: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_35, [32, 128, 2048]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_111: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_106, view_263);  add_106 = view_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_24 = torch.ops.aten.var_mean.correction(add_111, [2], correction = 0, keepdim = True)
        getitem_48: "f32[32, 128, 1]" = var_mean_24[0]
        getitem_49: "f32[32, 128, 1]" = var_mean_24[1];  var_mean_24 = None
        add_112: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_24: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
        sub_38: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_111, getitem_49);  getitem_49 = None
        mul_96: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_24);  sub_38 = None
        mul_97: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_96, primals_172)
        add_113: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_97, primals_173);  mul_97 = primals_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_132: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_174, [1, 0])
        view_264: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_113, [4096, 2048]);  add_113 = None
        mm_36: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_264, permute_132);  permute_132 = None
        view_265: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_36, [32, 128, 2048]);  mm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_133: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_175, [1, 0])
        mm_37: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_264, permute_133);  permute_133 = None
        view_267: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_37, [32, 128, 2048]);  mm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_134: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_176, [1, 0])
        mm_38: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_264, permute_134);  permute_134 = None
        view_269: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_38, [32, 128, 2048]);  mm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_270: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_265, [32, 128, 16, 128]);  view_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_135: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_270, [0, 2, 1, 3]);  view_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_271: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_267, [32, 128, 16, 128]);  view_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_136: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_271, [0, 2, 1, 3]);  view_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_272: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_269, [32, 128, 16, 128]);  view_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_137: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_272, [0, 2, 1, 3]);  view_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_138: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(permute_136, [0, 1, 3, 2]);  permute_136 = None
        expand_50: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_135, [32, 16, 128, 128]);  permute_135 = None
        clone_85: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_273: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_85, [512, 128, 128]);  clone_85 = None
        expand_51: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_138, [32, 16, 128, 128]);  permute_138 = None
        clone_86: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        view_274: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_86, [512, 128, 128]);  clone_86 = None
        bmm_24: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_273, view_274)
        view_275: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_24, [32, 16, 128, 128]);  bmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_28: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_177, 2, 0, 128)
        slice_29: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_28, 3, 0, 128);  slice_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_13: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_29, view_275, full_default_2);  slice_29 = view_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_114: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_13, where);  where_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_12: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_114, [-1], True)
        sub_39: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_114, amax_12);  add_114 = amax_12 = None
        exp_12: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_39);  sub_39 = None
        sum_13: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_12: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_52: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_12, [32, 16, 128, 128])
        view_276: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_52, [512, 128, 128]);  expand_52 = None
        expand_53: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_137, [32, 16, 128, 128]);  permute_137 = None
        clone_88: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
        view_277: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_88, [512, 128, 128]);  clone_88 = None
        bmm_25: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_276, view_277);  view_276 = None
        view_278: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_25, [32, 16, 128, 128]);  bmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_139: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_278, [0, 2, 1, 3]);  view_278 = None
        clone_89: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_279: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_89, [32, 128, 2048]);  clone_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_280: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_279, [4096, 2048]);  view_279 = None
        permute_140: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_178, [1, 0])
        addmm_36: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_179, view_280, permute_140);  primals_179 = permute_140 = None
        view_281: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_36, [32, 128, 2048]);  addmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_115: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_281, add_111);  view_281 = add_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_25 = torch.ops.aten.var_mean.correction(add_115, [2], correction = 0, keepdim = True)
        getitem_50: "f32[32, 128, 1]" = var_mean_25[0]
        getitem_51: "f32[32, 128, 1]" = var_mean_25[1];  var_mean_25 = None
        add_116: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
        rsqrt_25: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        sub_40: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_115, getitem_51);  getitem_51 = None
        mul_98: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_25);  sub_40 = None
        mul_99: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_98, primals_180)
        add_117: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_99, primals_181);  mul_99 = primals_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_282: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_117, [4096, 2048]);  add_117 = None
        permute_141: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_182, [1, 0])
        addmm_37: "f32[4096, 8192]" = torch.ops.aten.addmm.default(primals_183, view_282, permute_141);  primals_183 = permute_141 = None
        view_283: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_37, [32, 128, 8192])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_100: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_283, 0.5)
        pow_13: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_283, 3.0)
        mul_101: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
        add_118: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_283, mul_101);  view_283 = mul_101 = None
        mul_102: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_118, 0.7978845608028654);  add_118 = None
        tanh_12: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_102);  mul_102 = None
        add_119: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_12, 1.0);  tanh_12 = None
        mul_103: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_100, add_119);  mul_100 = add_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_284: "f32[4096, 8192]" = torch.ops.aten.reshape.default(mul_103, [4096, 8192]);  mul_103 = None
        permute_142: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_184, [1, 0])
        addmm_38: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_185, view_284, permute_142);  primals_185 = permute_142 = None
        view_285: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_38, [32, 128, 2048]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_120: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_115, view_285);  add_115 = view_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_26 = torch.ops.aten.var_mean.correction(add_120, [2], correction = 0, keepdim = True)
        getitem_52: "f32[32, 128, 1]" = var_mean_26[0]
        getitem_53: "f32[32, 128, 1]" = var_mean_26[1];  var_mean_26 = None
        add_121: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
        rsqrt_26: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
        sub_41: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_120, getitem_53);  getitem_53 = None
        mul_104: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_26);  sub_41 = None
        mul_105: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_104, primals_186)
        add_122: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_105, primals_187);  mul_105 = primals_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_143: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_188, [1, 0])
        view_286: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_122, [4096, 2048]);  add_122 = None
        mm_39: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_286, permute_143);  permute_143 = None
        view_287: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_39, [32, 128, 2048]);  mm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_144: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_189, [1, 0])
        mm_40: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_286, permute_144);  permute_144 = None
        view_289: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_40, [32, 128, 2048]);  mm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_145: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_190, [1, 0])
        mm_41: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_286, permute_145);  permute_145 = None
        view_291: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_41, [32, 128, 2048]);  mm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_292: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_287, [32, 128, 16, 128]);  view_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_146: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_293: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_289, [32, 128, 16, 128]);  view_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_147: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_293, [0, 2, 1, 3]);  view_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_294: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_291, [32, 128, 16, 128]);  view_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_148: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_149: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(permute_147, [0, 1, 3, 2]);  permute_147 = None
        expand_54: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_146, [32, 16, 128, 128]);  permute_146 = None
        clone_92: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_54, memory_format = torch.contiguous_format);  expand_54 = None
        view_295: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_92, [512, 128, 128]);  clone_92 = None
        expand_55: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_149, [32, 16, 128, 128]);  permute_149 = None
        clone_93: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
        view_296: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_93, [512, 128, 128]);  clone_93 = None
        bmm_26: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_295, view_296)
        view_297: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_26, [32, 16, 128, 128]);  bmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_30: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_191, 2, 0, 128)
        slice_31: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_30, 3, 0, 128);  slice_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_14: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_31, view_297, full_default_2);  slice_31 = view_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_123: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_14, where);  where_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_13: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_123, [-1], True)
        sub_42: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_123, amax_13);  add_123 = amax_13 = None
        exp_13: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_42);  sub_42 = None
        sum_14: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_13: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_56: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_13, [32, 16, 128, 128])
        view_298: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_56, [512, 128, 128]);  expand_56 = None
        expand_57: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_148, [32, 16, 128, 128]);  permute_148 = None
        clone_95: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_299: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_95, [512, 128, 128]);  clone_95 = None
        bmm_27: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_298, view_299);  view_298 = None
        view_300: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_27, [32, 16, 128, 128]);  bmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_150: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_300, [0, 2, 1, 3]);  view_300 = None
        clone_96: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_301: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_96, [32, 128, 2048]);  clone_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_302: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_301, [4096, 2048]);  view_301 = None
        permute_151: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_192, [1, 0])
        addmm_39: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_193, view_302, permute_151);  primals_193 = permute_151 = None
        view_303: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_39, [32, 128, 2048]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_124: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_303, add_120);  view_303 = add_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_27 = torch.ops.aten.var_mean.correction(add_124, [2], correction = 0, keepdim = True)
        getitem_54: "f32[32, 128, 1]" = var_mean_27[0]
        getitem_55: "f32[32, 128, 1]" = var_mean_27[1];  var_mean_27 = None
        add_125: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
        rsqrt_27: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_125);  add_125 = None
        sub_43: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_124, getitem_55);  getitem_55 = None
        mul_106: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_27);  sub_43 = None
        mul_107: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_106, primals_194)
        add_126: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_107, primals_195);  mul_107 = primals_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_304: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_126, [4096, 2048]);  add_126 = None
        permute_152: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_196, [1, 0])
        addmm_40: "f32[4096, 8192]" = torch.ops.aten.addmm.default(primals_197, view_304, permute_152);  primals_197 = permute_152 = None
        view_305: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_40, [32, 128, 8192])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_108: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_305, 0.5)
        pow_14: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_305, 3.0)
        mul_109: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_14, 0.044715);  pow_14 = None
        add_127: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_305, mul_109);  view_305 = mul_109 = None
        mul_110: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_127, 0.7978845608028654);  add_127 = None
        tanh_13: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_110);  mul_110 = None
        add_128: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_13, 1.0);  tanh_13 = None
        mul_111: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_108, add_128);  mul_108 = add_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_306: "f32[4096, 8192]" = torch.ops.aten.reshape.default(mul_111, [4096, 8192]);  mul_111 = None
        permute_153: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_198, [1, 0])
        addmm_41: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_199, view_306, permute_153);  primals_199 = permute_153 = None
        view_307: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_41, [32, 128, 2048]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_129: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_124, view_307);  add_124 = view_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_28 = torch.ops.aten.var_mean.correction(add_129, [2], correction = 0, keepdim = True)
        getitem_56: "f32[32, 128, 1]" = var_mean_28[0]
        getitem_57: "f32[32, 128, 1]" = var_mean_28[1];  var_mean_28 = None
        add_130: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
        rsqrt_28: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        sub_44: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_129, getitem_57);  getitem_57 = None
        mul_112: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_28);  sub_44 = None
        mul_113: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_112, primals_200)
        add_131: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_113, primals_201);  mul_113 = primals_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_154: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_202, [1, 0])
        view_308: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_131, [4096, 2048]);  add_131 = None
        mm_42: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_308, permute_154);  permute_154 = None
        view_309: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_42, [32, 128, 2048]);  mm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_155: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_203, [1, 0])
        mm_43: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_308, permute_155);  permute_155 = None
        view_311: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_43, [32, 128, 2048]);  mm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_156: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_204, [1, 0])
        mm_44: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_308, permute_156);  permute_156 = None
        view_313: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_44, [32, 128, 2048]);  mm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_314: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_309, [32, 128, 16, 128]);  view_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_157: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_314, [0, 2, 1, 3]);  view_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_315: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_311, [32, 128, 16, 128]);  view_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_158: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_315, [0, 2, 1, 3]);  view_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_316: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_313, [32, 128, 16, 128]);  view_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_159: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_160: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(permute_158, [0, 1, 3, 2]);  permute_158 = None
        expand_58: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_157, [32, 16, 128, 128]);  permute_157 = None
        clone_99: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
        view_317: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_99, [512, 128, 128]);  clone_99 = None
        expand_59: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_160, [32, 16, 128, 128]);  permute_160 = None
        clone_100: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
        view_318: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_100, [512, 128, 128]);  clone_100 = None
        bmm_28: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_317, view_318)
        view_319: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_28, [32, 16, 128, 128]);  bmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_32: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_205, 2, 0, 128)
        slice_33: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_32, 3, 0, 128);  slice_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_15: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_33, view_319, full_default_2);  slice_33 = view_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_132: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_15, where);  where_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_14: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_132, [-1], True)
        sub_45: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_132, amax_14);  add_132 = amax_14 = None
        exp_14: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_45);  sub_45 = None
        sum_15: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_14: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_60: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_14, [32, 16, 128, 128])
        view_320: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_60, [512, 128, 128]);  expand_60 = None
        expand_61: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_159, [32, 16, 128, 128]);  permute_159 = None
        clone_102: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
        view_321: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_102, [512, 128, 128]);  clone_102 = None
        bmm_29: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_320, view_321);  view_320 = None
        view_322: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_29, [32, 16, 128, 128]);  bmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_161: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_322, [0, 2, 1, 3]);  view_322 = None
        clone_103: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_323: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_103, [32, 128, 2048]);  clone_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_324: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_323, [4096, 2048]);  view_323 = None
        permute_162: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_206, [1, 0])
        addmm_42: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_207, view_324, permute_162);  primals_207 = permute_162 = None
        view_325: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_42, [32, 128, 2048]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_133: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_325, add_129);  view_325 = add_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_29 = torch.ops.aten.var_mean.correction(add_133, [2], correction = 0, keepdim = True)
        getitem_58: "f32[32, 128, 1]" = var_mean_29[0]
        getitem_59: "f32[32, 128, 1]" = var_mean_29[1];  var_mean_29 = None
        add_134: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
        rsqrt_29: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        sub_46: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_133, getitem_59);  getitem_59 = None
        mul_114: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_29);  sub_46 = None
        mul_115: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_114, primals_208)
        add_135: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_115, primals_209);  mul_115 = primals_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_326: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_135, [4096, 2048]);  add_135 = None
        permute_163: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_210, [1, 0])
        addmm_43: "f32[4096, 8192]" = torch.ops.aten.addmm.default(primals_211, view_326, permute_163);  primals_211 = permute_163 = None
        view_327: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_43, [32, 128, 8192])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_116: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_327, 0.5)
        pow_15: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_327, 3.0)
        mul_117: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
        add_136: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_327, mul_117);  view_327 = mul_117 = None
        mul_118: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_136, 0.7978845608028654);  add_136 = None
        tanh_14: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_118);  mul_118 = None
        add_137: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_14, 1.0);  tanh_14 = None
        mul_119: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_116, add_137);  mul_116 = add_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_328: "f32[4096, 8192]" = torch.ops.aten.reshape.default(mul_119, [4096, 8192]);  mul_119 = None
        permute_164: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_212, [1, 0])
        addmm_44: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_213, view_328, permute_164);  primals_213 = permute_164 = None
        view_329: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_44, [32, 128, 2048]);  addmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_138: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_133, view_329);  add_133 = view_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_30 = torch.ops.aten.var_mean.correction(add_138, [2], correction = 0, keepdim = True)
        getitem_60: "f32[32, 128, 1]" = var_mean_30[0]
        getitem_61: "f32[32, 128, 1]" = var_mean_30[1];  var_mean_30 = None
        add_139: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
        rsqrt_30: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
        sub_47: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_138, getitem_61);  getitem_61 = None
        mul_120: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_30);  sub_47 = None
        mul_121: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_120, primals_214)
        add_140: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_121, primals_215);  mul_121 = primals_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_165: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_216, [1, 0])
        view_330: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_140, [4096, 2048]);  add_140 = None
        mm_45: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_330, permute_165);  permute_165 = None
        view_331: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_45, [32, 128, 2048]);  mm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_166: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_217, [1, 0])
        mm_46: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_330, permute_166);  permute_166 = None
        view_333: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_46, [32, 128, 2048]);  mm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_167: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_218, [1, 0])
        mm_47: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_330, permute_167);  permute_167 = None
        view_335: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_47, [32, 128, 2048]);  mm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_336: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_331, [32, 128, 16, 128]);  view_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_168: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_336, [0, 2, 1, 3]);  view_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_337: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_333, [32, 128, 16, 128]);  view_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_169: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_337, [0, 2, 1, 3]);  view_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_338: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_335, [32, 128, 16, 128]);  view_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_170: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_171: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(permute_169, [0, 1, 3, 2]);  permute_169 = None
        expand_62: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_168, [32, 16, 128, 128]);  permute_168 = None
        clone_106: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_62, memory_format = torch.contiguous_format);  expand_62 = None
        view_339: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_106, [512, 128, 128]);  clone_106 = None
        expand_63: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_171, [32, 16, 128, 128]);  permute_171 = None
        clone_107: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
        view_340: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_107, [512, 128, 128]);  clone_107 = None
        bmm_30: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_339, view_340)
        view_341: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_30, [32, 16, 128, 128]);  bmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_34: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_219, 2, 0, 128)
        slice_35: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_34, 3, 0, 128);  slice_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_16: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_35, view_341, full_default_2);  slice_35 = view_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_141: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_16, where);  where_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_15: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_141, [-1], True)
        sub_48: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_141, amax_15);  add_141 = amax_15 = None
        exp_15: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_48);  sub_48 = None
        sum_16: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_15: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_64: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_15, [32, 16, 128, 128])
        view_342: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_64, [512, 128, 128]);  expand_64 = None
        expand_65: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_170, [32, 16, 128, 128]);  permute_170 = None
        clone_109: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        view_343: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_109, [512, 128, 128]);  clone_109 = None
        bmm_31: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_342, view_343);  view_342 = None
        view_344: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_31, [32, 16, 128, 128]);  bmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_172: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None
        clone_110: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_345: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_110, [32, 128, 2048]);  clone_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_346: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_345, [4096, 2048]);  view_345 = None
        permute_173: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_220, [1, 0])
        addmm_45: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_221, view_346, permute_173);  primals_221 = permute_173 = None
        view_347: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_45, [32, 128, 2048]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_142: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_347, add_138);  view_347 = add_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_31 = torch.ops.aten.var_mean.correction(add_142, [2], correction = 0, keepdim = True)
        getitem_62: "f32[32, 128, 1]" = var_mean_31[0]
        getitem_63: "f32[32, 128, 1]" = var_mean_31[1];  var_mean_31 = None
        add_143: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
        rsqrt_31: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
        sub_49: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_142, getitem_63);  getitem_63 = None
        mul_122: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_31);  sub_49 = None
        mul_123: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_122, primals_222)
        add_144: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_123, primals_223);  mul_123 = primals_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_348: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_144, [4096, 2048]);  add_144 = None
        permute_174: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_224, [1, 0])
        addmm_46: "f32[4096, 8192]" = torch.ops.aten.addmm.default(primals_225, view_348, permute_174);  primals_225 = permute_174 = None
        view_349: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_46, [32, 128, 8192])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_124: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_349, 0.5)
        pow_16: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_349, 3.0)
        mul_125: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_16, 0.044715);  pow_16 = None
        add_145: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_349, mul_125);  view_349 = mul_125 = None
        mul_126: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_145, 0.7978845608028654);  add_145 = None
        tanh_15: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_126);  mul_126 = None
        add_146: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_15, 1.0);  tanh_15 = None
        mul_127: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_124, add_146);  mul_124 = add_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_350: "f32[4096, 8192]" = torch.ops.aten.reshape.default(mul_127, [4096, 8192]);  mul_127 = None
        permute_175: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_226, [1, 0])
        addmm_47: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_227, view_350, permute_175);  primals_227 = permute_175 = None
        view_351: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_47, [32, 128, 2048]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_147: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_142, view_351);  add_142 = view_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_32 = torch.ops.aten.var_mean.correction(add_147, [2], correction = 0, keepdim = True)
        getitem_64: "f32[32, 128, 1]" = var_mean_32[0]
        getitem_65: "f32[32, 128, 1]" = var_mean_32[1];  var_mean_32 = None
        add_148: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_32: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
        sub_50: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_147, getitem_65);  getitem_65 = None
        mul_128: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_32);  sub_50 = None
        mul_129: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_128, primals_228)
        add_149: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_129, primals_229);  mul_129 = primals_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_176: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_230, [1, 0])
        view_352: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_149, [4096, 2048]);  add_149 = None
        mm_48: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_352, permute_176);  permute_176 = None
        view_353: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_48, [32, 128, 2048]);  mm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_177: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_231, [1, 0])
        mm_49: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_352, permute_177);  permute_177 = None
        view_355: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_49, [32, 128, 2048]);  mm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_178: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_232, [1, 0])
        mm_50: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_352, permute_178);  permute_178 = None
        view_357: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_50, [32, 128, 2048]);  mm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_358: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_353, [32, 128, 16, 128]);  view_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_179: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_358, [0, 2, 1, 3]);  view_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_359: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_355, [32, 128, 16, 128]);  view_355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_180: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_359, [0, 2, 1, 3]);  view_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_360: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_357, [32, 128, 16, 128]);  view_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_181: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_360, [0, 2, 1, 3]);  view_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_182: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(permute_180, [0, 1, 3, 2]);  permute_180 = None
        expand_66: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_179, [32, 16, 128, 128]);  permute_179 = None
        clone_113: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
        view_361: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_113, [512, 128, 128]);  clone_113 = None
        expand_67: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_182, [32, 16, 128, 128]);  permute_182 = None
        clone_114: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
        view_362: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_114, [512, 128, 128]);  clone_114 = None
        bmm_32: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_361, view_362)
        view_363: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_32, [32, 16, 128, 128]);  bmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_36: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_233, 2, 0, 128)
        slice_37: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_36, 3, 0, 128);  slice_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_17: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_37, view_363, full_default_2);  slice_37 = view_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_150: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_17, where);  where_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_16: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_150, [-1], True)
        sub_51: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_150, amax_16);  add_150 = amax_16 = None
        exp_16: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_51);  sub_51 = None
        sum_17: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_16: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_68: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_16, [32, 16, 128, 128])
        view_364: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_68, [512, 128, 128]);  expand_68 = None
        expand_69: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_181, [32, 16, 128, 128]);  permute_181 = None
        clone_116: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
        view_365: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_116, [512, 128, 128]);  clone_116 = None
        bmm_33: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_364, view_365);  view_364 = None
        view_366: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_33, [32, 16, 128, 128]);  bmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_183: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_366, [0, 2, 1, 3]);  view_366 = None
        clone_117: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_367: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_117, [32, 128, 2048]);  clone_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_368: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_367, [4096, 2048]);  view_367 = None
        permute_184: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_234, [1, 0])
        addmm_48: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_235, view_368, permute_184);  primals_235 = permute_184 = None
        view_369: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_48, [32, 128, 2048]);  addmm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_151: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_369, add_147);  view_369 = add_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_33 = torch.ops.aten.var_mean.correction(add_151, [2], correction = 0, keepdim = True)
        getitem_66: "f32[32, 128, 1]" = var_mean_33[0]
        getitem_67: "f32[32, 128, 1]" = var_mean_33[1];  var_mean_33 = None
        add_152: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_33: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
        sub_52: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_151, getitem_67);  getitem_67 = None
        mul_130: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_33);  sub_52 = None
        mul_131: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_130, primals_236)
        add_153: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_131, primals_237);  mul_131 = primals_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_370: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_153, [4096, 2048]);  add_153 = None
        permute_185: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_238, [1, 0])
        addmm_49: "f32[4096, 8192]" = torch.ops.aten.addmm.default(primals_239, view_370, permute_185);  primals_239 = permute_185 = None
        view_371: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_49, [32, 128, 8192])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_132: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_371, 0.5)
        pow_17: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_371, 3.0)
        mul_133: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_17, 0.044715);  pow_17 = None
        add_154: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_371, mul_133);  view_371 = mul_133 = None
        mul_134: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_154, 0.7978845608028654);  add_154 = None
        tanh_16: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_134);  mul_134 = None
        add_155: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_16, 1.0);  tanh_16 = None
        mul_135: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_132, add_155);  mul_132 = add_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_372: "f32[4096, 8192]" = torch.ops.aten.reshape.default(mul_135, [4096, 8192]);  mul_135 = None
        permute_186: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_240, [1, 0])
        addmm_50: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_241, view_372, permute_186);  primals_241 = permute_186 = None
        view_373: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_50, [32, 128, 2048]);  addmm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_156: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_151, view_373);  add_151 = view_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_34 = torch.ops.aten.var_mean.correction(add_156, [2], correction = 0, keepdim = True)
        getitem_68: "f32[32, 128, 1]" = var_mean_34[0]
        getitem_69: "f32[32, 128, 1]" = var_mean_34[1];  var_mean_34 = None
        add_157: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
        rsqrt_34: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
        sub_53: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_156, getitem_69);  getitem_69 = None
        mul_136: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_34);  sub_53 = None
        mul_137: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_136, primals_242)
        add_158: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_137, primals_243);  mul_137 = primals_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_187: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_244, [1, 0])
        view_374: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_158, [4096, 2048]);  add_158 = None
        mm_51: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_374, permute_187);  permute_187 = None
        view_375: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_51, [32, 128, 2048]);  mm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_188: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_245, [1, 0])
        mm_52: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_374, permute_188);  permute_188 = None
        view_377: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_52, [32, 128, 2048]);  mm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_189: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_246, [1, 0])
        mm_53: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_374, permute_189);  permute_189 = None
        view_379: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_53, [32, 128, 2048]);  mm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_380: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_375, [32, 128, 16, 128]);  view_375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_190: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_380, [0, 2, 1, 3]);  view_380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_381: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_377, [32, 128, 16, 128]);  view_377 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_191: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_381, [0, 2, 1, 3]);  view_381 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_382: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_379, [32, 128, 16, 128]);  view_379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_192: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_382, [0, 2, 1, 3]);  view_382 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_193: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(permute_191, [0, 1, 3, 2]);  permute_191 = None
        expand_70: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_190, [32, 16, 128, 128]);  permute_190 = None
        clone_120: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_70, memory_format = torch.contiguous_format);  expand_70 = None
        view_383: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_120, [512, 128, 128]);  clone_120 = None
        expand_71: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_193, [32, 16, 128, 128]);  permute_193 = None
        clone_121: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
        view_384: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_121, [512, 128, 128]);  clone_121 = None
        bmm_34: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_383, view_384)
        view_385: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_34, [32, 16, 128, 128]);  bmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_38: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_247, 2, 0, 128)
        slice_39: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_38, 3, 0, 128);  slice_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_18: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_39, view_385, full_default_2);  slice_39 = view_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_159: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_18, where);  where_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_17: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_159, [-1], True)
        sub_54: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_159, amax_17);  add_159 = amax_17 = None
        exp_17: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_54);  sub_54 = None
        sum_18: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_17: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_72: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_17, [32, 16, 128, 128])
        view_386: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_72, [512, 128, 128]);  expand_72 = None
        expand_73: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_192, [32, 16, 128, 128]);  permute_192 = None
        clone_123: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
        view_387: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_123, [512, 128, 128]);  clone_123 = None
        bmm_35: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_386, view_387);  view_386 = None
        view_388: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_35, [32, 16, 128, 128]);  bmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_194: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_388, [0, 2, 1, 3]);  view_388 = None
        clone_124: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_389: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_124, [32, 128, 2048]);  clone_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_390: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_389, [4096, 2048]);  view_389 = None
        permute_195: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_248, [1, 0])
        addmm_51: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_249, view_390, permute_195);  primals_249 = permute_195 = None
        view_391: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_51, [32, 128, 2048]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_160: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_391, add_156);  view_391 = add_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_35 = torch.ops.aten.var_mean.correction(add_160, [2], correction = 0, keepdim = True)
        getitem_70: "f32[32, 128, 1]" = var_mean_35[0]
        getitem_71: "f32[32, 128, 1]" = var_mean_35[1];  var_mean_35 = None
        add_161: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
        rsqrt_35: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
        sub_55: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_160, getitem_71);  getitem_71 = None
        mul_138: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_35);  sub_55 = None
        mul_139: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_138, primals_250)
        add_162: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_139, primals_251);  mul_139 = primals_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_392: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_162, [4096, 2048]);  add_162 = None
        permute_196: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_252, [1, 0])
        addmm_52: "f32[4096, 8192]" = torch.ops.aten.addmm.default(primals_253, view_392, permute_196);  primals_253 = permute_196 = None
        view_393: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_52, [32, 128, 8192])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_140: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_393, 0.5)
        pow_18: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_393, 3.0)
        mul_141: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_18, 0.044715);  pow_18 = None
        add_163: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_393, mul_141);  view_393 = mul_141 = None
        mul_142: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_163, 0.7978845608028654);  add_163 = None
        tanh_17: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_142);  mul_142 = None
        add_164: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_17, 1.0);  tanh_17 = None
        mul_143: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_140, add_164);  mul_140 = add_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_394: "f32[4096, 8192]" = torch.ops.aten.reshape.default(mul_143, [4096, 8192]);  mul_143 = None
        permute_197: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_254, [1, 0])
        addmm_53: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_255, view_394, permute_197);  primals_255 = permute_197 = None
        view_395: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_53, [32, 128, 2048]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_165: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_160, view_395);  add_160 = view_395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_36 = torch.ops.aten.var_mean.correction(add_165, [2], correction = 0, keepdim = True)
        getitem_72: "f32[32, 128, 1]" = var_mean_36[0]
        getitem_73: "f32[32, 128, 1]" = var_mean_36[1];  var_mean_36 = None
        add_166: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
        rsqrt_36: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
        sub_56: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_165, getitem_73);  getitem_73 = None
        mul_144: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_36);  sub_56 = None
        mul_145: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_144, primals_256)
        add_167: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_145, primals_257);  mul_145 = primals_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_198: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_258, [1, 0])
        view_396: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_167, [4096, 2048]);  add_167 = None
        mm_54: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_396, permute_198);  permute_198 = None
        view_397: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_54, [32, 128, 2048]);  mm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_199: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_259, [1, 0])
        mm_55: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_396, permute_199);  permute_199 = None
        view_399: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_55, [32, 128, 2048]);  mm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_200: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_260, [1, 0])
        mm_56: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_396, permute_200);  permute_200 = None
        view_401: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_56, [32, 128, 2048]);  mm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_402: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_397, [32, 128, 16, 128]);  view_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_201: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_403: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_399, [32, 128, 16, 128]);  view_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_202: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_403, [0, 2, 1, 3]);  view_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_404: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_401, [32, 128, 16, 128]);  view_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_203: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_404, [0, 2, 1, 3]);  view_404 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_204: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(permute_202, [0, 1, 3, 2]);  permute_202 = None
        expand_74: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_201, [32, 16, 128, 128]);  permute_201 = None
        clone_127: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
        view_405: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_127, [512, 128, 128]);  clone_127 = None
        expand_75: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_204, [32, 16, 128, 128]);  permute_204 = None
        clone_128: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
        view_406: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_128, [512, 128, 128]);  clone_128 = None
        bmm_36: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_405, view_406)
        view_407: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_36, [32, 16, 128, 128]);  bmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_40: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_261, 2, 0, 128)
        slice_41: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_40, 3, 0, 128);  slice_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_19: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_41, view_407, full_default_2);  slice_41 = view_407 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_168: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_19, where);  where_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_18: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_168, [-1], True)
        sub_57: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_168, amax_18);  add_168 = amax_18 = None
        exp_18: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_57);  sub_57 = None
        sum_19: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_18: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_76: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_18, [32, 16, 128, 128])
        view_408: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_76, [512, 128, 128]);  expand_76 = None
        expand_77: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_203, [32, 16, 128, 128]);  permute_203 = None
        clone_130: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
        view_409: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_130, [512, 128, 128]);  clone_130 = None
        bmm_37: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_408, view_409);  view_408 = None
        view_410: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_37, [32, 16, 128, 128]);  bmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_205: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_410, [0, 2, 1, 3]);  view_410 = None
        clone_131: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_411: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_131, [32, 128, 2048]);  clone_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_412: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_411, [4096, 2048]);  view_411 = None
        permute_206: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_262, [1, 0])
        addmm_54: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_263, view_412, permute_206);  primals_263 = permute_206 = None
        view_413: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_54, [32, 128, 2048]);  addmm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_169: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_413, add_165);  view_413 = add_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_37 = torch.ops.aten.var_mean.correction(add_169, [2], correction = 0, keepdim = True)
        getitem_74: "f32[32, 128, 1]" = var_mean_37[0]
        getitem_75: "f32[32, 128, 1]" = var_mean_37[1];  var_mean_37 = None
        add_170: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
        rsqrt_37: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        sub_58: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_169, getitem_75);  getitem_75 = None
        mul_146: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_37);  sub_58 = None
        mul_147: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_146, primals_264)
        add_171: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_147, primals_265);  mul_147 = primals_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_414: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_171, [4096, 2048]);  add_171 = None
        permute_207: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_266, [1, 0])
        addmm_55: "f32[4096, 8192]" = torch.ops.aten.addmm.default(primals_267, view_414, permute_207);  primals_267 = permute_207 = None
        view_415: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_55, [32, 128, 8192])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_148: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_415, 0.5)
        pow_19: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_415, 3.0)
        mul_149: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_19, 0.044715);  pow_19 = None
        add_172: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_415, mul_149);  view_415 = mul_149 = None
        mul_150: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_172, 0.7978845608028654);  add_172 = None
        tanh_18: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_150);  mul_150 = None
        add_173: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_18, 1.0);  tanh_18 = None
        mul_151: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_148, add_173);  mul_148 = add_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_416: "f32[4096, 8192]" = torch.ops.aten.reshape.default(mul_151, [4096, 8192]);  mul_151 = None
        permute_208: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_268, [1, 0])
        addmm_56: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_269, view_416, permute_208);  primals_269 = permute_208 = None
        view_417: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_56, [32, 128, 2048]);  addmm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_174: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_169, view_417);  add_169 = view_417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_38 = torch.ops.aten.var_mean.correction(add_174, [2], correction = 0, keepdim = True)
        getitem_76: "f32[32, 128, 1]" = var_mean_38[0]
        getitem_77: "f32[32, 128, 1]" = var_mean_38[1];  var_mean_38 = None
        add_175: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
        rsqrt_38: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
        sub_59: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_174, getitem_77);  getitem_77 = None
        mul_152: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_38);  sub_59 = None
        mul_153: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_152, primals_270)
        add_176: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_153, primals_271);  mul_153 = primals_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_209: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_272, [1, 0])
        view_418: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_176, [4096, 2048]);  add_176 = None
        mm_57: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_418, permute_209);  permute_209 = None
        view_419: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_57, [32, 128, 2048]);  mm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_210: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_273, [1, 0])
        mm_58: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_418, permute_210);  permute_210 = None
        view_421: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_58, [32, 128, 2048]);  mm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_211: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_274, [1, 0])
        mm_59: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_418, permute_211);  permute_211 = None
        view_423: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_59, [32, 128, 2048]);  mm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_424: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_419, [32, 128, 16, 128]);  view_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_212: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_424, [0, 2, 1, 3]);  view_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_425: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_421, [32, 128, 16, 128]);  view_421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_213: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_425, [0, 2, 1, 3]);  view_425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_426: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_423, [32, 128, 16, 128]);  view_423 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_214: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_426, [0, 2, 1, 3]);  view_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_215: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(permute_213, [0, 1, 3, 2]);  permute_213 = None
        expand_78: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_212, [32, 16, 128, 128]);  permute_212 = None
        clone_134: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_78, memory_format = torch.contiguous_format);  expand_78 = None
        view_427: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_134, [512, 128, 128]);  clone_134 = None
        expand_79: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_215, [32, 16, 128, 128]);  permute_215 = None
        clone_135: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_79, memory_format = torch.contiguous_format);  expand_79 = None
        view_428: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_135, [512, 128, 128]);  clone_135 = None
        bmm_38: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_427, view_428)
        view_429: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_38, [32, 16, 128, 128]);  bmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_42: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_275, 2, 0, 128)
        slice_43: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_42, 3, 0, 128);  slice_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_20: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_43, view_429, full_default_2);  slice_43 = view_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_177: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_20, where);  where_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_19: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_177, [-1], True)
        sub_60: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_177, amax_19);  add_177 = amax_19 = None
        exp_19: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_60);  sub_60 = None
        sum_20: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_19: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_80: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_19, [32, 16, 128, 128])
        view_430: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_80, [512, 128, 128]);  expand_80 = None
        expand_81: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_214, [32, 16, 128, 128]);  permute_214 = None
        clone_137: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
        view_431: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_137, [512, 128, 128]);  clone_137 = None
        bmm_39: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_430, view_431);  view_430 = None
        view_432: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_39, [32, 16, 128, 128]);  bmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_216: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_432, [0, 2, 1, 3]);  view_432 = None
        clone_138: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_433: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_138, [32, 128, 2048]);  clone_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_434: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_433, [4096, 2048]);  view_433 = None
        permute_217: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_276, [1, 0])
        addmm_57: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_277, view_434, permute_217);  primals_277 = permute_217 = None
        view_435: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_57, [32, 128, 2048]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_178: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_435, add_174);  view_435 = add_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_39 = torch.ops.aten.var_mean.correction(add_178, [2], correction = 0, keepdim = True)
        getitem_78: "f32[32, 128, 1]" = var_mean_39[0]
        getitem_79: "f32[32, 128, 1]" = var_mean_39[1];  var_mean_39 = None
        add_179: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
        rsqrt_39: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_179);  add_179 = None
        sub_61: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_178, getitem_79);  getitem_79 = None
        mul_154: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_39);  sub_61 = None
        mul_155: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_154, primals_278)
        add_180: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_155, primals_279);  mul_155 = primals_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_436: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_180, [4096, 2048]);  add_180 = None
        permute_218: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_280, [1, 0])
        addmm_58: "f32[4096, 8192]" = torch.ops.aten.addmm.default(primals_281, view_436, permute_218);  primals_281 = permute_218 = None
        view_437: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_58, [32, 128, 8192])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_156: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_437, 0.5)
        pow_20: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_437, 3.0)
        mul_157: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_20, 0.044715);  pow_20 = None
        add_181: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_437, mul_157);  view_437 = mul_157 = None
        mul_158: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_181, 0.7978845608028654);  add_181 = None
        tanh_19: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_158);  mul_158 = None
        add_182: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_19, 1.0);  tanh_19 = None
        mul_159: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_156, add_182);  mul_156 = add_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_438: "f32[4096, 8192]" = torch.ops.aten.reshape.default(mul_159, [4096, 8192]);  mul_159 = None
        permute_219: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_282, [1, 0])
        addmm_59: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_283, view_438, permute_219);  primals_283 = permute_219 = None
        view_439: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_59, [32, 128, 2048]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_183: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_178, view_439);  add_178 = view_439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_40 = torch.ops.aten.var_mean.correction(add_183, [2], correction = 0, keepdim = True)
        getitem_80: "f32[32, 128, 1]" = var_mean_40[0]
        getitem_81: "f32[32, 128, 1]" = var_mean_40[1];  var_mean_40 = None
        add_184: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
        rsqrt_40: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_184);  add_184 = None
        sub_62: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_183, getitem_81);  getitem_81 = None
        mul_160: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_40);  sub_62 = None
        mul_161: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_160, primals_284)
        add_185: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_161, primals_285);  mul_161 = primals_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_220: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_286, [1, 0])
        view_440: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_185, [4096, 2048]);  add_185 = None
        mm_60: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_440, permute_220);  permute_220 = None
        view_441: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_60, [32, 128, 2048]);  mm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_221: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_287, [1, 0])
        mm_61: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_440, permute_221);  permute_221 = None
        view_443: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_61, [32, 128, 2048]);  mm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_222: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_288, [1, 0])
        mm_62: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_440, permute_222);  permute_222 = None
        view_445: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_62, [32, 128, 2048]);  mm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_446: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_441, [32, 128, 16, 128]);  view_441 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_223: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_446, [0, 2, 1, 3]);  view_446 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_447: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_443, [32, 128, 16, 128]);  view_443 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_224: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_447, [0, 2, 1, 3]);  view_447 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_448: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_445, [32, 128, 16, 128]);  view_445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_225: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_226: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(permute_224, [0, 1, 3, 2]);  permute_224 = None
        expand_82: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_223, [32, 16, 128, 128]);  permute_223 = None
        clone_141: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_82, memory_format = torch.contiguous_format);  expand_82 = None
        view_449: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_141, [512, 128, 128]);  clone_141 = None
        expand_83: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_226, [32, 16, 128, 128]);  permute_226 = None
        clone_142: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_83, memory_format = torch.contiguous_format);  expand_83 = None
        view_450: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_142, [512, 128, 128]);  clone_142 = None
        bmm_40: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_449, view_450)
        view_451: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_40, [32, 16, 128, 128]);  bmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_44: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_289, 2, 0, 128)
        slice_45: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_44, 3, 0, 128);  slice_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_21: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_45, view_451, full_default_2);  slice_45 = view_451 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_186: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_21, where);  where_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_20: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_186, [-1], True)
        sub_63: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_186, amax_20);  add_186 = amax_20 = None
        exp_20: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_63);  sub_63 = None
        sum_21: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_20: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_84: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_20, [32, 16, 128, 128])
        view_452: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_84, [512, 128, 128]);  expand_84 = None
        expand_85: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_225, [32, 16, 128, 128]);  permute_225 = None
        clone_144: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_85, memory_format = torch.contiguous_format);  expand_85 = None
        view_453: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_144, [512, 128, 128]);  clone_144 = None
        bmm_41: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_452, view_453);  view_452 = None
        view_454: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_41, [32, 16, 128, 128]);  bmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_227: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_454, [0, 2, 1, 3]);  view_454 = None
        clone_145: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_455: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_145, [32, 128, 2048]);  clone_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_456: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_455, [4096, 2048]);  view_455 = None
        permute_228: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_290, [1, 0])
        addmm_60: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_291, view_456, permute_228);  primals_291 = permute_228 = None
        view_457: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_60, [32, 128, 2048]);  addmm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_187: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_457, add_183);  view_457 = add_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_41 = torch.ops.aten.var_mean.correction(add_187, [2], correction = 0, keepdim = True)
        getitem_82: "f32[32, 128, 1]" = var_mean_41[0]
        getitem_83: "f32[32, 128, 1]" = var_mean_41[1];  var_mean_41 = None
        add_188: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
        rsqrt_41: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
        sub_64: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_187, getitem_83);  getitem_83 = None
        mul_162: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_41);  sub_64 = None
        mul_163: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_162, primals_292)
        add_189: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_163, primals_293);  mul_163 = primals_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_458: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_189, [4096, 2048]);  add_189 = None
        permute_229: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_294, [1, 0])
        addmm_61: "f32[4096, 8192]" = torch.ops.aten.addmm.default(primals_295, view_458, permute_229);  primals_295 = permute_229 = None
        view_459: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_61, [32, 128, 8192])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_164: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_459, 0.5)
        pow_21: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_459, 3.0)
        mul_165: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
        add_190: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_459, mul_165);  view_459 = mul_165 = None
        mul_166: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_190, 0.7978845608028654);  add_190 = None
        tanh_20: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_166);  mul_166 = None
        add_191: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_20, 1.0);  tanh_20 = None
        mul_167: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_164, add_191);  mul_164 = add_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_460: "f32[4096, 8192]" = torch.ops.aten.reshape.default(mul_167, [4096, 8192]);  mul_167 = None
        permute_230: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_296, [1, 0])
        addmm_62: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_297, view_460, permute_230);  primals_297 = permute_230 = None
        view_461: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_62, [32, 128, 2048]);  addmm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_192: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_187, view_461);  add_187 = view_461 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_42 = torch.ops.aten.var_mean.correction(add_192, [2], correction = 0, keepdim = True)
        getitem_84: "f32[32, 128, 1]" = var_mean_42[0]
        getitem_85: "f32[32, 128, 1]" = var_mean_42[1];  var_mean_42 = None
        add_193: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
        rsqrt_42: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_193);  add_193 = None
        sub_65: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_192, getitem_85);  getitem_85 = None
        mul_168: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_42);  sub_65 = None
        mul_169: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_168, primals_298)
        add_194: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_169, primals_299);  mul_169 = primals_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_231: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_300, [1, 0])
        view_462: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_194, [4096, 2048]);  add_194 = None
        mm_63: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_462, permute_231);  permute_231 = None
        view_463: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_63, [32, 128, 2048]);  mm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_232: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_301, [1, 0])
        mm_64: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_462, permute_232);  permute_232 = None
        view_465: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_64, [32, 128, 2048]);  mm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_233: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_302, [1, 0])
        mm_65: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_462, permute_233);  permute_233 = None
        view_467: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_65, [32, 128, 2048]);  mm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_468: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_463, [32, 128, 16, 128]);  view_463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_234: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_468, [0, 2, 1, 3]);  view_468 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_469: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_465, [32, 128, 16, 128]);  view_465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_235: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_469, [0, 2, 1, 3]);  view_469 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_470: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_467, [32, 128, 16, 128]);  view_467 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_236: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_470, [0, 2, 1, 3]);  view_470 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_237: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(permute_235, [0, 1, 3, 2]);  permute_235 = None
        expand_86: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_234, [32, 16, 128, 128]);  permute_234 = None
        clone_148: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_86, memory_format = torch.contiguous_format);  expand_86 = None
        view_471: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_148, [512, 128, 128]);  clone_148 = None
        expand_87: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_237, [32, 16, 128, 128]);  permute_237 = None
        clone_149: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_87, memory_format = torch.contiguous_format);  expand_87 = None
        view_472: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_149, [512, 128, 128]);  clone_149 = None
        bmm_42: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_471, view_472)
        view_473: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_42, [32, 16, 128, 128]);  bmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_46: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_303, 2, 0, 128)
        slice_47: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_46, 3, 0, 128);  slice_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_22: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_47, view_473, full_default_2);  slice_47 = view_473 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_195: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_22, where);  where_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_21: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_195, [-1], True)
        sub_66: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_195, amax_21);  add_195 = amax_21 = None
        exp_21: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_66);  sub_66 = None
        sum_22: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_21: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_88: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_21, [32, 16, 128, 128])
        view_474: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_88, [512, 128, 128]);  expand_88 = None
        expand_89: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_236, [32, 16, 128, 128]);  permute_236 = None
        clone_151: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
        view_475: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_151, [512, 128, 128]);  clone_151 = None
        bmm_43: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_474, view_475);  view_474 = None
        view_476: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_43, [32, 16, 128, 128]);  bmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_238: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_476, [0, 2, 1, 3]);  view_476 = None
        clone_152: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_477: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_152, [32, 128, 2048]);  clone_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_478: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_477, [4096, 2048]);  view_477 = None
        permute_239: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_304, [1, 0])
        addmm_63: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_305, view_478, permute_239);  primals_305 = permute_239 = None
        view_479: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_63, [32, 128, 2048]);  addmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_196: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_479, add_192);  view_479 = add_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_43 = torch.ops.aten.var_mean.correction(add_196, [2], correction = 0, keepdim = True)
        getitem_86: "f32[32, 128, 1]" = var_mean_43[0]
        getitem_87: "f32[32, 128, 1]" = var_mean_43[1];  var_mean_43 = None
        add_197: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
        rsqrt_43: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
        sub_67: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_196, getitem_87);  getitem_87 = None
        mul_170: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_43);  sub_67 = None
        mul_171: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_170, primals_306)
        add_198: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_171, primals_307);  mul_171 = primals_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_480: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_198, [4096, 2048]);  add_198 = None
        permute_240: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_308, [1, 0])
        addmm_64: "f32[4096, 8192]" = torch.ops.aten.addmm.default(primals_309, view_480, permute_240);  primals_309 = permute_240 = None
        view_481: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_64, [32, 128, 8192])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_172: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_481, 0.5)
        pow_22: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_481, 3.0)
        mul_173: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_22, 0.044715);  pow_22 = None
        add_199: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_481, mul_173);  view_481 = mul_173 = None
        mul_174: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_199, 0.7978845608028654);  add_199 = None
        tanh_21: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_174);  mul_174 = None
        add_200: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_21, 1.0);  tanh_21 = None
        mul_175: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_172, add_200);  mul_172 = add_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_482: "f32[4096, 8192]" = torch.ops.aten.reshape.default(mul_175, [4096, 8192]);  mul_175 = None
        permute_241: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_310, [1, 0])
        addmm_65: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_311, view_482, permute_241);  primals_311 = permute_241 = None
        view_483: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_65, [32, 128, 2048]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_201: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_196, view_483);  add_196 = view_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_44 = torch.ops.aten.var_mean.correction(add_201, [2], correction = 0, keepdim = True)
        getitem_88: "f32[32, 128, 1]" = var_mean_44[0]
        getitem_89: "f32[32, 128, 1]" = var_mean_44[1];  var_mean_44 = None
        add_202: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_44: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_202);  add_202 = None
        sub_68: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_201, getitem_89);  getitem_89 = None
        mul_176: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_44);  sub_68 = None
        mul_177: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_176, primals_312)
        add_203: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_177, primals_313);  mul_177 = primals_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_242: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_314, [1, 0])
        view_484: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_203, [4096, 2048]);  add_203 = None
        mm_66: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_484, permute_242);  permute_242 = None
        view_485: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_66, [32, 128, 2048]);  mm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_243: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_315, [1, 0])
        mm_67: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_484, permute_243);  permute_243 = None
        view_487: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_67, [32, 128, 2048]);  mm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_244: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_316, [1, 0])
        mm_68: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_484, permute_244);  permute_244 = None
        view_489: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_68, [32, 128, 2048]);  mm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_490: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_485, [32, 128, 16, 128]);  view_485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_245: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_490, [0, 2, 1, 3]);  view_490 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_491: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_487, [32, 128, 16, 128]);  view_487 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_246: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_491, [0, 2, 1, 3]);  view_491 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_492: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_489, [32, 128, 16, 128]);  view_489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_247: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_492, [0, 2, 1, 3]);  view_492 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_248: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(permute_246, [0, 1, 3, 2]);  permute_246 = None
        expand_90: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_245, [32, 16, 128, 128]);  permute_245 = None
        clone_155: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_90, memory_format = torch.contiguous_format);  expand_90 = None
        view_493: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_155, [512, 128, 128]);  clone_155 = None
        expand_91: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_248, [32, 16, 128, 128]);  permute_248 = None
        clone_156: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_91, memory_format = torch.contiguous_format);  expand_91 = None
        view_494: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_156, [512, 128, 128]);  clone_156 = None
        bmm_44: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_493, view_494)
        view_495: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_44, [32, 16, 128, 128]);  bmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_48: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_317, 2, 0, 128)
        slice_49: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_48, 3, 0, 128);  slice_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_23: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_49, view_495, full_default_2);  slice_49 = view_495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_204: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_23, where);  where_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_22: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_204, [-1], True)
        sub_69: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_204, amax_22);  add_204 = amax_22 = None
        exp_22: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_69);  sub_69 = None
        sum_23: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_22: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_92: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_22, [32, 16, 128, 128])
        view_496: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_92, [512, 128, 128]);  expand_92 = None
        expand_93: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_247, [32, 16, 128, 128]);  permute_247 = None
        clone_158: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_93, memory_format = torch.contiguous_format);  expand_93 = None
        view_497: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_158, [512, 128, 128]);  clone_158 = None
        bmm_45: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_496, view_497);  view_496 = None
        view_498: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_45, [32, 16, 128, 128]);  bmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_249: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_498, [0, 2, 1, 3]);  view_498 = None
        clone_159: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_499: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_159, [32, 128, 2048]);  clone_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_500: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_499, [4096, 2048]);  view_499 = None
        permute_250: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_318, [1, 0])
        addmm_66: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_319, view_500, permute_250);  primals_319 = permute_250 = None
        view_501: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_66, [32, 128, 2048]);  addmm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_205: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_501, add_201);  view_501 = add_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_45 = torch.ops.aten.var_mean.correction(add_205, [2], correction = 0, keepdim = True)
        getitem_90: "f32[32, 128, 1]" = var_mean_45[0]
        getitem_91: "f32[32, 128, 1]" = var_mean_45[1];  var_mean_45 = None
        add_206: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
        rsqrt_45: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_206);  add_206 = None
        sub_70: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_205, getitem_91);  getitem_91 = None
        mul_178: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_45);  sub_70 = None
        mul_179: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_178, primals_320)
        add_207: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_179, primals_321);  mul_179 = primals_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_502: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_207, [4096, 2048]);  add_207 = None
        permute_251: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_322, [1, 0])
        addmm_67: "f32[4096, 8192]" = torch.ops.aten.addmm.default(primals_323, view_502, permute_251);  primals_323 = permute_251 = None
        view_503: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_67, [32, 128, 8192])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_180: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_503, 0.5)
        pow_23: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_503, 3.0)
        mul_181: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_23, 0.044715);  pow_23 = None
        add_208: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_503, mul_181);  view_503 = mul_181 = None
        mul_182: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_208, 0.7978845608028654);  add_208 = None
        tanh_22: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_182);  mul_182 = None
        add_209: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_22, 1.0);  tanh_22 = None
        mul_183: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_180, add_209);  mul_180 = add_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_504: "f32[4096, 8192]" = torch.ops.aten.reshape.default(mul_183, [4096, 8192]);  mul_183 = None
        permute_252: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_324, [1, 0])
        addmm_68: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_325, view_504, permute_252);  primals_325 = permute_252 = None
        view_505: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_68, [32, 128, 2048]);  addmm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_210: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_205, view_505);  add_205 = view_505 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_46 = torch.ops.aten.var_mean.correction(add_210, [2], correction = 0, keepdim = True)
        getitem_92: "f32[32, 128, 1]" = var_mean_46[0]
        getitem_93: "f32[32, 128, 1]" = var_mean_46[1];  var_mean_46 = None
        add_211: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
        rsqrt_46: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_211);  add_211 = None
        sub_71: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_210, getitem_93);  getitem_93 = None
        mul_184: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_46);  sub_71 = None
        mul_185: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_184, primals_326)
        add_212: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_185, primals_327);  mul_185 = primals_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_253: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_328, [1, 0])
        view_506: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_212, [4096, 2048]);  add_212 = None
        mm_69: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_506, permute_253);  permute_253 = None
        view_507: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_69, [32, 128, 2048]);  mm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_254: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_329, [1, 0])
        mm_70: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_506, permute_254);  permute_254 = None
        view_509: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_70, [32, 128, 2048]);  mm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        permute_255: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_330, [1, 0])
        mm_71: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_506, permute_255);  permute_255 = None
        view_511: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_71, [32, 128, 2048]);  mm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_512: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_507, [32, 128, 16, 128]);  view_507 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_256: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_512, [0, 2, 1, 3]);  view_512 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_513: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_509, [32, 128, 16, 128]);  view_509 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_257: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_513, [0, 2, 1, 3]);  view_513 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_514: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_511, [32, 128, 16, 128]);  view_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_258: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_514, [0, 2, 1, 3]);  view_514 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_259: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(permute_257, [0, 1, 3, 2]);  permute_257 = None
        expand_94: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_256, [32, 16, 128, 128]);  permute_256 = None
        clone_162: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_94, memory_format = torch.contiguous_format);  expand_94 = None
        view_515: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_162, [512, 128, 128]);  clone_162 = None
        expand_95: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_259, [32, 16, 128, 128]);  permute_259 = None
        clone_163: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_95, memory_format = torch.contiguous_format);  expand_95 = None
        view_516: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_163, [512, 128, 128]);  clone_163 = None
        bmm_46: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_515, view_516)
        view_517: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_46, [32, 16, 128, 128]);  bmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_50: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_331, 2, 0, 128)
        slice_51: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_50, 3, 0, 128);  slice_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_24: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_51, view_517, full_default_2);  slice_51 = view_517 = full_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_213: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_24, where);  where_24 = where = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_23: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_213, [-1], True)
        sub_72: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_213, amax_23);  add_213 = amax_23 = None
        exp_23: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_72);  sub_72 = None
        sum_24: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_23: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_96: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_23, [32, 16, 128, 128])
        view_518: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_96, [512, 128, 128]);  expand_96 = None
        expand_97: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_258, [32, 16, 128, 128]);  permute_258 = None
        clone_165: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_97, memory_format = torch.contiguous_format);  expand_97 = None
        view_519: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_165, [512, 128, 128]);  clone_165 = None
        bmm_47: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_518, view_519);  view_518 = None
        view_520: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_47, [32, 16, 128, 128]);  bmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_260: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_520, [0, 2, 1, 3]);  view_520 = None
        clone_166: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_521: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_166, [32, 128, 2048]);  clone_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_522: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_521, [4096, 2048]);  view_521 = None
        permute_261: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_332, [1, 0])
        addmm_69: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_333, view_522, permute_261);  primals_333 = permute_261 = None
        view_523: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_69, [32, 128, 2048]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_214: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_523, add_210);  view_523 = add_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_47 = torch.ops.aten.var_mean.correction(add_214, [2], correction = 0, keepdim = True)
        getitem_94: "f32[32, 128, 1]" = var_mean_47[0]
        getitem_95: "f32[32, 128, 1]" = var_mean_47[1];  var_mean_47 = None
        add_215: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
        rsqrt_47: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_215);  add_215 = None
        sub_73: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_214, getitem_95);  getitem_95 = None
        mul_186: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_47);  sub_73 = None
        mul_187: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_186, primals_334)
        add_216: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_187, primals_335);  mul_187 = primals_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_524: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_216, [4096, 2048]);  add_216 = None
        permute_262: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_336, [1, 0])
        addmm_70: "f32[4096, 8192]" = torch.ops.aten.addmm.default(primals_337, view_524, permute_262);  primals_337 = permute_262 = None
        view_525: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_70, [32, 128, 8192])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_188: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_525, 0.5)
        pow_24: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_525, 3.0)
        mul_189: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_24, 0.044715);  pow_24 = None
        add_217: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_525, mul_189);  view_525 = mul_189 = None
        mul_190: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_217, 0.7978845608028654);  add_217 = None
        tanh_23: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_190);  mul_190 = None
        add_218: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_23, 1.0);  tanh_23 = None
        mul_191: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_188, add_218);  mul_188 = add_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_526: "f32[4096, 8192]" = torch.ops.aten.reshape.default(mul_191, [4096, 8192]);  mul_191 = None
        permute_263: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_338, [1, 0])
        addmm_71: "f32[4096, 2048]" = torch.ops.aten.addmm.default(primals_339, view_526, permute_263);  primals_339 = permute_263 = None
        view_527: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_71, [32, 128, 2048]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_219: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_214, view_527);  add_214 = view_527 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:492 in forward, code: hidden_states = self.ln_f(hidden_states)
        var_mean_48 = torch.ops.aten.var_mean.correction(add_219, [2], correction = 0, keepdim = True)
        getitem_96: "f32[32, 128, 1]" = var_mean_48[0]
        getitem_97: "f32[32, 128, 1]" = var_mean_48[1];  var_mean_48 = None
        add_220: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
        rsqrt_48: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_220);  add_220 = None
        sub_74: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_219, getitem_97);  add_219 = getitem_97 = None
        mul_192: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_48);  sub_74 = None
        mul_193: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_192, primals_340)
        add_221: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_193, primals_341);  mul_193 = primals_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:583 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        permute_264: "f32[2048, 50257]" = torch.ops.aten.permute.default(primals_2, [1, 0])
        view_529: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_221, [4096, 2048]);  add_221 = None
        constant_pad_nd_default_3: "f32[2048, 50260]" = torch.ops.aten.constant_pad_nd.default(permute_264, [0, 3, 0, 0]);  permute_264 = None
        mm_default_2: "f32[4096, 50260]" = torch.ops.aten.mm.default(view_529, constant_pad_nd_default_3);  constant_pad_nd_default_3 = None
        slice_tensor_1: "f32[4096, 50257]" = torch.ops.aten.slice.Tensor(mm_default_2, 1, 0, -3);  mm_default_2 = None
        view_530: "f32[32, 128, 50257]" = torch.ops.aten.reshape.default(slice_tensor_1, [32, 128, 50257]);  slice_tensor_1 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "i64[32, 129]" = torch.ops.aten.constant_pad_nd.default(primals_342, [0, 1], -100.0);  primals_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:61 in ForCausalLMLoss, code: shift_labels = labels[..., 1:].contiguous()
        slice_52: "i64[32, 128]" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807)
        clone_169: "i64[32, 128]" = torch.ops.aten.clone.default(slice_52, memory_format = torch.contiguous_format);  slice_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_531: "f32[4096, 50257]" = torch.ops.aten.reshape.default(view_530, [-1, 50257])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:65 in ForCausalLMLoss, code: shift_labels = shift_labels.view(-1)
        view_532: "i64[4096]" = torch.ops.aten.reshape.default(clone_169, [-1]);  clone_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        amax_24: "f32[4096, 1]" = torch.ops.aten.amax.default(view_531, [1], True)
        sub_75: "f32[4096, 50257]" = torch.ops.aten.sub.Tensor(view_531, amax_24);  view_531 = None
        exp_24: "f32[4096, 50257]" = torch.ops.aten.exp.default(sub_75)
        sum_25: "f32[4096, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log: "f32[4096, 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_76: "f32[4096, 50257]" = torch.ops.aten.sub.Tensor(sub_75, log);  sub_75 = None
        ne_1: "b8[4096]" = torch.ops.aten.ne.Scalar(view_532, -100)
        full_default_27: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_25: "i64[4096]" = torch.ops.aten.where.self(ne_1, view_532, full_default_27);  view_532 = full_default_27 = None
        unsqueeze_10: "i64[4096, 1]" = torch.ops.aten.unsqueeze.default(where_25, 1);  where_25 = None
        gather: "f32[4096, 1]" = torch.ops.aten.gather.default(sub_76, 1, unsqueeze_10);  sub_76 = unsqueeze_10 = None
        squeeze: "f32[4096]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[4096]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        where_26: "f32[4096]" = torch.ops.aten.where.self(ne_1, neg, full_default_1);  neg = full_default_1 = None
        sum_26: "i64[]" = torch.ops.aten.sum.default(ne_1);  ne_1 = None
        convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
        sum_27: "f32[]" = torch.ops.aten.sum.default(where_26);  where_26 = None
        div_24: "f32[]" = torch.ops.aten.div.Tensor(sum_27, convert_element_type);  sum_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:492 in forward, code: hidden_states = self.ln_f(hidden_states)
        div_26: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_48, 2048);  rsqrt_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_27: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_47, 2048);  rsqrt_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_283: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_519, [0, 2, 1]);  view_519 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_284: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_515, [0, 2, 1]);  view_515 = None
        permute_285: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_516, [0, 2, 1]);  view_516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_28: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 2048);  rsqrt_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_29: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 2048);  rsqrt_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_316: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_497, [0, 2, 1]);  view_497 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_317: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_493, [0, 2, 1]);  view_493 = None
        permute_318: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_494, [0, 2, 1]);  view_494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_30: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 2048);  rsqrt_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_31: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 2048);  rsqrt_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_349: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_475, [0, 2, 1]);  view_475 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_350: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_471, [0, 2, 1]);  view_471 = None
        permute_351: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_472, [0, 2, 1]);  view_472 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_32: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_42, 2048);  rsqrt_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_33: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_41, 2048);  rsqrt_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_382: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_453, [0, 2, 1]);  view_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_383: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_449, [0, 2, 1]);  view_449 = None
        permute_384: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_450, [0, 2, 1]);  view_450 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_34: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 2048);  rsqrt_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_35: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 2048);  rsqrt_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_415: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_431, [0, 2, 1]);  view_431 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_416: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_427, [0, 2, 1]);  view_427 = None
        permute_417: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_428, [0, 2, 1]);  view_428 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_36: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 2048);  rsqrt_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_37: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 2048);  rsqrt_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_448: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_409, [0, 2, 1]);  view_409 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_449: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_405, [0, 2, 1]);  view_405 = None
        permute_450: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_406, [0, 2, 1]);  view_406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_38: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 2048);  rsqrt_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_39: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 2048);  rsqrt_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_481: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_387, [0, 2, 1]);  view_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_482: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_383, [0, 2, 1]);  view_383 = None
        permute_483: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_384, [0, 2, 1]);  view_384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_40: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 2048);  rsqrt_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_41: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 2048);  rsqrt_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_514: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_365, [0, 2, 1]);  view_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_515: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_361, [0, 2, 1]);  view_361 = None
        permute_516: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_362, [0, 2, 1]);  view_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_42: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 2048);  rsqrt_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_43: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 2048);  rsqrt_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_547: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_343, [0, 2, 1]);  view_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_548: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_339, [0, 2, 1]);  view_339 = None
        permute_549: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_340, [0, 2, 1]);  view_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_44: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 2048);  rsqrt_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_45: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 2048);  rsqrt_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_580: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_321, [0, 2, 1]);  view_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_581: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_317, [0, 2, 1]);  view_317 = None
        permute_582: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_318, [0, 2, 1]);  view_318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_46: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 2048);  rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_47: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 2048);  rsqrt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_613: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_299, [0, 2, 1]);  view_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_614: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_295, [0, 2, 1]);  view_295 = None
        permute_615: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_296, [0, 2, 1]);  view_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_48: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 2048);  rsqrt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_49: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 2048);  rsqrt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_646: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_277, [0, 2, 1]);  view_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_647: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_273, [0, 2, 1]);  view_273 = None
        permute_648: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_274, [0, 2, 1]);  view_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_50: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 2048);  rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_51: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 2048);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_679: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_255, [0, 2, 1]);  view_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_680: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_251, [0, 2, 1]);  view_251 = None
        permute_681: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_252, [0, 2, 1]);  view_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_52: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 2048);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_53: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 2048);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_712: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_233, [0, 2, 1]);  view_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_713: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_229, [0, 2, 1]);  view_229 = None
        permute_714: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_230, [0, 2, 1]);  view_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_54: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 2048);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_55: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 2048);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_745: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_211, [0, 2, 1]);  view_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_746: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_207, [0, 2, 1]);  view_207 = None
        permute_747: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_208, [0, 2, 1]);  view_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_56: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 2048);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_57: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 2048);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_778: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_779: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_185, [0, 2, 1]);  view_185 = None
        permute_780: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_186, [0, 2, 1]);  view_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_58: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 2048);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_59: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 2048);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_811: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_812: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_163, [0, 2, 1]);  view_163 = None
        permute_813: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_164, [0, 2, 1]);  view_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_60: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 2048);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_61: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 2048);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_844: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_845: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_141, [0, 2, 1]);  view_141 = None
        permute_846: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_142, [0, 2, 1]);  view_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_62: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 2048);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_63: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 2048);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_877: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_878: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_119, [0, 2, 1]);  view_119 = None
        permute_879: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_120, [0, 2, 1]);  view_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_64: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 2048);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_65: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 2048);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_910: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_101, [0, 2, 1]);  view_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_911: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
        permute_912: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_66: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 2048);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_67: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 2048);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_943: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_944: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_75, [0, 2, 1]);  view_75 = None
        permute_945: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_76, [0, 2, 1]);  view_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_68: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 2048);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_69: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 2048);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_976: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_57, [0, 2, 1]);  view_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_977: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
        permute_978: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_54, [0, 2, 1]);  view_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_70: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 2048);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_71: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 2048);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_1009: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_35, [0, 2, 1]);  view_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_1010: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
        permute_1011: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_72: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 2048);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_73: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 2048);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_1042: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_1043: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        permute_1044: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
        return (div_24, view_530, primals_1, primals_2, primals_4, primals_6, primals_7, primals_8, primals_9, primals_10, primals_12, primals_14, primals_16, primals_18, primals_20, primals_21, primals_22, primals_23, primals_24, primals_26, primals_28, primals_30, primals_32, primals_34, primals_35, primals_36, primals_37, primals_38, primals_40, primals_42, primals_44, primals_46, primals_48, primals_49, primals_50, primals_51, primals_52, primals_54, primals_56, primals_58, primals_60, primals_62, primals_63, primals_64, primals_65, primals_66, primals_68, primals_70, primals_72, primals_74, primals_76, primals_77, primals_78, primals_79, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_91, primals_92, primals_93, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_105, primals_106, primals_107, primals_108, primals_110, primals_112, primals_114, primals_116, primals_118, primals_119, primals_120, primals_121, primals_122, primals_124, primals_126, primals_128, primals_130, primals_132, primals_133, primals_134, primals_135, primals_136, primals_138, primals_140, primals_142, primals_144, primals_146, primals_147, primals_148, primals_149, primals_150, primals_152, primals_154, primals_156, primals_158, primals_160, primals_161, primals_162, primals_163, primals_164, primals_166, primals_168, primals_170, primals_172, primals_174, primals_175, primals_176, primals_177, primals_178, primals_180, primals_182, primals_184, primals_186, primals_188, primals_189, primals_190, primals_191, primals_192, primals_194, primals_196, primals_198, primals_200, primals_202, primals_203, primals_204, primals_205, primals_206, primals_208, primals_210, primals_212, primals_214, primals_216, primals_217, primals_218, primals_219, primals_220, primals_222, primals_224, primals_226, primals_228, primals_230, primals_231, primals_232, primals_233, primals_234, primals_236, primals_238, primals_240, primals_242, primals_244, primals_245, primals_246, primals_247, primals_248, primals_250, primals_252, primals_254, primals_256, primals_258, primals_259, primals_260, primals_261, primals_262, primals_264, primals_266, primals_268, primals_270, primals_272, primals_273, primals_274, primals_275, primals_276, primals_278, primals_280, primals_282, primals_284, primals_286, primals_287, primals_288, primals_289, primals_290, primals_292, primals_294, primals_296, primals_298, primals_300, primals_301, primals_302, primals_303, primals_304, primals_306, primals_308, primals_310, primals_312, primals_314, primals_315, primals_316, primals_317, primals_318, primals_320, primals_322, primals_324, primals_326, primals_328, primals_329, primals_330, primals_331, primals_332, primals_334, primals_336, primals_338, primals_340, embedding, unsqueeze, embedding_1, getitem_1, rsqrt, view, div, view_16, mul_2, view_18, addmm_1, view_20, mul_8, view_22, div_1, view_38, mul_10, view_40, addmm_4, view_42, mul_16, view_44, div_2, view_60, mul_18, view_62, addmm_7, view_64, mul_24, view_66, div_3, view_82, mul_26, view_84, addmm_10, view_86, mul_32, view_88, div_4, view_104, mul_34, view_106, addmm_13, view_108, mul_40, view_110, div_5, view_126, mul_42, view_128, addmm_16, view_130, mul_48, view_132, div_6, view_148, mul_50, view_150, addmm_19, view_152, mul_56, view_154, div_7, view_170, mul_58, view_172, addmm_22, view_174, mul_64, view_176, div_8, view_192, mul_66, view_194, addmm_25, view_196, mul_72, view_198, div_9, view_214, mul_74, view_216, addmm_28, view_218, mul_80, view_220, div_10, view_236, mul_82, view_238, addmm_31, view_240, mul_88, view_242, div_11, view_258, mul_90, view_260, addmm_34, view_262, mul_96, view_264, div_12, view_280, mul_98, view_282, addmm_37, view_284, mul_104, view_286, div_13, view_302, mul_106, view_304, addmm_40, view_306, mul_112, view_308, div_14, view_324, mul_114, view_326, addmm_43, view_328, mul_120, view_330, div_15, view_346, mul_122, view_348, addmm_46, view_350, mul_128, view_352, div_16, view_368, mul_130, view_370, addmm_49, view_372, mul_136, view_374, div_17, view_390, mul_138, view_392, addmm_52, view_394, mul_144, view_396, div_18, view_412, mul_146, view_414, addmm_55, view_416, mul_152, view_418, div_19, view_434, mul_154, view_436, addmm_58, view_438, mul_160, view_440, div_20, view_456, mul_162, view_458, addmm_61, view_460, mul_168, view_462, div_21, view_478, mul_170, view_480, addmm_64, view_482, mul_176, view_484, div_22, view_500, mul_178, view_502, addmm_67, view_504, mul_184, view_506, div_23, view_522, mul_186, view_524, addmm_70, view_526, mul_192, view_529, view_530, constant_pad_nd, amax_24, log, convert_element_type, div_26, div_27, permute_283, permute_284, permute_285, div_28, div_29, permute_316, permute_317, permute_318, div_30, div_31, permute_349, permute_350, permute_351, div_32, div_33, permute_382, permute_383, permute_384, div_34, div_35, permute_415, permute_416, permute_417, div_36, div_37, permute_448, permute_449, permute_450, div_38, div_39, permute_481, permute_482, permute_483, div_40, div_41, permute_514, permute_515, permute_516, div_42, div_43, permute_547, permute_548, permute_549, div_44, div_45, permute_580, permute_581, permute_582, div_46, div_47, permute_613, permute_614, permute_615, div_48, div_49, permute_646, permute_647, permute_648, div_50, div_51, permute_679, permute_680, permute_681, div_52, div_53, permute_712, permute_713, permute_714, div_54, div_55, permute_745, permute_746, permute_747, div_56, div_57, permute_778, permute_779, permute_780, div_58, div_59, permute_811, permute_812, permute_813, div_60, div_61, permute_844, permute_845, permute_846, div_62, div_63, permute_877, permute_878, permute_879, div_64, div_65, permute_910, permute_911, permute_912, div_66, div_67, permute_943, permute_944, permute_945, div_68, div_69, permute_976, permute_977, permute_978, div_70, div_71, permute_1009, permute_1010, permute_1011, div_72, div_73, permute_1042, permute_1043, permute_1044)
