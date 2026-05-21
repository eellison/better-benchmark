class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[8, 1024]", primals_2: "f32[32128, 768]", primals_3: "f32[768]", primals_4: "f32[768, 768]", primals_5: "f32[768, 768]", primals_6: "f32[768, 768]", primals_7: "f32[32, 12]", primals_8: "f32[768, 768]", primals_9: "f32[768]", primals_10: "f32[3072, 768]", primals_11: "f32[768, 3072]", primals_12: "f32[768]", primals_13: "f32[768, 768]", primals_14: "f32[768, 768]", primals_15: "f32[768, 768]", primals_16: "f32[768, 768]", primals_17: "f32[768]", primals_18: "f32[3072, 768]", primals_19: "f32[768, 3072]", primals_20: "f32[768]", primals_21: "f32[768, 768]", primals_22: "f32[768, 768]", primals_23: "f32[768, 768]", primals_24: "f32[768, 768]", primals_25: "f32[768]", primals_26: "f32[3072, 768]", primals_27: "f32[768, 3072]", primals_28: "f32[768]", primals_29: "f32[768, 768]", primals_30: "f32[768, 768]", primals_31: "f32[768, 768]", primals_32: "f32[768, 768]", primals_33: "f32[768]", primals_34: "f32[3072, 768]", primals_35: "f32[768, 3072]", primals_36: "f32[768]", primals_37: "f32[768, 768]", primals_38: "f32[768, 768]", primals_39: "f32[768, 768]", primals_40: "f32[768, 768]", primals_41: "f32[768]", primals_42: "f32[3072, 768]", primals_43: "f32[768, 3072]", primals_44: "f32[768]", primals_45: "f32[768, 768]", primals_46: "f32[768, 768]", primals_47: "f32[768, 768]", primals_48: "f32[768, 768]", primals_49: "f32[768]", primals_50: "f32[3072, 768]", primals_51: "f32[768, 3072]", primals_52: "f32[768]", primals_53: "f32[768, 768]", primals_54: "f32[768, 768]", primals_55: "f32[768, 768]", primals_56: "f32[768, 768]", primals_57: "f32[768]", primals_58: "f32[3072, 768]", primals_59: "f32[768, 3072]", primals_60: "f32[768]", primals_61: "f32[768, 768]", primals_62: "f32[768, 768]", primals_63: "f32[768, 768]", primals_64: "f32[768, 768]", primals_65: "f32[768]", primals_66: "f32[3072, 768]", primals_67: "f32[768, 3072]", primals_68: "f32[768]", primals_69: "f32[768, 768]", primals_70: "f32[768, 768]", primals_71: "f32[768, 768]", primals_72: "f32[768, 768]", primals_73: "f32[768]", primals_74: "f32[3072, 768]", primals_75: "f32[768, 3072]", primals_76: "f32[768]", primals_77: "f32[768, 768]", primals_78: "f32[768, 768]", primals_79: "f32[768, 768]", primals_80: "f32[768, 768]", primals_81: "f32[768]", primals_82: "f32[3072, 768]", primals_83: "f32[768, 3072]", primals_84: "f32[768]", primals_85: "f32[768, 768]", primals_86: "f32[768, 768]", primals_87: "f32[768, 768]", primals_88: "f32[768, 768]", primals_89: "f32[768]", primals_90: "f32[3072, 768]", primals_91: "f32[768, 3072]", primals_92: "f32[768]", primals_93: "f32[768, 768]", primals_94: "f32[768, 768]", primals_95: "f32[768, 768]", primals_96: "f32[768, 768]", primals_97: "f32[768]", primals_98: "f32[3072, 768]", primals_99: "f32[768, 3072]", primals_100: "f32[768]", primals_101: "i64[8, 1024]", primals_102: "f32[768]", primals_103: "f32[768, 768]", primals_104: "f32[768, 768]", primals_105: "f32[768, 768]", primals_106: "f32[32, 12]", primals_107: "f32[768, 768]", primals_108: "f32[768]", primals_109: "f32[768, 768]", primals_110: "f32[768, 768]", primals_111: "f32[768, 768]", primals_112: "f32[768, 768]", primals_113: "f32[768]", primals_114: "f32[3072, 768]", primals_115: "f32[768, 3072]", primals_116: "f32[768]", primals_117: "f32[768, 768]", primals_118: "f32[768, 768]", primals_119: "f32[768, 768]", primals_120: "f32[768, 768]", primals_121: "f32[768]", primals_122: "f32[768, 768]", primals_123: "f32[768, 768]", primals_124: "f32[768, 768]", primals_125: "f32[768, 768]", primals_126: "f32[768]", primals_127: "f32[3072, 768]", primals_128: "f32[768, 3072]", primals_129: "f32[768]", primals_130: "f32[768, 768]", primals_131: "f32[768, 768]", primals_132: "f32[768, 768]", primals_133: "f32[768, 768]", primals_134: "f32[768]", primals_135: "f32[768, 768]", primals_136: "f32[768, 768]", primals_137: "f32[768, 768]", primals_138: "f32[768, 768]", primals_139: "f32[768]", primals_140: "f32[3072, 768]", primals_141: "f32[768, 3072]", primals_142: "f32[768]", primals_143: "f32[768, 768]", primals_144: "f32[768, 768]", primals_145: "f32[768, 768]", primals_146: "f32[768, 768]", primals_147: "f32[768]", primals_148: "f32[768, 768]", primals_149: "f32[768, 768]", primals_150: "f32[768, 768]", primals_151: "f32[768, 768]", primals_152: "f32[768]", primals_153: "f32[3072, 768]", primals_154: "f32[768, 3072]", primals_155: "f32[768]", primals_156: "f32[768, 768]", primals_157: "f32[768, 768]", primals_158: "f32[768, 768]", primals_159: "f32[768, 768]", primals_160: "f32[768]", primals_161: "f32[768, 768]", primals_162: "f32[768, 768]", primals_163: "f32[768, 768]", primals_164: "f32[768, 768]", primals_165: "f32[768]", primals_166: "f32[3072, 768]", primals_167: "f32[768, 3072]", primals_168: "f32[768]", primals_169: "f32[768, 768]", primals_170: "f32[768, 768]", primals_171: "f32[768, 768]", primals_172: "f32[768, 768]", primals_173: "f32[768]", primals_174: "f32[768, 768]", primals_175: "f32[768, 768]", primals_176: "f32[768, 768]", primals_177: "f32[768, 768]", primals_178: "f32[768]", primals_179: "f32[3072, 768]", primals_180: "f32[768, 3072]", primals_181: "f32[768]", primals_182: "f32[768, 768]", primals_183: "f32[768, 768]", primals_184: "f32[768, 768]", primals_185: "f32[768, 768]", primals_186: "f32[768]", primals_187: "f32[768, 768]", primals_188: "f32[768, 768]", primals_189: "f32[768, 768]", primals_190: "f32[768, 768]", primals_191: "f32[768]", primals_192: "f32[3072, 768]", primals_193: "f32[768, 3072]", primals_194: "f32[768]", primals_195: "f32[768, 768]", primals_196: "f32[768, 768]", primals_197: "f32[768, 768]", primals_198: "f32[768, 768]", primals_199: "f32[768]", primals_200: "f32[768, 768]", primals_201: "f32[768, 768]", primals_202: "f32[768, 768]", primals_203: "f32[768, 768]", primals_204: "f32[768]", primals_205: "f32[3072, 768]", primals_206: "f32[768, 3072]", primals_207: "f32[768]", primals_208: "f32[768, 768]", primals_209: "f32[768, 768]", primals_210: "f32[768, 768]", primals_211: "f32[768, 768]", primals_212: "f32[768]", primals_213: "f32[768, 768]", primals_214: "f32[768, 768]", primals_215: "f32[768, 768]", primals_216: "f32[768, 768]", primals_217: "f32[768]", primals_218: "f32[3072, 768]", primals_219: "f32[768, 3072]", primals_220: "f32[768]", primals_221: "f32[768, 768]", primals_222: "f32[768, 768]", primals_223: "f32[768, 768]", primals_224: "f32[768, 768]", primals_225: "f32[768]", primals_226: "f32[768, 768]", primals_227: "f32[768, 768]", primals_228: "f32[768, 768]", primals_229: "f32[768, 768]", primals_230: "f32[768]", primals_231: "f32[3072, 768]", primals_232: "f32[768, 3072]", primals_233: "f32[768]", primals_234: "f32[768, 768]", primals_235: "f32[768, 768]", primals_236: "f32[768, 768]", primals_237: "f32[768, 768]", primals_238: "f32[768]", primals_239: "f32[768, 768]", primals_240: "f32[768, 768]", primals_241: "f32[768, 768]", primals_242: "f32[768, 768]", primals_243: "f32[768]", primals_244: "f32[3072, 768]", primals_245: "f32[768, 3072]", primals_246: "f32[768]", primals_247: "f32[768, 768]", primals_248: "f32[768, 768]", primals_249: "f32[768, 768]", primals_250: "f32[768, 768]", primals_251: "f32[768]", primals_252: "f32[768, 768]", primals_253: "f32[768, 768]", primals_254: "f32[768, 768]", primals_255: "f32[768, 768]", primals_256: "f32[768]", primals_257: "f32[3072, 768]", primals_258: "f32[768, 3072]", primals_259: "f32[768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:669 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding: "f32[8, 1024, 768]" = torch.ops.aten.embedding.default(primals_2, primals_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_2: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[1024]" = torch.ops.aten.add.Tensor(iota_2, 0)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None
        unsqueeze_1: "i64[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "i64[1, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:87 in bidirectional_mask_function, code: return q_idx >= 0
        ge: "b8[1, 1, 1024, 1]" = torch.ops.aten.ge.Scalar(unsqueeze_2, 0)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand: "b8[8, 1, 1024, 1024]" = torch.ops.aten.expand.default(ge, [8, -1, 1024, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default, full_default_1);  expand = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[124]" = torch.ops.prims.inductor_seeds.default(124, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:721 in forward, code: hidden_states = self.dropout(inputs_embeds)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_123: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_123, 0.1);  inductor_random_default_123 = None
        mul: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt, embedding)
        mul_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_1: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(mul_1, 2)
        mean: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_2: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1, rsqrt)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_3, mul_2);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_4, [1, 0])
        view_1: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_3, [8192, 768]);  mul_3 = None
        mm: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1, permute);  permute = None
        view_2: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm, [8, 1024, 768]);  mm = None
        view_3: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_2, [8, 1024, -1, 64]);  view_2 = None
        permute_1: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(primals_5, [1, 0])
        mm_1: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1, permute_2);  permute_2 = None
        view_5: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_1, [8, 1024, 768]);  mm_1 = None
        view_6: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_5, [8, 1024, -1, 64]);  view_5 = None
        permute_3: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_4: "f32[768, 768]" = torch.ops.aten.permute.default(primals_6, [1, 0])
        mm_2: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1, permute_4);  permute_4 = None
        view_8: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_2, [8, 1024, 768]);  mm_2 = None
        view_9: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_8, [8, 1024, -1, 64]);  view_8 = None
        permute_5: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_6: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        expand_1: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_1, [8, 12, 1024, 64]);  permute_1 = None
        clone: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_10: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone, [96, 1024, 64]);  clone = None
        expand_2: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_6, [8, 12, 64, 1024]);  permute_6 = None
        clone_1: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_11: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_1, [96, 64, 1024]);  clone_1 = None
        bmm: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_10, view_11)
        view_12: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm, [8, 12, 1024, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:227 in compute_bias, code: context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None] + past_seen_tokens
        unsqueeze_3: "i64[1024, 1]" = torch.ops.aten.unsqueeze.default(iota_2, 1)
        add_3: "i64[1024, 1]" = torch.ops.aten.add.Tensor(unsqueeze_3, 0);  unsqueeze_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:228 in compute_bias, code: memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        unsqueeze_4: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(iota_2, 0);  iota_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:229 in compute_bias, code: relative_position = memory_position - context_position  # shape (query_length, key_length)
        sub: "i64[1024, 1024]" = torch.ops.aten.sub.Tensor(unsqueeze_4, add_3);  unsqueeze_4 = add_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:200 in _relative_position_bucket, code: relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        gt_1: "b8[1024, 1024]" = torch.ops.aten.gt.Scalar(sub, 0)
        convert_element_type: "i64[1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_1, torch.int64);  gt_1 = None
        mul_4: "i64[1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type, 16);  convert_element_type = None
        add_4: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(mul_4, 0);  mul_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:201 in _relative_position_bucket, code: relative_position = torch.abs(relative_position)
        abs_1: "i64[1024, 1024]" = torch.ops.aten.abs.default(sub)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:208 in _relative_position_bucket, code: is_small = relative_position < max_exact
        lt: "b8[1024, 1024]" = torch.ops.aten.lt.Scalar(abs_1, 8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:212 in _relative_position_bucket, code: torch.log(relative_position.float() / max_exact)
        convert_element_type_1: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(abs_1, torch.float32)
        div: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(convert_element_type_1, 8);  convert_element_type_1 = None
        log: "f32[1024, 1024]" = torch.ops.aten.log.default(div);  div = None
        div_1: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(log, 2.772588722239781);  log = None
        mul_5: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(div_1, 8);  div_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:215 in _relative_position_bucket, code: ).to(torch.long)
        convert_element_type_2: "i64[1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_5, torch.int64);  mul_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:211 in _relative_position_bucket, code: relative_position_if_large = max_exact + (
        add_5: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_2, 8);  convert_element_type_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:217 in _relative_position_bucket, code: relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        full_default_2: "i64[1024, 1024]" = torch.ops.aten.full.default([1024, 1024], 15, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:216 in _relative_position_bucket, code: relative_position_if_large = torch.min(
        minimum: "i64[1024, 1024]" = torch.ops.aten.minimum.default(add_5, full_default_2);  add_5 = full_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:220 in _relative_position_bucket, code: relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        where_1: "i64[1024, 1024]" = torch.ops.aten.where.self(lt, abs_1, minimum);  lt = abs_1 = minimum = None
        add_6: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(add_4, where_1);  add_4 = where_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:236 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        embedding_1: "f32[1024, 1024, 12]" = torch.ops.aten.embedding.default(primals_7, add_6);  primals_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:237 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        permute_7: "f32[12, 1024, 1024]" = torch.ops.aten.permute.default(embedding_1, [2, 0, 1])
        unsqueeze_5: "f32[1, 12, 1024, 1024]" = torch.ops.aten.unsqueeze.default(permute_7, 0);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        add_7: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_5, where);  unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_8: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_12, add_7);  view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_8, [-1], True)
        sub_1: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_8, amax);  add_8 = None
        exp: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_2: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_122: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_2: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_122, 0.1);  inductor_random_default_122 = None
        mul_6: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_2, div_2);  div_2 = None
        mul_7: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_6, 1.1111111111111112);  mul_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_3: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_7, [8, 12, 1024, 1024]);  mul_7 = None
        view_16: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_3, [96, 1024, 1024]);  expand_3 = None
        expand_4: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_5, [8, 12, 1024, 64]);  permute_5 = None
        clone_2: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_17: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_2, [96, 1024, 64]);  clone_2 = None
        bmm_1: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_16, view_17)
        view_18: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_1, [8, 12, 1024, 64]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_8: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        clone_3: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_19: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_3, [8, 1024, -1]);  clone_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_9: "f32[768, 768]" = torch.ops.aten.permute.default(primals_8, [1, 0])
        view_20: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_19, [8192, 768]);  view_19 = None
        mm_3: "f32[8192, 768]" = torch.ops.aten.mm.default(view_20, permute_9);  permute_9 = None
        view_21: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_3, [8, 1024, 768]);  mm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_121: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        gt_3: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_121, 0.1);  inductor_random_default_121 = None
        mul_8: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_3, view_21);  view_21 = None
        mul_9: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_8, 1.1111111111111112);  mul_8 = None
        add_9: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_1, mul_9);  mul_1 = mul_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_2: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_9, 2)
        mean_1: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_10: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        rsqrt_1: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_10: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_9, rsqrt_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_11: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_9, mul_10);  mul_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_10: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_10, [1, 0])
        view_22: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_11, [8192, 768]);  mul_11 = None
        mm_4: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_22, permute_10);  permute_10 = None
        view_23: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_4, [8, 1024, 3072]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu: "f32[8, 1024, 3072]" = torch.ops.aten.relu.default(view_23);  view_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_3: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_120: "f32[8, 1024, 3072]" = torch.ops.prims.inductor_random.default([8, 1024, 3072], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        gt_4: "b8[8, 1024, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_120, 0.1);  inductor_random_default_120 = None
        mul_12: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(gt_4, relu)
        mul_13: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_12, 1.1111111111111112);  mul_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_11: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_11, [1, 0])
        view_24: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_13, [8192, 3072]);  mul_13 = None
        mm_5: "f32[8192, 768]" = torch.ops.aten.mm.default(view_24, permute_11);  permute_11 = None
        view_25: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_5, [8, 1024, 768]);  mm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_4: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_random_default_119: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        gt_5: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_119, 0.1);  inductor_random_default_119 = None
        mul_14: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_5, view_25);  view_25 = None
        mul_15: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_14, 1.1111111111111112);  mul_14 = None
        add_11: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_9, mul_15);  mul_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_3: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_11, 2)
        mean_2: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_12: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
        rsqrt_2: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_16: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_11, rsqrt_2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_17: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_12, mul_16);  mul_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_12: "f32[768, 768]" = torch.ops.aten.permute.default(primals_13, [1, 0])
        view_26: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_17, [8192, 768]);  mul_17 = None
        mm_6: "f32[8192, 768]" = torch.ops.aten.mm.default(view_26, permute_12);  permute_12 = None
        view_27: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_6, [8, 1024, 768]);  mm_6 = None
        view_28: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_27, [8, 1024, -1, 64]);  view_27 = None
        permute_13: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_14: "f32[768, 768]" = torch.ops.aten.permute.default(primals_14, [1, 0])
        mm_7: "f32[8192, 768]" = torch.ops.aten.mm.default(view_26, permute_14);  permute_14 = None
        view_30: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_7, [8, 1024, 768]);  mm_7 = None
        view_31: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_30, [8, 1024, -1, 64]);  view_30 = None
        permute_15: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_16: "f32[768, 768]" = torch.ops.aten.permute.default(primals_15, [1, 0])
        mm_8: "f32[8192, 768]" = torch.ops.aten.mm.default(view_26, permute_16);  permute_16 = None
        view_33: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_8, [8, 1024, 768]);  mm_8 = None
        view_34: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_33, [8, 1024, -1, 64]);  view_33 = None
        permute_17: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_18: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_15, [0, 1, 3, 2]);  permute_15 = None
        expand_5: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_13, [8, 12, 1024, 64]);  permute_13 = None
        clone_4: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_35: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_4, [96, 1024, 64]);  clone_4 = None
        expand_6: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_18, [8, 12, 64, 1024]);  permute_18 = None
        clone_5: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_36: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_5, [96, 64, 1024]);  clone_5 = None
        bmm_2: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_35, view_36)
        view_37: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_2, [8, 12, 1024, 1024]);  bmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_13: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_37, add_7);  view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_1: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_13, [-1], True)
        sub_2: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_13, amax_1);  add_13 = amax_1 = None
        exp_1: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_2: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_3: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_5: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_random_default_118: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_5, 'rand');  inductor_lookup_seed_default_5 = None
        gt_6: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_118, 0.1);  inductor_random_default_118 = None
        mul_18: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_6, div_3)
        mul_19: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_18, 1.1111111111111112);  mul_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_7: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_19, [8, 12, 1024, 1024]);  mul_19 = None
        view_41: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_7, [96, 1024, 1024]);  expand_7 = None
        expand_8: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_17, [8, 12, 1024, 64]);  permute_17 = None
        clone_6: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_42: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_6, [96, 1024, 64]);  clone_6 = None
        bmm_3: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_41, view_42)
        view_43: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_3, [8, 12, 1024, 64]);  bmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_19: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_43, [0, 2, 1, 3]);  view_43 = None
        clone_7: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_44: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_7, [8, 1024, -1]);  clone_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_20: "f32[768, 768]" = torch.ops.aten.permute.default(primals_16, [1, 0])
        view_45: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_44, [8192, 768]);  view_44 = None
        mm_9: "f32[8192, 768]" = torch.ops.aten.mm.default(view_45, permute_20);  permute_20 = None
        view_46: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_9, [8, 1024, 768]);  mm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_6: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6)
        inductor_random_default_117: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_6, 'rand');  inductor_lookup_seed_default_6 = None
        gt_7: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_117, 0.1);  inductor_random_default_117 = None
        mul_20: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_7, view_46);  view_46 = None
        mul_21: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_20, 1.1111111111111112);  mul_20 = None
        add_14: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_11, mul_21);  mul_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_4: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_14, 2)
        mean_3: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_15: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None
        rsqrt_3: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        mul_22: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_14, rsqrt_3)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_23: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_17, mul_22);  mul_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_21: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_18, [1, 0])
        view_47: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_23, [8192, 768]);  mul_23 = None
        mm_10: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_47, permute_21);  permute_21 = None
        view_48: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_10, [8, 1024, 3072]);  mm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_1: "f32[8, 1024, 3072]" = torch.ops.aten.relu.default(view_48);  view_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_7: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 7)
        inductor_random_default_116: "f32[8, 1024, 3072]" = torch.ops.prims.inductor_random.default([8, 1024, 3072], inductor_lookup_seed_default_7, 'rand');  inductor_lookup_seed_default_7 = None
        gt_8: "b8[8, 1024, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_116, 0.1);  inductor_random_default_116 = None
        mul_24: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(gt_8, relu_1)
        mul_25: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_24, 1.1111111111111112);  mul_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_22: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_19, [1, 0])
        view_49: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_25, [8192, 3072]);  mul_25 = None
        mm_11: "f32[8192, 768]" = torch.ops.aten.mm.default(view_49, permute_22);  permute_22 = None
        view_50: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_11, [8, 1024, 768]);  mm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_8: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 8)
        inductor_random_default_115: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_8, 'rand');  inductor_lookup_seed_default_8 = None
        gt_9: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_115, 0.1);  inductor_random_default_115 = None
        mul_26: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_9, view_50);  view_50 = None
        mul_27: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_26, 1.1111111111111112);  mul_26 = None
        add_16: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_14, mul_27);  mul_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_5: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_16, 2)
        mean_4: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_17: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_4, 1e-06);  mean_4 = None
        rsqrt_4: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        mul_28: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_16, rsqrt_4)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_29: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_20, mul_28);  mul_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_23: "f32[768, 768]" = torch.ops.aten.permute.default(primals_21, [1, 0])
        view_51: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_29, [8192, 768]);  mul_29 = None
        mm_12: "f32[8192, 768]" = torch.ops.aten.mm.default(view_51, permute_23);  permute_23 = None
        view_52: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_12, [8, 1024, 768]);  mm_12 = None
        view_53: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_52, [8, 1024, -1, 64]);  view_52 = None
        permute_24: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_25: "f32[768, 768]" = torch.ops.aten.permute.default(primals_22, [1, 0])
        mm_13: "f32[8192, 768]" = torch.ops.aten.mm.default(view_51, permute_25);  permute_25 = None
        view_55: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_13, [8, 1024, 768]);  mm_13 = None
        view_56: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_55, [8, 1024, -1, 64]);  view_55 = None
        permute_26: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_27: "f32[768, 768]" = torch.ops.aten.permute.default(primals_23, [1, 0])
        mm_14: "f32[8192, 768]" = torch.ops.aten.mm.default(view_51, permute_27);  permute_27 = None
        view_58: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_14, [8, 1024, 768]);  mm_14 = None
        view_59: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_58, [8, 1024, -1, 64]);  view_58 = None
        permute_28: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_59, [0, 2, 1, 3]);  view_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_29: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_26, [0, 1, 3, 2]);  permute_26 = None
        expand_9: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_24, [8, 12, 1024, 64]);  permute_24 = None
        clone_8: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_60: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_8, [96, 1024, 64]);  clone_8 = None
        expand_10: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_29, [8, 12, 64, 1024]);  permute_29 = None
        clone_9: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_61: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_9, [96, 64, 1024]);  clone_9 = None
        bmm_4: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_60, view_61)
        view_62: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_4, [8, 12, 1024, 1024]);  bmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_18: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_62, add_7);  view_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_2: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_18, [-1], True)
        sub_3: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_18, amax_2);  add_18 = amax_2 = None
        exp_2: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        sum_3: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_4: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_9: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 9)
        inductor_random_default_114: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_9, 'rand');  inductor_lookup_seed_default_9 = None
        gt_10: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_114, 0.1);  inductor_random_default_114 = None
        mul_30: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_10, div_4)
        mul_31: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_30, 1.1111111111111112);  mul_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_11: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_31, [8, 12, 1024, 1024]);  mul_31 = None
        view_66: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_11, [96, 1024, 1024]);  expand_11 = None
        expand_12: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_28, [8, 12, 1024, 64]);  permute_28 = None
        clone_10: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_67: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_10, [96, 1024, 64]);  clone_10 = None
        bmm_5: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_66, view_67)
        view_68: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_5, [8, 12, 1024, 64]);  bmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_30: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
        clone_11: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_69: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_11, [8, 1024, -1]);  clone_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_31: "f32[768, 768]" = torch.ops.aten.permute.default(primals_24, [1, 0])
        view_70: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_69, [8192, 768]);  view_69 = None
        mm_15: "f32[8192, 768]" = torch.ops.aten.mm.default(view_70, permute_31);  permute_31 = None
        view_71: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_15, [8, 1024, 768]);  mm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_10: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 10)
        inductor_random_default_113: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_10, 'rand');  inductor_lookup_seed_default_10 = None
        gt_11: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_113, 0.1);  inductor_random_default_113 = None
        mul_32: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_11, view_71);  view_71 = None
        mul_33: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_32, 1.1111111111111112);  mul_32 = None
        add_19: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_16, mul_33);  mul_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_6: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_19, 2)
        mean_5: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_6, [-1], True);  pow_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_20: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_5, 1e-06);  mean_5 = None
        rsqrt_5: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_34: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_19, rsqrt_5)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_35: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_25, mul_34);  mul_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_32: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_26, [1, 0])
        view_72: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_35, [8192, 768]);  mul_35 = None
        mm_16: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_72, permute_32);  permute_32 = None
        view_73: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_16, [8, 1024, 3072]);  mm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_2: "f32[8, 1024, 3072]" = torch.ops.aten.relu.default(view_73);  view_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_11: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 11)
        inductor_random_default_112: "f32[8, 1024, 3072]" = torch.ops.prims.inductor_random.default([8, 1024, 3072], inductor_lookup_seed_default_11, 'rand');  inductor_lookup_seed_default_11 = None
        gt_12: "b8[8, 1024, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_112, 0.1);  inductor_random_default_112 = None
        mul_36: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(gt_12, relu_2)
        mul_37: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_36, 1.1111111111111112);  mul_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_33: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_27, [1, 0])
        view_74: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_37, [8192, 3072]);  mul_37 = None
        mm_17: "f32[8192, 768]" = torch.ops.aten.mm.default(view_74, permute_33);  permute_33 = None
        view_75: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_17, [8, 1024, 768]);  mm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_12: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 12)
        inductor_random_default_111: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_12, 'rand');  inductor_lookup_seed_default_12 = None
        gt_13: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_111, 0.1);  inductor_random_default_111 = None
        mul_38: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_13, view_75);  view_75 = None
        mul_39: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_38, 1.1111111111111112);  mul_38 = None
        add_21: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_19, mul_39);  mul_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_7: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_21, 2)
        mean_6: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_22: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_6, 1e-06);  mean_6 = None
        rsqrt_6: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        mul_40: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_21, rsqrt_6)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_41: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_28, mul_40);  mul_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_34: "f32[768, 768]" = torch.ops.aten.permute.default(primals_29, [1, 0])
        view_76: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_41, [8192, 768]);  mul_41 = None
        mm_18: "f32[8192, 768]" = torch.ops.aten.mm.default(view_76, permute_34);  permute_34 = None
        view_77: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_18, [8, 1024, 768]);  mm_18 = None
        view_78: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_77, [8, 1024, -1, 64]);  view_77 = None
        permute_35: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_36: "f32[768, 768]" = torch.ops.aten.permute.default(primals_30, [1, 0])
        mm_19: "f32[8192, 768]" = torch.ops.aten.mm.default(view_76, permute_36);  permute_36 = None
        view_80: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_19, [8, 1024, 768]);  mm_19 = None
        view_81: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_80, [8, 1024, -1, 64]);  view_80 = None
        permute_37: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_81, [0, 2, 1, 3]);  view_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_38: "f32[768, 768]" = torch.ops.aten.permute.default(primals_31, [1, 0])
        mm_20: "f32[8192, 768]" = torch.ops.aten.mm.default(view_76, permute_38);  permute_38 = None
        view_83: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_20, [8, 1024, 768]);  mm_20 = None
        view_84: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_83, [8, 1024, -1, 64]);  view_83 = None
        permute_39: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_40: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_37, [0, 1, 3, 2]);  permute_37 = None
        expand_13: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_35, [8, 12, 1024, 64]);  permute_35 = None
        clone_12: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_85: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_12, [96, 1024, 64]);  clone_12 = None
        expand_14: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_40, [8, 12, 64, 1024]);  permute_40 = None
        clone_13: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_86: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_13, [96, 64, 1024]);  clone_13 = None
        bmm_6: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_85, view_86)
        view_87: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_6, [8, 12, 1024, 1024]);  bmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_23: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_87, add_7);  view_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_3: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_23, [-1], True)
        sub_4: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_23, amax_3);  add_23 = amax_3 = None
        exp_3: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_4: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_5: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_13: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 13)
        inductor_random_default_110: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_13, 'rand');  inductor_lookup_seed_default_13 = None
        gt_14: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_110, 0.1);  inductor_random_default_110 = None
        mul_42: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_14, div_5)
        mul_43: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_42, 1.1111111111111112);  mul_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_15: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_43, [8, 12, 1024, 1024]);  mul_43 = None
        view_91: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_15, [96, 1024, 1024]);  expand_15 = None
        expand_16: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_39, [8, 12, 1024, 64]);  permute_39 = None
        clone_14: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_92: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_14, [96, 1024, 64]);  clone_14 = None
        bmm_7: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_91, view_92)
        view_93: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_7, [8, 12, 1024, 64]);  bmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_41: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None
        clone_15: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_94: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_15, [8, 1024, -1]);  clone_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_42: "f32[768, 768]" = torch.ops.aten.permute.default(primals_32, [1, 0])
        view_95: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_94, [8192, 768]);  view_94 = None
        mm_21: "f32[8192, 768]" = torch.ops.aten.mm.default(view_95, permute_42);  permute_42 = None
        view_96: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_21, [8, 1024, 768]);  mm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_14: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 14)
        inductor_random_default_109: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_14, 'rand');  inductor_lookup_seed_default_14 = None
        gt_15: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_109, 0.1);  inductor_random_default_109 = None
        mul_44: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_15, view_96);  view_96 = None
        mul_45: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_44, 1.1111111111111112);  mul_44 = None
        add_24: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_21, mul_45);  mul_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_8: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_24, 2)
        mean_7: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_8, [-1], True);  pow_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_25: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_7, 1e-06);  mean_7 = None
        rsqrt_7: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        mul_46: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_24, rsqrt_7)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_47: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_33, mul_46);  mul_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_43: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_34, [1, 0])
        view_97: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_47, [8192, 768]);  mul_47 = None
        mm_22: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_97, permute_43);  permute_43 = None
        view_98: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_22, [8, 1024, 3072]);  mm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_3: "f32[8, 1024, 3072]" = torch.ops.aten.relu.default(view_98);  view_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_15: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 15)
        inductor_random_default_108: "f32[8, 1024, 3072]" = torch.ops.prims.inductor_random.default([8, 1024, 3072], inductor_lookup_seed_default_15, 'rand');  inductor_lookup_seed_default_15 = None
        gt_16: "b8[8, 1024, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_108, 0.1);  inductor_random_default_108 = None
        mul_48: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(gt_16, relu_3)
        mul_49: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_48, 1.1111111111111112);  mul_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_44: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_35, [1, 0])
        view_99: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_49, [8192, 3072]);  mul_49 = None
        mm_23: "f32[8192, 768]" = torch.ops.aten.mm.default(view_99, permute_44);  permute_44 = None
        view_100: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_23, [8, 1024, 768]);  mm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_16: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 16)
        inductor_random_default_107: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_16, 'rand');  inductor_lookup_seed_default_16 = None
        gt_17: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_107, 0.1);  inductor_random_default_107 = None
        mul_50: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_17, view_100);  view_100 = None
        mul_51: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_50, 1.1111111111111112);  mul_50 = None
        add_26: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_24, mul_51);  mul_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_9: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_26, 2)
        mean_8: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_9, [-1], True);  pow_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_27: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_8, 1e-06);  mean_8 = None
        rsqrt_8: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        mul_52: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_26, rsqrt_8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_53: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_36, mul_52);  mul_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_45: "f32[768, 768]" = torch.ops.aten.permute.default(primals_37, [1, 0])
        view_101: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_53, [8192, 768]);  mul_53 = None
        mm_24: "f32[8192, 768]" = torch.ops.aten.mm.default(view_101, permute_45);  permute_45 = None
        view_102: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_24, [8, 1024, 768]);  mm_24 = None
        view_103: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_102, [8, 1024, -1, 64]);  view_102 = None
        permute_46: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_103, [0, 2, 1, 3]);  view_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_47: "f32[768, 768]" = torch.ops.aten.permute.default(primals_38, [1, 0])
        mm_25: "f32[8192, 768]" = torch.ops.aten.mm.default(view_101, permute_47);  permute_47 = None
        view_105: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_25, [8, 1024, 768]);  mm_25 = None
        view_106: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_105, [8, 1024, -1, 64]);  view_105 = None
        permute_48: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_49: "f32[768, 768]" = torch.ops.aten.permute.default(primals_39, [1, 0])
        mm_26: "f32[8192, 768]" = torch.ops.aten.mm.default(view_101, permute_49);  permute_49 = None
        view_108: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_26, [8, 1024, 768]);  mm_26 = None
        view_109: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_108, [8, 1024, -1, 64]);  view_108 = None
        permute_50: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_109, [0, 2, 1, 3]);  view_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_51: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_48, [0, 1, 3, 2]);  permute_48 = None
        expand_17: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_46, [8, 12, 1024, 64]);  permute_46 = None
        clone_16: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_110: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_16, [96, 1024, 64]);  clone_16 = None
        expand_18: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_51, [8, 12, 64, 1024]);  permute_51 = None
        clone_17: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_111: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_17, [96, 64, 1024]);  clone_17 = None
        bmm_8: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_110, view_111)
        view_112: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_8, [8, 12, 1024, 1024]);  bmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_28: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_112, add_7);  view_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_4: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_28, [-1], True)
        sub_5: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_28, amax_4);  add_28 = amax_4 = None
        exp_4: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        sum_5: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_6: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_17: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 17)
        inductor_random_default_106: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_17, 'rand');  inductor_lookup_seed_default_17 = None
        gt_18: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_106, 0.1);  inductor_random_default_106 = None
        mul_54: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_18, div_6)
        mul_55: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_54, 1.1111111111111112);  mul_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_19: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_55, [8, 12, 1024, 1024]);  mul_55 = None
        view_116: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_19, [96, 1024, 1024]);  expand_19 = None
        expand_20: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_50, [8, 12, 1024, 64]);  permute_50 = None
        clone_18: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_117: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_18, [96, 1024, 64]);  clone_18 = None
        bmm_9: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_116, view_117)
        view_118: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_9, [8, 12, 1024, 64]);  bmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_52: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
        clone_19: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_119: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_19, [8, 1024, -1]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_53: "f32[768, 768]" = torch.ops.aten.permute.default(primals_40, [1, 0])
        view_120: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_119, [8192, 768]);  view_119 = None
        mm_27: "f32[8192, 768]" = torch.ops.aten.mm.default(view_120, permute_53);  permute_53 = None
        view_121: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_27, [8, 1024, 768]);  mm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_18: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 18)
        inductor_random_default_105: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_18, 'rand');  inductor_lookup_seed_default_18 = None
        gt_19: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_105, 0.1);  inductor_random_default_105 = None
        mul_56: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_19, view_121);  view_121 = None
        mul_57: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_56, 1.1111111111111112);  mul_56 = None
        add_29: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_26, mul_57);  mul_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_10: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_29, 2)
        mean_9: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_30: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_9, 1e-06);  mean_9 = None
        rsqrt_9: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_58: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_29, rsqrt_9)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_59: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_41, mul_58);  mul_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_54: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_42, [1, 0])
        view_122: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_59, [8192, 768]);  mul_59 = None
        mm_28: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_122, permute_54);  permute_54 = None
        view_123: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_28, [8, 1024, 3072]);  mm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_4: "f32[8, 1024, 3072]" = torch.ops.aten.relu.default(view_123);  view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_19: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 19)
        inductor_random_default_104: "f32[8, 1024, 3072]" = torch.ops.prims.inductor_random.default([8, 1024, 3072], inductor_lookup_seed_default_19, 'rand');  inductor_lookup_seed_default_19 = None
        gt_20: "b8[8, 1024, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_104, 0.1);  inductor_random_default_104 = None
        mul_60: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(gt_20, relu_4)
        mul_61: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_60, 1.1111111111111112);  mul_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_55: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_43, [1, 0])
        view_124: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_61, [8192, 3072]);  mul_61 = None
        mm_29: "f32[8192, 768]" = torch.ops.aten.mm.default(view_124, permute_55);  permute_55 = None
        view_125: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_29, [8, 1024, 768]);  mm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_20: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 20)
        inductor_random_default_103: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_20, 'rand');  inductor_lookup_seed_default_20 = None
        gt_21: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_103, 0.1);  inductor_random_default_103 = None
        mul_62: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_21, view_125);  view_125 = None
        mul_63: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_62, 1.1111111111111112);  mul_62 = None
        add_31: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_29, mul_63);  mul_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_11: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_31, 2)
        mean_10: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_32: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_10, 1e-06);  mean_10 = None
        rsqrt_10: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        mul_64: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_31, rsqrt_10)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_65: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_44, mul_64);  mul_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_56: "f32[768, 768]" = torch.ops.aten.permute.default(primals_45, [1, 0])
        view_126: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_65, [8192, 768]);  mul_65 = None
        mm_30: "f32[8192, 768]" = torch.ops.aten.mm.default(view_126, permute_56);  permute_56 = None
        view_127: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_30, [8, 1024, 768]);  mm_30 = None
        view_128: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_127, [8, 1024, -1, 64]);  view_127 = None
        permute_57: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_128, [0, 2, 1, 3]);  view_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_58: "f32[768, 768]" = torch.ops.aten.permute.default(primals_46, [1, 0])
        mm_31: "f32[8192, 768]" = torch.ops.aten.mm.default(view_126, permute_58);  permute_58 = None
        view_130: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_31, [8, 1024, 768]);  mm_31 = None
        view_131: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_130, [8, 1024, -1, 64]);  view_130 = None
        permute_59: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_131, [0, 2, 1, 3]);  view_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_60: "f32[768, 768]" = torch.ops.aten.permute.default(primals_47, [1, 0])
        mm_32: "f32[8192, 768]" = torch.ops.aten.mm.default(view_126, permute_60);  permute_60 = None
        view_133: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_32, [8, 1024, 768]);  mm_32 = None
        view_134: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_133, [8, 1024, -1, 64]);  view_133 = None
        permute_61: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_62: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_59, [0, 1, 3, 2]);  permute_59 = None
        expand_21: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_57, [8, 12, 1024, 64]);  permute_57 = None
        clone_20: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_135: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_20, [96, 1024, 64]);  clone_20 = None
        expand_22: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_62, [8, 12, 64, 1024]);  permute_62 = None
        clone_21: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_136: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_21, [96, 64, 1024]);  clone_21 = None
        bmm_10: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_135, view_136)
        view_137: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_10, [8, 12, 1024, 1024]);  bmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_33: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_137, add_7);  view_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_5: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_33, [-1], True)
        sub_6: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_33, amax_5);  add_33 = amax_5 = None
        exp_5: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_6: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_7: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_21: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 21)
        inductor_random_default_102: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_21, 'rand');  inductor_lookup_seed_default_21 = None
        gt_22: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_102, 0.1);  inductor_random_default_102 = None
        mul_66: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_22, div_7)
        mul_67: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_66, 1.1111111111111112);  mul_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_23: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_67, [8, 12, 1024, 1024]);  mul_67 = None
        view_141: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_23, [96, 1024, 1024]);  expand_23 = None
        expand_24: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_61, [8, 12, 1024, 64]);  permute_61 = None
        clone_22: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_142: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_22, [96, 1024, 64]);  clone_22 = None
        bmm_11: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_141, view_142)
        view_143: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_11, [8, 12, 1024, 64]);  bmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_63: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_143, [0, 2, 1, 3]);  view_143 = None
        clone_23: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_144: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_23, [8, 1024, -1]);  clone_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_64: "f32[768, 768]" = torch.ops.aten.permute.default(primals_48, [1, 0])
        view_145: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_144, [8192, 768]);  view_144 = None
        mm_33: "f32[8192, 768]" = torch.ops.aten.mm.default(view_145, permute_64);  permute_64 = None
        view_146: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_33, [8, 1024, 768]);  mm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_22: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 22)
        inductor_random_default_101: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_22, 'rand');  inductor_lookup_seed_default_22 = None
        gt_23: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_101, 0.1);  inductor_random_default_101 = None
        mul_68: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_23, view_146);  view_146 = None
        mul_69: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_68, 1.1111111111111112);  mul_68 = None
        add_34: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_31, mul_69);  mul_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_12: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_34, 2)
        mean_11: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_12, [-1], True);  pow_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_35: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_11, 1e-06);  mean_11 = None
        rsqrt_11: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        mul_70: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_34, rsqrt_11)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_71: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_49, mul_70);  mul_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_65: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_50, [1, 0])
        view_147: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_71, [8192, 768]);  mul_71 = None
        mm_34: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_147, permute_65);  permute_65 = None
        view_148: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_34, [8, 1024, 3072]);  mm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_5: "f32[8, 1024, 3072]" = torch.ops.aten.relu.default(view_148);  view_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_23: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 23)
        inductor_random_default_100: "f32[8, 1024, 3072]" = torch.ops.prims.inductor_random.default([8, 1024, 3072], inductor_lookup_seed_default_23, 'rand');  inductor_lookup_seed_default_23 = None
        gt_24: "b8[8, 1024, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_100, 0.1);  inductor_random_default_100 = None
        mul_72: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(gt_24, relu_5)
        mul_73: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_72, 1.1111111111111112);  mul_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_66: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_51, [1, 0])
        view_149: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_73, [8192, 3072]);  mul_73 = None
        mm_35: "f32[8192, 768]" = torch.ops.aten.mm.default(view_149, permute_66);  permute_66 = None
        view_150: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_35, [8, 1024, 768]);  mm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_24: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 24)
        inductor_random_default_99: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_24, 'rand');  inductor_lookup_seed_default_24 = None
        gt_25: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_99, 0.1);  inductor_random_default_99 = None
        mul_74: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_25, view_150);  view_150 = None
        mul_75: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_74, 1.1111111111111112);  mul_74 = None
        add_36: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_34, mul_75);  mul_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_13: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_36, 2)
        mean_12: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_37: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_12, 1e-06);  mean_12 = None
        rsqrt_12: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        mul_76: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_36, rsqrt_12)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_77: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_52, mul_76);  mul_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_67: "f32[768, 768]" = torch.ops.aten.permute.default(primals_53, [1, 0])
        view_151: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_77, [8192, 768]);  mul_77 = None
        mm_36: "f32[8192, 768]" = torch.ops.aten.mm.default(view_151, permute_67);  permute_67 = None
        view_152: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_36, [8, 1024, 768]);  mm_36 = None
        view_153: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_152, [8, 1024, -1, 64]);  view_152 = None
        permute_68: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_153, [0, 2, 1, 3]);  view_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_69: "f32[768, 768]" = torch.ops.aten.permute.default(primals_54, [1, 0])
        mm_37: "f32[8192, 768]" = torch.ops.aten.mm.default(view_151, permute_69);  permute_69 = None
        view_155: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_37, [8, 1024, 768]);  mm_37 = None
        view_156: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_155, [8, 1024, -1, 64]);  view_155 = None
        permute_70: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_71: "f32[768, 768]" = torch.ops.aten.permute.default(primals_55, [1, 0])
        mm_38: "f32[8192, 768]" = torch.ops.aten.mm.default(view_151, permute_71);  permute_71 = None
        view_158: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_38, [8, 1024, 768]);  mm_38 = None
        view_159: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_158, [8, 1024, -1, 64]);  view_158 = None
        permute_72: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_159, [0, 2, 1, 3]);  view_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_73: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_70, [0, 1, 3, 2]);  permute_70 = None
        expand_25: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_68, [8, 12, 1024, 64]);  permute_68 = None
        clone_24: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_160: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_24, [96, 1024, 64]);  clone_24 = None
        expand_26: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_73, [8, 12, 64, 1024]);  permute_73 = None
        clone_25: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_161: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_25, [96, 64, 1024]);  clone_25 = None
        bmm_12: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_160, view_161)
        view_162: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_12, [8, 12, 1024, 1024]);  bmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_38: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_162, add_7);  view_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_6: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_38, [-1], True)
        sub_7: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_38, amax_6);  add_38 = amax_6 = None
        exp_6: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_7: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_8: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_25: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 25)
        inductor_random_default_98: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_25, 'rand');  inductor_lookup_seed_default_25 = None
        gt_26: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_98, 0.1);  inductor_random_default_98 = None
        mul_78: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_26, div_8)
        mul_79: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_78, 1.1111111111111112);  mul_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_27: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_79, [8, 12, 1024, 1024]);  mul_79 = None
        view_166: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_27, [96, 1024, 1024]);  expand_27 = None
        expand_28: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_72, [8, 12, 1024, 64]);  permute_72 = None
        clone_26: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_167: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_26, [96, 1024, 64]);  clone_26 = None
        bmm_13: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_166, view_167)
        view_168: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_13, [8, 12, 1024, 64]);  bmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_74: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        clone_27: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format);  permute_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_169: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_27, [8, 1024, -1]);  clone_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_75: "f32[768, 768]" = torch.ops.aten.permute.default(primals_56, [1, 0])
        view_170: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_169, [8192, 768]);  view_169 = None
        mm_39: "f32[8192, 768]" = torch.ops.aten.mm.default(view_170, permute_75);  permute_75 = None
        view_171: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_39, [8, 1024, 768]);  mm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_26: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 26)
        inductor_random_default_97: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_26, 'rand');  inductor_lookup_seed_default_26 = None
        gt_27: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_97, 0.1);  inductor_random_default_97 = None
        mul_80: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_27, view_171);  view_171 = None
        mul_81: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_80, 1.1111111111111112);  mul_80 = None
        add_39: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_36, mul_81);  mul_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_14: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_39, 2)
        mean_13: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_14, [-1], True);  pow_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_40: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_13, 1e-06);  mean_13 = None
        rsqrt_13: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        mul_82: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_39, rsqrt_13)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_83: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_57, mul_82);  mul_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_76: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_58, [1, 0])
        view_172: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_83, [8192, 768]);  mul_83 = None
        mm_40: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_172, permute_76);  permute_76 = None
        view_173: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_40, [8, 1024, 3072]);  mm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_6: "f32[8, 1024, 3072]" = torch.ops.aten.relu.default(view_173);  view_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_27: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 27)
        inductor_random_default_96: "f32[8, 1024, 3072]" = torch.ops.prims.inductor_random.default([8, 1024, 3072], inductor_lookup_seed_default_27, 'rand');  inductor_lookup_seed_default_27 = None
        gt_28: "b8[8, 1024, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_96, 0.1);  inductor_random_default_96 = None
        mul_84: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(gt_28, relu_6)
        mul_85: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_84, 1.1111111111111112);  mul_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_77: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_59, [1, 0])
        view_174: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_85, [8192, 3072]);  mul_85 = None
        mm_41: "f32[8192, 768]" = torch.ops.aten.mm.default(view_174, permute_77);  permute_77 = None
        view_175: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_41, [8, 1024, 768]);  mm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_28: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 28)
        inductor_random_default_95: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_28, 'rand');  inductor_lookup_seed_default_28 = None
        gt_29: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_95, 0.1);  inductor_random_default_95 = None
        mul_86: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_29, view_175);  view_175 = None
        mul_87: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_86, 1.1111111111111112);  mul_86 = None
        add_41: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_39, mul_87);  mul_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_15: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_41, 2)
        mean_14: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_15, [-1], True);  pow_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_42: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_14, 1e-06);  mean_14 = None
        rsqrt_14: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_88: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_41, rsqrt_14)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_89: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_60, mul_88);  mul_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_78: "f32[768, 768]" = torch.ops.aten.permute.default(primals_61, [1, 0])
        view_176: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_89, [8192, 768]);  mul_89 = None
        mm_42: "f32[8192, 768]" = torch.ops.aten.mm.default(view_176, permute_78);  permute_78 = None
        view_177: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_42, [8, 1024, 768]);  mm_42 = None
        view_178: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_177, [8, 1024, -1, 64]);  view_177 = None
        permute_79: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_80: "f32[768, 768]" = torch.ops.aten.permute.default(primals_62, [1, 0])
        mm_43: "f32[8192, 768]" = torch.ops.aten.mm.default(view_176, permute_80);  permute_80 = None
        view_180: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_43, [8, 1024, 768]);  mm_43 = None
        view_181: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_180, [8, 1024, -1, 64]);  view_180 = None
        permute_81: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_82: "f32[768, 768]" = torch.ops.aten.permute.default(primals_63, [1, 0])
        mm_44: "f32[8192, 768]" = torch.ops.aten.mm.default(view_176, permute_82);  permute_82 = None
        view_183: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_44, [8, 1024, 768]);  mm_44 = None
        view_184: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_183, [8, 1024, -1, 64]);  view_183 = None
        permute_83: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_84: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_81, [0, 1, 3, 2]);  permute_81 = None
        expand_29: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_79, [8, 12, 1024, 64]);  permute_79 = None
        clone_28: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_185: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_28, [96, 1024, 64]);  clone_28 = None
        expand_30: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_84, [8, 12, 64, 1024]);  permute_84 = None
        clone_29: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_186: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_29, [96, 64, 1024]);  clone_29 = None
        bmm_14: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_185, view_186)
        view_187: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_14, [8, 12, 1024, 1024]);  bmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_43: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_187, add_7);  view_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_7: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_43, [-1], True)
        sub_8: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_43, amax_7);  add_43 = amax_7 = None
        exp_7: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
        sum_8: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_9: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_29: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 29)
        inductor_random_default_94: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_29, 'rand');  inductor_lookup_seed_default_29 = None
        gt_30: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_94, 0.1);  inductor_random_default_94 = None
        mul_90: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_30, div_9)
        mul_91: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_90, 1.1111111111111112);  mul_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_31: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_91, [8, 12, 1024, 1024]);  mul_91 = None
        view_191: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_31, [96, 1024, 1024]);  expand_31 = None
        expand_32: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_83, [8, 12, 1024, 64]);  permute_83 = None
        clone_30: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_192: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_30, [96, 1024, 64]);  clone_30 = None
        bmm_15: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_191, view_192)
        view_193: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_15, [8, 12, 1024, 64]);  bmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_85: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_193, [0, 2, 1, 3]);  view_193 = None
        clone_31: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_194: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_31, [8, 1024, -1]);  clone_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_86: "f32[768, 768]" = torch.ops.aten.permute.default(primals_64, [1, 0])
        view_195: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_194, [8192, 768]);  view_194 = None
        mm_45: "f32[8192, 768]" = torch.ops.aten.mm.default(view_195, permute_86);  permute_86 = None
        view_196: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_45, [8, 1024, 768]);  mm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_30: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 30)
        inductor_random_default_93: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_30, 'rand');  inductor_lookup_seed_default_30 = None
        gt_31: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_93, 0.1);  inductor_random_default_93 = None
        mul_92: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_31, view_196);  view_196 = None
        mul_93: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_92, 1.1111111111111112);  mul_92 = None
        add_44: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_41, mul_93);  mul_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_16: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_44, 2)
        mean_15: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_45: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_15, 1e-06);  mean_15 = None
        rsqrt_15: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        mul_94: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_44, rsqrt_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_95: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_65, mul_94);  mul_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_87: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_66, [1, 0])
        view_197: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_95, [8192, 768]);  mul_95 = None
        mm_46: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_197, permute_87);  permute_87 = None
        view_198: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_46, [8, 1024, 3072]);  mm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_7: "f32[8, 1024, 3072]" = torch.ops.aten.relu.default(view_198);  view_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_31: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 31)
        inductor_random_default_92: "f32[8, 1024, 3072]" = torch.ops.prims.inductor_random.default([8, 1024, 3072], inductor_lookup_seed_default_31, 'rand');  inductor_lookup_seed_default_31 = None
        gt_32: "b8[8, 1024, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_92, 0.1);  inductor_random_default_92 = None
        mul_96: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(gt_32, relu_7)
        mul_97: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_96, 1.1111111111111112);  mul_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_88: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_67, [1, 0])
        view_199: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_97, [8192, 3072]);  mul_97 = None
        mm_47: "f32[8192, 768]" = torch.ops.aten.mm.default(view_199, permute_88);  permute_88 = None
        view_200: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_47, [8, 1024, 768]);  mm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_32: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 32)
        inductor_random_default_91: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_32, 'rand');  inductor_lookup_seed_default_32 = None
        gt_33: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_91, 0.1);  inductor_random_default_91 = None
        mul_98: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_33, view_200);  view_200 = None
        mul_99: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_98, 1.1111111111111112);  mul_98 = None
        add_46: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_44, mul_99);  mul_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_17: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_46, 2)
        mean_16: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_47: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_16, 1e-06);  mean_16 = None
        rsqrt_16: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        mul_100: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_46, rsqrt_16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_101: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_68, mul_100);  mul_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_89: "f32[768, 768]" = torch.ops.aten.permute.default(primals_69, [1, 0])
        view_201: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_101, [8192, 768]);  mul_101 = None
        mm_48: "f32[8192, 768]" = torch.ops.aten.mm.default(view_201, permute_89);  permute_89 = None
        view_202: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_48, [8, 1024, 768]);  mm_48 = None
        view_203: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_202, [8, 1024, -1, 64]);  view_202 = None
        permute_90: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_91: "f32[768, 768]" = torch.ops.aten.permute.default(primals_70, [1, 0])
        mm_49: "f32[8192, 768]" = torch.ops.aten.mm.default(view_201, permute_91);  permute_91 = None
        view_205: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_49, [8, 1024, 768]);  mm_49 = None
        view_206: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_205, [8, 1024, -1, 64]);  view_205 = None
        permute_92: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_93: "f32[768, 768]" = torch.ops.aten.permute.default(primals_71, [1, 0])
        mm_50: "f32[8192, 768]" = torch.ops.aten.mm.default(view_201, permute_93);  permute_93 = None
        view_208: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_50, [8, 1024, 768]);  mm_50 = None
        view_209: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_208, [8, 1024, -1, 64]);  view_208 = None
        permute_94: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_209, [0, 2, 1, 3]);  view_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_95: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_92, [0, 1, 3, 2]);  permute_92 = None
        expand_33: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_90, [8, 12, 1024, 64]);  permute_90 = None
        clone_32: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_210: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_32, [96, 1024, 64]);  clone_32 = None
        expand_34: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_95, [8, 12, 64, 1024]);  permute_95 = None
        clone_33: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_211: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_33, [96, 64, 1024]);  clone_33 = None
        bmm_16: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_210, view_211)
        view_212: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_16, [8, 12, 1024, 1024]);  bmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_48: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_212, add_7);  view_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_8: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_48, [-1], True)
        sub_9: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_48, amax_8);  add_48 = amax_8 = None
        exp_8: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        sum_9: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_10: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_33: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 33)
        inductor_random_default_90: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_33, 'rand');  inductor_lookup_seed_default_33 = None
        gt_34: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_90, 0.1);  inductor_random_default_90 = None
        mul_102: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_34, div_10)
        mul_103: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_102, 1.1111111111111112);  mul_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_35: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_103, [8, 12, 1024, 1024]);  mul_103 = None
        view_216: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_35, [96, 1024, 1024]);  expand_35 = None
        expand_36: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_94, [8, 12, 1024, 64]);  permute_94 = None
        clone_34: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_217: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_34, [96, 1024, 64]);  clone_34 = None
        bmm_17: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_216, view_217)
        view_218: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_17, [8, 12, 1024, 64]);  bmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_96: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_218, [0, 2, 1, 3]);  view_218 = None
        clone_35: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_96, memory_format = torch.contiguous_format);  permute_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_219: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_35, [8, 1024, -1]);  clone_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_97: "f32[768, 768]" = torch.ops.aten.permute.default(primals_72, [1, 0])
        view_220: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_219, [8192, 768]);  view_219 = None
        mm_51: "f32[8192, 768]" = torch.ops.aten.mm.default(view_220, permute_97);  permute_97 = None
        view_221: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_51, [8, 1024, 768]);  mm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_34: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 34)
        inductor_random_default_89: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_34, 'rand');  inductor_lookup_seed_default_34 = None
        gt_35: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_89, 0.1);  inductor_random_default_89 = None
        mul_104: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_35, view_221);  view_221 = None
        mul_105: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_104, 1.1111111111111112);  mul_104 = None
        add_49: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_46, mul_105);  mul_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_18: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_49, 2)
        mean_17: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_18, [-1], True);  pow_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_50: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_17, 1e-06);  mean_17 = None
        rsqrt_17: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        mul_106: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_49, rsqrt_17)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_107: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_73, mul_106);  mul_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_98: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_74, [1, 0])
        view_222: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_107, [8192, 768]);  mul_107 = None
        mm_52: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_222, permute_98);  permute_98 = None
        view_223: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_52, [8, 1024, 3072]);  mm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_8: "f32[8, 1024, 3072]" = torch.ops.aten.relu.default(view_223);  view_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_35: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 35)
        inductor_random_default_88: "f32[8, 1024, 3072]" = torch.ops.prims.inductor_random.default([8, 1024, 3072], inductor_lookup_seed_default_35, 'rand');  inductor_lookup_seed_default_35 = None
        gt_36: "b8[8, 1024, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_88, 0.1);  inductor_random_default_88 = None
        mul_108: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(gt_36, relu_8)
        mul_109: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_108, 1.1111111111111112);  mul_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_99: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_75, [1, 0])
        view_224: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_109, [8192, 3072]);  mul_109 = None
        mm_53: "f32[8192, 768]" = torch.ops.aten.mm.default(view_224, permute_99);  permute_99 = None
        view_225: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_53, [8, 1024, 768]);  mm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_36: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 36)
        inductor_random_default_87: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_36, 'rand');  inductor_lookup_seed_default_36 = None
        gt_37: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_87, 0.1);  inductor_random_default_87 = None
        mul_110: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_37, view_225);  view_225 = None
        mul_111: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_110, 1.1111111111111112);  mul_110 = None
        add_51: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_49, mul_111);  mul_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_19: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_51, 2)
        mean_18: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_52: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_18, 1e-06);  mean_18 = None
        rsqrt_18: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_112: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_51, rsqrt_18)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_113: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_76, mul_112);  mul_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_100: "f32[768, 768]" = torch.ops.aten.permute.default(primals_77, [1, 0])
        view_226: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_113, [8192, 768]);  mul_113 = None
        mm_54: "f32[8192, 768]" = torch.ops.aten.mm.default(view_226, permute_100);  permute_100 = None
        view_227: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_54, [8, 1024, 768]);  mm_54 = None
        view_228: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_227, [8, 1024, -1, 64]);  view_227 = None
        permute_101: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_102: "f32[768, 768]" = torch.ops.aten.permute.default(primals_78, [1, 0])
        mm_55: "f32[8192, 768]" = torch.ops.aten.mm.default(view_226, permute_102);  permute_102 = None
        view_230: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_55, [8, 1024, 768]);  mm_55 = None
        view_231: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_230, [8, 1024, -1, 64]);  view_230 = None
        permute_103: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_231, [0, 2, 1, 3]);  view_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_104: "f32[768, 768]" = torch.ops.aten.permute.default(primals_79, [1, 0])
        mm_56: "f32[8192, 768]" = torch.ops.aten.mm.default(view_226, permute_104);  permute_104 = None
        view_233: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_56, [8, 1024, 768]);  mm_56 = None
        view_234: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_233, [8, 1024, -1, 64]);  view_233 = None
        permute_105: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_106: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_103, [0, 1, 3, 2]);  permute_103 = None
        expand_37: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_101, [8, 12, 1024, 64]);  permute_101 = None
        clone_36: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_235: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_36, [96, 1024, 64]);  clone_36 = None
        expand_38: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_106, [8, 12, 64, 1024]);  permute_106 = None
        clone_37: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_236: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_37, [96, 64, 1024]);  clone_37 = None
        bmm_18: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_235, view_236)
        view_237: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_18, [8, 12, 1024, 1024]);  bmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_53: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_237, add_7);  view_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_9: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_53, [-1], True)
        sub_10: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_53, amax_9);  add_53 = amax_9 = None
        exp_9: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_10: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_11: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_37: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 37)
        inductor_random_default_86: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_37, 'rand');  inductor_lookup_seed_default_37 = None
        gt_38: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_86, 0.1);  inductor_random_default_86 = None
        mul_114: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_38, div_11)
        mul_115: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_114, 1.1111111111111112);  mul_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_39: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_115, [8, 12, 1024, 1024]);  mul_115 = None
        view_241: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_39, [96, 1024, 1024]);  expand_39 = None
        expand_40: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_105, [8, 12, 1024, 64]);  permute_105 = None
        clone_38: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_242: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_38, [96, 1024, 64]);  clone_38 = None
        bmm_19: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_241, view_242)
        view_243: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_19, [8, 12, 1024, 64]);  bmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_107: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_243, [0, 2, 1, 3]);  view_243 = None
        clone_39: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_107, memory_format = torch.contiguous_format);  permute_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_244: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_39, [8, 1024, -1]);  clone_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_108: "f32[768, 768]" = torch.ops.aten.permute.default(primals_80, [1, 0])
        view_245: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_244, [8192, 768]);  view_244 = None
        mm_57: "f32[8192, 768]" = torch.ops.aten.mm.default(view_245, permute_108);  permute_108 = None
        view_246: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_57, [8, 1024, 768]);  mm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_38: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 38)
        inductor_random_default_85: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_38, 'rand');  inductor_lookup_seed_default_38 = None
        gt_39: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_85, 0.1);  inductor_random_default_85 = None
        mul_116: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_39, view_246);  view_246 = None
        mul_117: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_116, 1.1111111111111112);  mul_116 = None
        add_54: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_51, mul_117);  mul_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_20: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_54, 2)
        mean_19: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_20, [-1], True);  pow_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_55: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_19, 1e-06);  mean_19 = None
        rsqrt_19: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
        mul_118: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_54, rsqrt_19)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_119: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_81, mul_118);  mul_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_109: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_82, [1, 0])
        view_247: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_119, [8192, 768]);  mul_119 = None
        mm_58: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_247, permute_109);  permute_109 = None
        view_248: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_58, [8, 1024, 3072]);  mm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_9: "f32[8, 1024, 3072]" = torch.ops.aten.relu.default(view_248);  view_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_39: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 39)
        inductor_random_default_84: "f32[8, 1024, 3072]" = torch.ops.prims.inductor_random.default([8, 1024, 3072], inductor_lookup_seed_default_39, 'rand');  inductor_lookup_seed_default_39 = None
        gt_40: "b8[8, 1024, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_84, 0.1);  inductor_random_default_84 = None
        mul_120: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(gt_40, relu_9)
        mul_121: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_120, 1.1111111111111112);  mul_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_110: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_83, [1, 0])
        view_249: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_121, [8192, 3072]);  mul_121 = None
        mm_59: "f32[8192, 768]" = torch.ops.aten.mm.default(view_249, permute_110);  permute_110 = None
        view_250: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_59, [8, 1024, 768]);  mm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_40: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 40)
        inductor_random_default_83: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_40, 'rand');  inductor_lookup_seed_default_40 = None
        gt_41: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_83, 0.1);  inductor_random_default_83 = None
        mul_122: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_41, view_250);  view_250 = None
        mul_123: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_122, 1.1111111111111112);  mul_122 = None
        add_56: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_54, mul_123);  mul_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_21: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_56, 2)
        mean_20: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_21, [-1], True);  pow_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_57: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_20, 1e-06);  mean_20 = None
        rsqrt_20: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
        mul_124: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_56, rsqrt_20)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_125: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_84, mul_124);  mul_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_111: "f32[768, 768]" = torch.ops.aten.permute.default(primals_85, [1, 0])
        view_251: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_125, [8192, 768]);  mul_125 = None
        mm_60: "f32[8192, 768]" = torch.ops.aten.mm.default(view_251, permute_111);  permute_111 = None
        view_252: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_60, [8, 1024, 768]);  mm_60 = None
        view_253: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_252, [8, 1024, -1, 64]);  view_252 = None
        permute_112: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_253, [0, 2, 1, 3]);  view_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_113: "f32[768, 768]" = torch.ops.aten.permute.default(primals_86, [1, 0])
        mm_61: "f32[8192, 768]" = torch.ops.aten.mm.default(view_251, permute_113);  permute_113 = None
        view_255: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_61, [8, 1024, 768]);  mm_61 = None
        view_256: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_255, [8, 1024, -1, 64]);  view_255 = None
        permute_114: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_115: "f32[768, 768]" = torch.ops.aten.permute.default(primals_87, [1, 0])
        mm_62: "f32[8192, 768]" = torch.ops.aten.mm.default(view_251, permute_115);  permute_115 = None
        view_258: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_62, [8, 1024, 768]);  mm_62 = None
        view_259: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_258, [8, 1024, -1, 64]);  view_258 = None
        permute_116: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_259, [0, 2, 1, 3]);  view_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_117: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_114, [0, 1, 3, 2]);  permute_114 = None
        expand_41: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_112, [8, 12, 1024, 64]);  permute_112 = None
        clone_40: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_260: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_40, [96, 1024, 64]);  clone_40 = None
        expand_42: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_117, [8, 12, 64, 1024]);  permute_117 = None
        clone_41: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_261: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_41, [96, 64, 1024]);  clone_41 = None
        bmm_20: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_260, view_261)
        view_262: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_20, [8, 12, 1024, 1024]);  bmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_58: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_262, add_7);  view_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_10: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_58, [-1], True)
        sub_11: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_58, amax_10);  add_58 = amax_10 = None
        exp_10: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        sum_11: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_12: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_41: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 41)
        inductor_random_default_82: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_41, 'rand');  inductor_lookup_seed_default_41 = None
        gt_42: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_82, 0.1);  inductor_random_default_82 = None
        mul_126: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_42, div_12)
        mul_127: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_126, 1.1111111111111112);  mul_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_43: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_127, [8, 12, 1024, 1024]);  mul_127 = None
        view_266: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_43, [96, 1024, 1024]);  expand_43 = None
        expand_44: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_116, [8, 12, 1024, 64]);  permute_116 = None
        clone_42: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_267: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_42, [96, 1024, 64]);  clone_42 = None
        bmm_21: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_266, view_267)
        view_268: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_21, [8, 12, 1024, 64]);  bmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_118: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_268, [0, 2, 1, 3]);  view_268 = None
        clone_43: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_118, memory_format = torch.contiguous_format);  permute_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_269: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_43, [8, 1024, -1]);  clone_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_119: "f32[768, 768]" = torch.ops.aten.permute.default(primals_88, [1, 0])
        view_270: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_269, [8192, 768]);  view_269 = None
        mm_63: "f32[8192, 768]" = torch.ops.aten.mm.default(view_270, permute_119);  permute_119 = None
        view_271: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_63, [8, 1024, 768]);  mm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_42: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 42)
        inductor_random_default_81: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_42, 'rand');  inductor_lookup_seed_default_42 = None
        gt_43: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_81, 0.1);  inductor_random_default_81 = None
        mul_128: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_43, view_271);  view_271 = None
        mul_129: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_128, 1.1111111111111112);  mul_128 = None
        add_59: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_56, mul_129);  mul_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_22: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_59, 2)
        mean_21: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_22, [-1], True);  pow_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_60: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_21, 1e-06);  mean_21 = None
        rsqrt_21: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_130: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_59, rsqrt_21)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_131: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_89, mul_130);  mul_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_120: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_90, [1, 0])
        view_272: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_131, [8192, 768]);  mul_131 = None
        mm_64: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_272, permute_120);  permute_120 = None
        view_273: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_64, [8, 1024, 3072]);  mm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_10: "f32[8, 1024, 3072]" = torch.ops.aten.relu.default(view_273);  view_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_43: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 43)
        inductor_random_default_80: "f32[8, 1024, 3072]" = torch.ops.prims.inductor_random.default([8, 1024, 3072], inductor_lookup_seed_default_43, 'rand');  inductor_lookup_seed_default_43 = None
        gt_44: "b8[8, 1024, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_80, 0.1);  inductor_random_default_80 = None
        mul_132: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(gt_44, relu_10)
        mul_133: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_132, 1.1111111111111112);  mul_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_121: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_91, [1, 0])
        view_274: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_133, [8192, 3072]);  mul_133 = None
        mm_65: "f32[8192, 768]" = torch.ops.aten.mm.default(view_274, permute_121);  permute_121 = None
        view_275: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_65, [8, 1024, 768]);  mm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_44: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 44)
        inductor_random_default_79: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_44, 'rand');  inductor_lookup_seed_default_44 = None
        gt_45: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_79, 0.1);  inductor_random_default_79 = None
        mul_134: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_45, view_275);  view_275 = None
        mul_135: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_134, 1.1111111111111112);  mul_134 = None
        add_61: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_59, mul_135);  mul_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_23: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_61, 2)
        mean_22: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_23, [-1], True);  pow_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_62: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_22, 1e-06);  mean_22 = None
        rsqrt_22: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        mul_136: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_61, rsqrt_22)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_137: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_92, mul_136);  mul_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_122: "f32[768, 768]" = torch.ops.aten.permute.default(primals_93, [1, 0])
        view_276: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_137, [8192, 768]);  mul_137 = None
        mm_66: "f32[8192, 768]" = torch.ops.aten.mm.default(view_276, permute_122);  permute_122 = None
        view_277: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_66, [8, 1024, 768]);  mm_66 = None
        view_278: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_277, [8, 1024, -1, 64]);  view_277 = None
        permute_123: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_278, [0, 2, 1, 3]);  view_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_124: "f32[768, 768]" = torch.ops.aten.permute.default(primals_94, [1, 0])
        mm_67: "f32[8192, 768]" = torch.ops.aten.mm.default(view_276, permute_124);  permute_124 = None
        view_280: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_67, [8, 1024, 768]);  mm_67 = None
        view_281: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_280, [8, 1024, -1, 64]);  view_280 = None
        permute_125: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_281, [0, 2, 1, 3]);  view_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_126: "f32[768, 768]" = torch.ops.aten.permute.default(primals_95, [1, 0])
        mm_68: "f32[8192, 768]" = torch.ops.aten.mm.default(view_276, permute_126);  permute_126 = None
        view_283: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_68, [8, 1024, 768]);  mm_68 = None
        view_284: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_283, [8, 1024, -1, 64]);  view_283 = None
        permute_127: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_284, [0, 2, 1, 3]);  view_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_128: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_125, [0, 1, 3, 2]);  permute_125 = None
        expand_45: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_123, [8, 12, 1024, 64]);  permute_123 = None
        clone_44: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_285: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_44, [96, 1024, 64]);  clone_44 = None
        expand_46: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_128, [8, 12, 64, 1024]);  permute_128 = None
        clone_45: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_286: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_45, [96, 64, 1024]);  clone_45 = None
        bmm_22: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_285, view_286)
        view_287: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_22, [8, 12, 1024, 1024]);  bmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_63: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_287, add_7);  view_287 = add_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_11: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_63, [-1], True)
        sub_12: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_63, amax_11);  add_63 = amax_11 = None
        exp_11: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_12: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_13: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_45: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 45)
        inductor_random_default_78: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_45, 'rand');  inductor_lookup_seed_default_45 = None
        gt_46: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_78, 0.1);  inductor_random_default_78 = None
        mul_138: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_46, div_13)
        mul_139: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_138, 1.1111111111111112);  mul_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_47: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_139, [8, 12, 1024, 1024]);  mul_139 = None
        view_291: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_47, [96, 1024, 1024]);  expand_47 = None
        expand_48: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_127, [8, 12, 1024, 64]);  permute_127 = None
        clone_46: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_292: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_46, [96, 1024, 64]);  clone_46 = None
        bmm_23: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_291, view_292)
        view_293: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_23, [8, 12, 1024, 64]);  bmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_129: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_293, [0, 2, 1, 3]);  view_293 = None
        clone_47: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_294: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_47, [8, 1024, -1]);  clone_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_130: "f32[768, 768]" = torch.ops.aten.permute.default(primals_96, [1, 0])
        view_295: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_294, [8192, 768]);  view_294 = None
        mm_69: "f32[8192, 768]" = torch.ops.aten.mm.default(view_295, permute_130);  permute_130 = None
        view_296: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_69, [8, 1024, 768]);  mm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_46: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 46)
        inductor_random_default_77: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_46, 'rand');  inductor_lookup_seed_default_46 = None
        gt_47: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_77, 0.1);  inductor_random_default_77 = None
        mul_140: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_47, view_296);  view_296 = None
        mul_141: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_140, 1.1111111111111112);  mul_140 = None
        add_64: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_61, mul_141);  mul_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_24: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_64, 2)
        mean_23: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_24, [-1], True);  pow_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_65: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_23, 1e-06);  mean_23 = None
        rsqrt_23: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        mul_142: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_64, rsqrt_23)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_143: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_97, mul_142);  mul_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_131: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_98, [1, 0])
        view_297: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_143, [8192, 768]);  mul_143 = None
        mm_70: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_297, permute_131);  permute_131 = None
        view_298: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_70, [8, 1024, 3072]);  mm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_11: "f32[8, 1024, 3072]" = torch.ops.aten.relu.default(view_298);  view_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_47: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 47)
        inductor_random_default_76: "f32[8, 1024, 3072]" = torch.ops.prims.inductor_random.default([8, 1024, 3072], inductor_lookup_seed_default_47, 'rand');  inductor_lookup_seed_default_47 = None
        gt_48: "b8[8, 1024, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_76, 0.1);  inductor_random_default_76 = None
        mul_144: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(gt_48, relu_11)
        mul_145: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_144, 1.1111111111111112);  mul_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_132: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_99, [1, 0])
        view_299: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_145, [8192, 3072]);  mul_145 = None
        mm_71: "f32[8192, 768]" = torch.ops.aten.mm.default(view_299, permute_132);  permute_132 = None
        view_300: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_71, [8, 1024, 768]);  mm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_48: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 48)
        inductor_random_default_75: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_48, 'rand');  inductor_lookup_seed_default_48 = None
        gt_49: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_75, 0.1);  inductor_random_default_75 = None
        mul_146: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_49, view_300);  view_300 = None
        mul_147: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_146, 1.1111111111111112);  mul_146 = None
        add_66: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_64, mul_147);  mul_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_25: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_66, 2)
        mean_24: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_67: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_24, 1e-06);  mean_24 = None
        rsqrt_24: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
        mul_148: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_66, rsqrt_24)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_149: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_100, mul_148);  mul_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:755 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_49: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 49)
        inductor_random_default_74: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_49, 'rand');  inductor_lookup_seed_default_49 = None
        gt_50: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_74, 0.1);  inductor_random_default_74 = None
        mul_150: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_50, mul_149);  mul_149 = None
        mul_151: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_150, 1.1111111111111112);  mul_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:592 in _shift_right, code: shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        full_1: "i64[8, 1024]" = torch.ops.aten.full.default([8, 1024], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:593 in _shift_right, code: shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        slice_1: "i64[8, 1023]" = torch.ops.aten.slice.Tensor(primals_101, 1, 0, -1)
        clone_48: "i64[8, 1023]" = torch.ops.aten.clone.default(slice_1);  slice_1 = None
        slice_2: "i64[8, 1023]" = torch.ops.aten.slice.Tensor(full_1, 1, 1, 9223372036854775807)
        copy: "i64[8, 1023]" = torch.ops.aten.copy.default(slice_2, clone_48);  slice_2 = clone_48 = None
        slice_scatter: "i64[8, 1024]" = torch.ops.aten.slice_scatter.default(full_1, copy, 1, 1, 9223372036854775807);  full_1 = copy = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:594 in _shift_right, code: shifted_input_ids[..., 0] = decoder_start_token_id
        full_default_3: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        select_1: "i64[8]" = torch.ops.aten.select.int(slice_scatter, 1, 0)
        copy_1: "i64[8]" = torch.ops.aten.copy.default(select_1, full_default_3);  select_1 = full_default_3 = None
        select_scatter: "i64[8, 1024]" = torch.ops.aten.select_scatter.default(slice_scatter, copy_1, 1, 0);  slice_scatter = copy_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:599 in _shift_right, code: shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        eq: "b8[8, 1024]" = torch.ops.aten.eq.Scalar(select_scatter, -100)
        full_default_4: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "i64[8, 1024]" = torch.ops.aten.where.self(eq, full_default_4, select_scatter);  eq = select_scatter = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:669 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding_2: "f32[8, 1024, 768]" = torch.ops.aten.embedding.default(primals_2, where_2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_11: "i64[1, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 1024, 1024]" = torch.ops.aten.le.Tensor(unsqueeze_11, unsqueeze_2);  unsqueeze_11 = unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_49: "b8[8, 1, 1024, 1024]" = torch.ops.aten.expand.default(le, [8, -1, 1024, 1024]);  le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        where_3: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_49, full_default, full_default_1);  expand_49 = full_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:721 in forward, code: hidden_states = self.dropout(inputs_embeds)
        inductor_lookup_seed_default_50: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 50)
        inductor_random_default_73: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_50, 'rand');  inductor_lookup_seed_default_50 = None
        gt_51: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_73, 0.1);  inductor_random_default_73 = None
        mul_152: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_51, embedding_2)
        mul_153: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_152, 1.1111111111111112);  mul_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_26: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(mul_153, 2)
        mean_25: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_26, [-1], True);  pow_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_72: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_25, 1e-06);  mean_25 = None
        rsqrt_25: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        mul_154: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_153, rsqrt_25)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_155: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_102, mul_154);  mul_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_133: "f32[768, 768]" = torch.ops.aten.permute.default(primals_103, [1, 0])
        view_303: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_155, [8192, 768]);  mul_155 = None
        mm_72: "f32[8192, 768]" = torch.ops.aten.mm.default(view_303, permute_133);  permute_133 = None
        view_304: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_72, [8, 1024, 768]);  mm_72 = None
        view_305: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_304, [8, 1024, -1, 64]);  view_304 = None
        permute_134: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_305, [0, 2, 1, 3]);  view_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_135: "f32[768, 768]" = torch.ops.aten.permute.default(primals_104, [1, 0])
        mm_73: "f32[8192, 768]" = torch.ops.aten.mm.default(view_303, permute_135);  permute_135 = None
        view_307: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_73, [8, 1024, 768]);  mm_73 = None
        view_308: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_307, [8, 1024, -1, 64]);  view_307 = None
        permute_136: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_308, [0, 2, 1, 3]);  view_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_137: "f32[768, 768]" = torch.ops.aten.permute.default(primals_105, [1, 0])
        mm_74: "f32[8192, 768]" = torch.ops.aten.mm.default(view_303, permute_137);  permute_137 = None
        view_310: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_74, [8, 1024, 768]);  mm_74 = None
        view_311: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_310, [8, 1024, -1, 64]);  view_310 = None
        permute_138: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_311, [0, 2, 1, 3]);  view_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_139: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_136, [0, 1, 3, 2]);  permute_136 = None
        expand_51: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_134, [8, 12, 1024, 64]);  permute_134 = None
        clone_49: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        view_312: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_49, [96, 1024, 64]);  clone_49 = None
        expand_52: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_139, [8, 12, 64, 1024]);  permute_139 = None
        clone_50: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_313: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_50, [96, 64, 1024]);  clone_50 = None
        bmm_24: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_312, view_313)
        view_314: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_24, [8, 12, 1024, 1024]);  bmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:203 in _relative_position_bucket, code: relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        full_default_9: "i64[1024, 1024]" = torch.ops.aten.full.default([1024, 1024], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_1: "i64[1024, 1024]" = torch.ops.aten.minimum.default(sub, full_default_9);  sub = full_default_9 = None
        neg: "i64[1024, 1024]" = torch.ops.aten.neg.default(minimum_1);  minimum_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:208 in _relative_position_bucket, code: is_small = relative_position < max_exact
        lt_1: "b8[1024, 1024]" = torch.ops.aten.lt.Scalar(neg, 16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:212 in _relative_position_bucket, code: torch.log(relative_position.float() / max_exact)
        convert_element_type_3: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(neg, torch.float32)
        div_14: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(convert_element_type_3, 16);  convert_element_type_3 = None
        log_1: "f32[1024, 1024]" = torch.ops.aten.log.default(div_14);  div_14 = None
        div_15: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(log_1, 2.0794415416798357);  log_1 = None
        mul_156: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(div_15, 16);  div_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:215 in _relative_position_bucket, code: ).to(torch.long)
        convert_element_type_4: "i64[1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_156, torch.int64);  mul_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:211 in _relative_position_bucket, code: relative_position_if_large = max_exact + (
        add_74: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_4, 16);  convert_element_type_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:217 in _relative_position_bucket, code: relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        full_default_10: "i64[1024, 1024]" = torch.ops.aten.full.default([1024, 1024], 31, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:216 in _relative_position_bucket, code: relative_position_if_large = torch.min(
        minimum_2: "i64[1024, 1024]" = torch.ops.aten.minimum.default(add_74, full_default_10);  add_74 = full_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:220 in _relative_position_bucket, code: relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        where_5: "i64[1024, 1024]" = torch.ops.aten.where.self(lt_1, neg, minimum_2);  lt_1 = neg = minimum_2 = None
        add_75: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(where_5, 0);  where_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:236 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        embedding_3: "f32[1024, 1024, 12]" = torch.ops.aten.embedding.default(primals_106, add_75);  primals_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:237 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        permute_140: "f32[12, 1024, 1024]" = torch.ops.aten.permute.default(embedding_3, [2, 0, 1]);  embedding_3 = None
        unsqueeze_17: "f32[1, 12, 1024, 1024]" = torch.ops.aten.unsqueeze.default(permute_140, 0);  permute_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        add_76: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_17, where_3);  unsqueeze_17 = where_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_77: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_314, add_76);  view_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_12: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_77, [-1], True)
        sub_14: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_77, amax_12);  add_77 = amax_12 = None
        exp_12: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
        sum_13: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_16: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_51: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 51)
        inductor_random_default_72: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_51, 'rand');  inductor_lookup_seed_default_51 = None
        gt_52: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_72, 0.1);  inductor_random_default_72 = None
        mul_157: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_52, div_16)
        mul_158: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_157, 1.1111111111111112);  mul_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_53: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_158, [8, 12, 1024, 1024]);  mul_158 = None
        view_318: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_53, [96, 1024, 1024]);  expand_53 = None
        expand_54: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_138, [8, 12, 1024, 64]);  permute_138 = None
        clone_51: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_54, memory_format = torch.contiguous_format);  expand_54 = None
        view_319: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_51, [96, 1024, 64]);  clone_51 = None
        bmm_25: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_318, view_319)
        view_320: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_25, [8, 12, 1024, 64]);  bmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_141: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_320, [0, 2, 1, 3]);  view_320 = None
        clone_52: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_141, memory_format = torch.contiguous_format);  permute_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_321: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_52, [8, 1024, -1]);  clone_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_142: "f32[768, 768]" = torch.ops.aten.permute.default(primals_107, [1, 0])
        view_322: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_321, [8192, 768]);  view_321 = None
        mm_75: "f32[8192, 768]" = torch.ops.aten.mm.default(view_322, permute_142);  permute_142 = None
        view_323: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_75, [8, 1024, 768]);  mm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_52: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 52)
        inductor_random_default_71: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_52, 'rand');  inductor_lookup_seed_default_52 = None
        gt_53: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_71, 0.1);  inductor_random_default_71 = None
        mul_159: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_53, view_323);  view_323 = None
        mul_160: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_159, 1.1111111111111112);  mul_159 = None
        add_78: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_153, mul_160);  mul_153 = mul_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_27: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_78, 2)
        mean_26: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_27, [-1], True);  pow_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_79: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_26, 1e-06);  mean_26 = None
        rsqrt_26: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
        mul_161: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_78, rsqrt_26)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_162: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_108, mul_161);  mul_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_143: "f32[768, 768]" = torch.ops.aten.permute.default(primals_109, [1, 0])
        view_324: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_162, [8192, 768]);  mul_162 = None
        mm_76: "f32[8192, 768]" = torch.ops.aten.mm.default(view_324, permute_143);  permute_143 = None
        view_325: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_76, [8, 1024, 768]);  mm_76 = None
        view_326: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_325, [8, 1024, -1, 64]);  view_325 = None
        permute_144: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_326, [0, 2, 1, 3]);  view_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_145: "f32[768, 768]" = torch.ops.aten.permute.default(primals_110, [1, 0])
        view_327: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_77: "f32[8192, 768]" = torch.ops.aten.mm.default(view_327, permute_145);  view_327 = permute_145 = None
        view_328: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_77, [8, 1024, 768]);  mm_77 = None
        view_329: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_328, [8, 1024, -1, 64]);  view_328 = None
        permute_146: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_329, [0, 2, 1, 3]);  view_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_147: "f32[768, 768]" = torch.ops.aten.permute.default(primals_111, [1, 0])
        view_330: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_78: "f32[8192, 768]" = torch.ops.aten.mm.default(view_330, permute_147);  view_330 = permute_147 = None
        view_331: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_78, [8, 1024, 768]);  mm_78 = None
        view_332: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_331, [8, 1024, -1, 64]);  view_331 = None
        permute_148: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_332, [0, 2, 1, 3]);  view_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_149: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_146, [0, 1, 3, 2]);  permute_146 = None
        expand_55: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_144, [8, 12, 1024, 64]);  permute_144 = None
        clone_53: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
        view_333: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_53, [96, 1024, 64]);  clone_53 = None
        expand_56: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_149, [8, 12, 64, 1024]);  permute_149 = None
        clone_54: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_334: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_54, [96, 64, 1024]);  clone_54 = None
        bmm_26: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_333, view_334)
        view_335: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_26, [8, 12, 1024, 1024]);  bmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:300 in forward, code: position_bias = torch.zeros(
        full_default_11: "f32[1, 12, 1024, 1024]" = torch.ops.aten.full.default([1, 12, 1024, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        add_80: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(full_default_11, where);  full_default_11 = where = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_81: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_335, add_80);  view_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_13: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_81, [-1], True)
        sub_15: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_81, amax_13);  add_81 = amax_13 = None
        exp_13: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
        sum_14: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_17: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_53: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 53)
        inductor_random_default_70: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_53, 'rand');  inductor_lookup_seed_default_53 = None
        gt_54: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_70, 0.1);  inductor_random_default_70 = None
        mul_163: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_54, div_17)
        mul_164: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_163, 1.1111111111111112);  mul_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_57: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_164, [8, 12, 1024, 1024]);  mul_164 = None
        view_339: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_57, [96, 1024, 1024]);  expand_57 = None
        expand_58: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_148, [8, 12, 1024, 64]);  permute_148 = None
        clone_55: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
        view_340: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_55, [96, 1024, 64]);  clone_55 = None
        bmm_27: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_339, view_340)
        view_341: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_27, [8, 12, 1024, 64]);  bmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_150: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_341, [0, 2, 1, 3]);  view_341 = None
        clone_56: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_342: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_56, [8, 1024, -1]);  clone_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_151: "f32[768, 768]" = torch.ops.aten.permute.default(primals_112, [1, 0])
        view_343: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_342, [8192, 768]);  view_342 = None
        mm_79: "f32[8192, 768]" = torch.ops.aten.mm.default(view_343, permute_151);  permute_151 = None
        view_344: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_79, [8, 1024, 768]);  mm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_54: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 54)
        inductor_random_default_69: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_54, 'rand');  inductor_lookup_seed_default_54 = None
        gt_55: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_69, 0.1);  inductor_random_default_69 = None
        mul_165: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_55, view_344);  view_344 = None
        mul_166: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_165, 1.1111111111111112);  mul_165 = None
        add_82: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_78, mul_166);  mul_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_28: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_82, 2)
        mean_27: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_28, [-1], True);  pow_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_83: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_27, 1e-06);  mean_27 = None
        rsqrt_27: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
        mul_167: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_82, rsqrt_27)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_168: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_113, mul_167);  mul_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_152: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_114, [1, 0])
        view_345: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_168, [8192, 768]);  mul_168 = None
        mm_80: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_345, permute_152);  permute_152 = None
        view_346: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_80, [8, 1024, 3072]);  mm_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_12: "f32[8, 1024, 3072]" = torch.ops.aten.relu.default(view_346);  view_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_55: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 55)
        inductor_random_default_68: "f32[8, 1024, 3072]" = torch.ops.prims.inductor_random.default([8, 1024, 3072], inductor_lookup_seed_default_55, 'rand');  inductor_lookup_seed_default_55 = None
        gt_56: "b8[8, 1024, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_68, 0.1);  inductor_random_default_68 = None
        mul_169: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(gt_56, relu_12)
        mul_170: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_169, 1.1111111111111112);  mul_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_153: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_115, [1, 0])
        view_347: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_170, [8192, 3072]);  mul_170 = None
        mm_81: "f32[8192, 768]" = torch.ops.aten.mm.default(view_347, permute_153);  permute_153 = None
        view_348: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_81, [8, 1024, 768]);  mm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_56: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 56)
        inductor_random_default_67: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_56, 'rand');  inductor_lookup_seed_default_56 = None
        gt_57: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_67, 0.1);  inductor_random_default_67 = None
        mul_171: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_57, view_348);  view_348 = None
        mul_172: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_171, 1.1111111111111112);  mul_171 = None
        add_84: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_82, mul_172);  mul_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_29: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_84, 2)
        mean_28: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_29, [-1], True);  pow_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_85: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_28, 1e-06);  mean_28 = None
        rsqrt_28: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        mul_173: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_84, rsqrt_28)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_174: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_116, mul_173);  mul_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_154: "f32[768, 768]" = torch.ops.aten.permute.default(primals_117, [1, 0])
        view_349: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_174, [8192, 768]);  mul_174 = None
        mm_82: "f32[8192, 768]" = torch.ops.aten.mm.default(view_349, permute_154);  permute_154 = None
        view_350: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_82, [8, 1024, 768]);  mm_82 = None
        view_351: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_350, [8, 1024, -1, 64]);  view_350 = None
        permute_155: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_351, [0, 2, 1, 3]);  view_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_156: "f32[768, 768]" = torch.ops.aten.permute.default(primals_118, [1, 0])
        mm_83: "f32[8192, 768]" = torch.ops.aten.mm.default(view_349, permute_156);  permute_156 = None
        view_353: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_83, [8, 1024, 768]);  mm_83 = None
        view_354: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_353, [8, 1024, -1, 64]);  view_353 = None
        permute_157: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_354, [0, 2, 1, 3]);  view_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_158: "f32[768, 768]" = torch.ops.aten.permute.default(primals_119, [1, 0])
        mm_84: "f32[8192, 768]" = torch.ops.aten.mm.default(view_349, permute_158);  permute_158 = None
        view_356: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_84, [8, 1024, 768]);  mm_84 = None
        view_357: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_356, [8, 1024, -1, 64]);  view_356 = None
        permute_159: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_357, [0, 2, 1, 3]);  view_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_160: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_157, [0, 1, 3, 2]);  permute_157 = None
        expand_59: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_155, [8, 12, 1024, 64]);  permute_155 = None
        clone_57: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
        view_358: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_57, [96, 1024, 64]);  clone_57 = None
        expand_60: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_160, [8, 12, 64, 1024]);  permute_160 = None
        clone_58: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_359: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_58, [96, 64, 1024]);  clone_58 = None
        bmm_28: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_358, view_359)
        view_360: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_28, [8, 12, 1024, 1024]);  bmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_86: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_360, add_76);  view_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_14: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_86, [-1], True)
        sub_16: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_86, amax_14);  add_86 = amax_14 = None
        exp_14: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_15: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_18: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_57: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 57)
        inductor_random_default_66: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_57, 'rand');  inductor_lookup_seed_default_57 = None
        gt_58: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_66, 0.1);  inductor_random_default_66 = None
        mul_175: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_58, div_18)
        mul_176: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_175, 1.1111111111111112);  mul_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_61: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_176, [8, 12, 1024, 1024]);  mul_176 = None
        view_364: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_61, [96, 1024, 1024]);  expand_61 = None
        expand_62: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_159, [8, 12, 1024, 64]);  permute_159 = None
        clone_59: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_62, memory_format = torch.contiguous_format);  expand_62 = None
        view_365: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_59, [96, 1024, 64]);  clone_59 = None
        bmm_29: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_364, view_365)
        view_366: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_29, [8, 12, 1024, 64]);  bmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_161: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_366, [0, 2, 1, 3]);  view_366 = None
        clone_60: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_367: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_60, [8, 1024, -1]);  clone_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_162: "f32[768, 768]" = torch.ops.aten.permute.default(primals_120, [1, 0])
        view_368: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_367, [8192, 768]);  view_367 = None
        mm_85: "f32[8192, 768]" = torch.ops.aten.mm.default(view_368, permute_162);  permute_162 = None
        view_369: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_85, [8, 1024, 768]);  mm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_58: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 58)
        inductor_random_default_65: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_58, 'rand');  inductor_lookup_seed_default_58 = None
        gt_59: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_65, 0.1);  inductor_random_default_65 = None
        mul_177: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_59, view_369);  view_369 = None
        mul_178: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_177, 1.1111111111111112);  mul_177 = None
        add_87: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_84, mul_178);  mul_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_30: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_87, 2)
        mean_29: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_30, [-1], True);  pow_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_88: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_29, 1e-06);  mean_29 = None
        rsqrt_29: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        mul_179: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_87, rsqrt_29)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_180: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_121, mul_179);  mul_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_163: "f32[768, 768]" = torch.ops.aten.permute.default(primals_122, [1, 0])
        view_370: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_180, [8192, 768]);  mul_180 = None
        mm_86: "f32[8192, 768]" = torch.ops.aten.mm.default(view_370, permute_163);  permute_163 = None
        view_371: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_86, [8, 1024, 768]);  mm_86 = None
        view_372: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_371, [8, 1024, -1, 64]);  view_371 = None
        permute_164: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_372, [0, 2, 1, 3]);  view_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_165: "f32[768, 768]" = torch.ops.aten.permute.default(primals_123, [1, 0])
        view_373: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_87: "f32[8192, 768]" = torch.ops.aten.mm.default(view_373, permute_165);  view_373 = permute_165 = None
        view_374: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_87, [8, 1024, 768]);  mm_87 = None
        view_375: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_374, [8, 1024, -1, 64]);  view_374 = None
        permute_166: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_375, [0, 2, 1, 3]);  view_375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_167: "f32[768, 768]" = torch.ops.aten.permute.default(primals_124, [1, 0])
        view_376: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_88: "f32[8192, 768]" = torch.ops.aten.mm.default(view_376, permute_167);  view_376 = permute_167 = None
        view_377: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_88, [8, 1024, 768]);  mm_88 = None
        view_378: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_377, [8, 1024, -1, 64]);  view_377 = None
        permute_168: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_378, [0, 2, 1, 3]);  view_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_169: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_166, [0, 1, 3, 2]);  permute_166 = None
        expand_63: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_164, [8, 12, 1024, 64]);  permute_164 = None
        clone_61: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
        view_379: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_61, [96, 1024, 64]);  clone_61 = None
        expand_64: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_169, [8, 12, 64, 1024]);  permute_169 = None
        clone_62: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
        view_380: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_62, [96, 64, 1024]);  clone_62 = None
        bmm_30: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_379, view_380)
        view_381: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_30, [8, 12, 1024, 1024]);  bmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_89: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_381, add_80);  view_381 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_15: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_89, [-1], True)
        sub_17: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_89, amax_15);  add_89 = amax_15 = None
        exp_15: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
        sum_16: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_19: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_59: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 59)
        inductor_random_default_64: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_59, 'rand');  inductor_lookup_seed_default_59 = None
        gt_60: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_64, 0.1);  inductor_random_default_64 = None
        mul_181: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_60, div_19)
        mul_182: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_181, 1.1111111111111112);  mul_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_65: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_182, [8, 12, 1024, 1024]);  mul_182 = None
        view_385: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_65, [96, 1024, 1024]);  expand_65 = None
        expand_66: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_168, [8, 12, 1024, 64]);  permute_168 = None
        clone_63: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
        view_386: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_63, [96, 1024, 64]);  clone_63 = None
        bmm_31: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_385, view_386)
        view_387: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_31, [8, 12, 1024, 64]);  bmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_170: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_387, [0, 2, 1, 3]);  view_387 = None
        clone_64: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_388: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_64, [8, 1024, -1]);  clone_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_171: "f32[768, 768]" = torch.ops.aten.permute.default(primals_125, [1, 0])
        view_389: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_388, [8192, 768]);  view_388 = None
        mm_89: "f32[8192, 768]" = torch.ops.aten.mm.default(view_389, permute_171);  permute_171 = None
        view_390: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_89, [8, 1024, 768]);  mm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_60: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 60)
        inductor_random_default_63: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_60, 'rand');  inductor_lookup_seed_default_60 = None
        gt_61: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_63, 0.1);  inductor_random_default_63 = None
        mul_183: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_61, view_390);  view_390 = None
        mul_184: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_183, 1.1111111111111112);  mul_183 = None
        add_90: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_87, mul_184);  mul_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_31: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_90, 2)
        mean_30: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_31, [-1], True);  pow_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_91: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_30, 1e-06);  mean_30 = None
        rsqrt_30: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
        mul_185: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_90, rsqrt_30)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_186: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_126, mul_185);  mul_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_172: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_127, [1, 0])
        view_391: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_186, [8192, 768]);  mul_186 = None
        mm_90: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_391, permute_172);  permute_172 = None
        view_392: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_90, [8, 1024, 3072]);  mm_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_13: "f32[8, 1024, 3072]" = torch.ops.aten.relu.default(view_392);  view_392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_61: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 61)
        inductor_random_default_62: "f32[8, 1024, 3072]" = torch.ops.prims.inductor_random.default([8, 1024, 3072], inductor_lookup_seed_default_61, 'rand');  inductor_lookup_seed_default_61 = None
        gt_62: "b8[8, 1024, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_62, 0.1);  inductor_random_default_62 = None
        mul_187: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(gt_62, relu_13)
        mul_188: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_187, 1.1111111111111112);  mul_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_173: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_128, [1, 0])
        view_393: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_188, [8192, 3072]);  mul_188 = None
        mm_91: "f32[8192, 768]" = torch.ops.aten.mm.default(view_393, permute_173);  permute_173 = None
        view_394: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_91, [8, 1024, 768]);  mm_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_62: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 62)
        inductor_random_default_61: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_62, 'rand');  inductor_lookup_seed_default_62 = None
        gt_63: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_61, 0.1);  inductor_random_default_61 = None
        mul_189: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_63, view_394);  view_394 = None
        mul_190: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_189, 1.1111111111111112);  mul_189 = None
        add_92: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_90, mul_190);  mul_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_32: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_92, 2)
        mean_31: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_32, [-1], True);  pow_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_93: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_31, 1e-06);  mean_31 = None
        rsqrt_31: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        mul_191: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_92, rsqrt_31)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_192: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_129, mul_191);  mul_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_174: "f32[768, 768]" = torch.ops.aten.permute.default(primals_130, [1, 0])
        view_395: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_192, [8192, 768]);  mul_192 = None
        mm_92: "f32[8192, 768]" = torch.ops.aten.mm.default(view_395, permute_174);  permute_174 = None
        view_396: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_92, [8, 1024, 768]);  mm_92 = None
        view_397: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_396, [8, 1024, -1, 64]);  view_396 = None
        permute_175: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_397, [0, 2, 1, 3]);  view_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_176: "f32[768, 768]" = torch.ops.aten.permute.default(primals_131, [1, 0])
        mm_93: "f32[8192, 768]" = torch.ops.aten.mm.default(view_395, permute_176);  permute_176 = None
        view_399: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_93, [8, 1024, 768]);  mm_93 = None
        view_400: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_399, [8, 1024, -1, 64]);  view_399 = None
        permute_177: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_400, [0, 2, 1, 3]);  view_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_178: "f32[768, 768]" = torch.ops.aten.permute.default(primals_132, [1, 0])
        mm_94: "f32[8192, 768]" = torch.ops.aten.mm.default(view_395, permute_178);  permute_178 = None
        view_402: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_94, [8, 1024, 768]);  mm_94 = None
        view_403: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_402, [8, 1024, -1, 64]);  view_402 = None
        permute_179: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_403, [0, 2, 1, 3]);  view_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_180: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_177, [0, 1, 3, 2]);  permute_177 = None
        expand_67: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_175, [8, 12, 1024, 64]);  permute_175 = None
        clone_65: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
        view_404: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_65, [96, 1024, 64]);  clone_65 = None
        expand_68: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_180, [8, 12, 64, 1024]);  permute_180 = None
        clone_66: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
        view_405: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_66, [96, 64, 1024]);  clone_66 = None
        bmm_32: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_404, view_405)
        view_406: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_32, [8, 12, 1024, 1024]);  bmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_94: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_406, add_76);  view_406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_16: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_94, [-1], True)
        sub_18: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_94, amax_16);  add_94 = amax_16 = None
        exp_16: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
        sum_17: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_20: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_63: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 63)
        inductor_random_default_60: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_63, 'rand');  inductor_lookup_seed_default_63 = None
        gt_64: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_60, 0.1);  inductor_random_default_60 = None
        mul_193: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_64, div_20)
        mul_194: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_193, 1.1111111111111112);  mul_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_69: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_194, [8, 12, 1024, 1024]);  mul_194 = None
        view_410: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_69, [96, 1024, 1024]);  expand_69 = None
        expand_70: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_179, [8, 12, 1024, 64]);  permute_179 = None
        clone_67: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_70, memory_format = torch.contiguous_format);  expand_70 = None
        view_411: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_67, [96, 1024, 64]);  clone_67 = None
        bmm_33: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_410, view_411)
        view_412: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_33, [8, 12, 1024, 64]);  bmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_181: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_412, [0, 2, 1, 3]);  view_412 = None
        clone_68: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_181, memory_format = torch.contiguous_format);  permute_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_413: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_68, [8, 1024, -1]);  clone_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_182: "f32[768, 768]" = torch.ops.aten.permute.default(primals_133, [1, 0])
        view_414: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_413, [8192, 768]);  view_413 = None
        mm_95: "f32[8192, 768]" = torch.ops.aten.mm.default(view_414, permute_182);  permute_182 = None
        view_415: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_95, [8, 1024, 768]);  mm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_64: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 64)
        inductor_random_default_59: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_64, 'rand');  inductor_lookup_seed_default_64 = None
        gt_65: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_59, 0.1);  inductor_random_default_59 = None
        mul_195: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_65, view_415);  view_415 = None
        mul_196: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_195, 1.1111111111111112);  mul_195 = None
        add_95: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_92, mul_196);  mul_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_33: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_95, 2)
        mean_32: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_33, [-1], True);  pow_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_96: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_32, 1e-06);  mean_32 = None
        rsqrt_32: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        mul_197: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_95, rsqrt_32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_198: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_134, mul_197);  mul_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_183: "f32[768, 768]" = torch.ops.aten.permute.default(primals_135, [1, 0])
        view_416: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_198, [8192, 768]);  mul_198 = None
        mm_96: "f32[8192, 768]" = torch.ops.aten.mm.default(view_416, permute_183);  permute_183 = None
        view_417: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_96, [8, 1024, 768]);  mm_96 = None
        view_418: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_417, [8, 1024, -1, 64]);  view_417 = None
        permute_184: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_418, [0, 2, 1, 3]);  view_418 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_185: "f32[768, 768]" = torch.ops.aten.permute.default(primals_136, [1, 0])
        view_419: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_97: "f32[8192, 768]" = torch.ops.aten.mm.default(view_419, permute_185);  view_419 = permute_185 = None
        view_420: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_97, [8, 1024, 768]);  mm_97 = None
        view_421: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_420, [8, 1024, -1, 64]);  view_420 = None
        permute_186: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_421, [0, 2, 1, 3]);  view_421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_187: "f32[768, 768]" = torch.ops.aten.permute.default(primals_137, [1, 0])
        view_422: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_98: "f32[8192, 768]" = torch.ops.aten.mm.default(view_422, permute_187);  view_422 = permute_187 = None
        view_423: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_98, [8, 1024, 768]);  mm_98 = None
        view_424: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_423, [8, 1024, -1, 64]);  view_423 = None
        permute_188: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_424, [0, 2, 1, 3]);  view_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_189: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_186, [0, 1, 3, 2]);  permute_186 = None
        expand_71: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_184, [8, 12, 1024, 64]);  permute_184 = None
        clone_69: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
        view_425: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_69, [96, 1024, 64]);  clone_69 = None
        expand_72: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_189, [8, 12, 64, 1024]);  permute_189 = None
        clone_70: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
        view_426: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_70, [96, 64, 1024]);  clone_70 = None
        bmm_34: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_425, view_426)
        view_427: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_34, [8, 12, 1024, 1024]);  bmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_97: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_427, add_80);  view_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_17: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_97, [-1], True)
        sub_19: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_97, amax_17);  add_97 = amax_17 = None
        exp_17: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_18: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_21: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_65: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 65)
        inductor_random_default_58: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_65, 'rand');  inductor_lookup_seed_default_65 = None
        gt_66: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_58, 0.1);  inductor_random_default_58 = None
        mul_199: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_66, div_21)
        mul_200: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_199, 1.1111111111111112);  mul_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_73: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_200, [8, 12, 1024, 1024]);  mul_200 = None
        view_431: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_73, [96, 1024, 1024]);  expand_73 = None
        expand_74: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_188, [8, 12, 1024, 64]);  permute_188 = None
        clone_71: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
        view_432: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_71, [96, 1024, 64]);  clone_71 = None
        bmm_35: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_431, view_432)
        view_433: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_35, [8, 12, 1024, 64]);  bmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_190: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_433, [0, 2, 1, 3]);  view_433 = None
        clone_72: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_190, memory_format = torch.contiguous_format);  permute_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_434: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_72, [8, 1024, -1]);  clone_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_191: "f32[768, 768]" = torch.ops.aten.permute.default(primals_138, [1, 0])
        view_435: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_434, [8192, 768]);  view_434 = None
        mm_99: "f32[8192, 768]" = torch.ops.aten.mm.default(view_435, permute_191);  permute_191 = None
        view_436: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_99, [8, 1024, 768]);  mm_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_66: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 66)
        inductor_random_default_57: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_66, 'rand');  inductor_lookup_seed_default_66 = None
        gt_67: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_57, 0.1);  inductor_random_default_57 = None
        mul_201: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_67, view_436);  view_436 = None
        mul_202: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_201, 1.1111111111111112);  mul_201 = None
        add_98: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_95, mul_202);  mul_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_34: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_98, 2)
        mean_33: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_34, [-1], True);  pow_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_99: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_33, 1e-06);  mean_33 = None
        rsqrt_33: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
        mul_203: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_98, rsqrt_33)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_204: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_139, mul_203);  mul_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_192: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_140, [1, 0])
        view_437: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_204, [8192, 768]);  mul_204 = None
        mm_100: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_437, permute_192);  permute_192 = None
        view_438: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_100, [8, 1024, 3072]);  mm_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_14: "f32[8, 1024, 3072]" = torch.ops.aten.relu.default(view_438);  view_438 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_67: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 67)
        inductor_random_default_56: "f32[8, 1024, 3072]" = torch.ops.prims.inductor_random.default([8, 1024, 3072], inductor_lookup_seed_default_67, 'rand');  inductor_lookup_seed_default_67 = None
        gt_68: "b8[8, 1024, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_56, 0.1);  inductor_random_default_56 = None
        mul_205: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(gt_68, relu_14)
        mul_206: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_205, 1.1111111111111112);  mul_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_193: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_141, [1, 0])
        view_439: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_206, [8192, 3072]);  mul_206 = None
        mm_101: "f32[8192, 768]" = torch.ops.aten.mm.default(view_439, permute_193);  permute_193 = None
        view_440: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_101, [8, 1024, 768]);  mm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_68: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 68)
        inductor_random_default_55: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_68, 'rand');  inductor_lookup_seed_default_68 = None
        gt_69: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_55, 0.1);  inductor_random_default_55 = None
        mul_207: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_69, view_440);  view_440 = None
        mul_208: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_207, 1.1111111111111112);  mul_207 = None
        add_100: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_98, mul_208);  mul_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_35: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_100, 2)
        mean_34: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_35, [-1], True);  pow_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_101: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_34, 1e-06);  mean_34 = None
        rsqrt_34: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
        mul_209: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_100, rsqrt_34)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_210: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_142, mul_209);  mul_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_194: "f32[768, 768]" = torch.ops.aten.permute.default(primals_143, [1, 0])
        view_441: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_210, [8192, 768]);  mul_210 = None
        mm_102: "f32[8192, 768]" = torch.ops.aten.mm.default(view_441, permute_194);  permute_194 = None
        view_442: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_102, [8, 1024, 768]);  mm_102 = None
        view_443: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_442, [8, 1024, -1, 64]);  view_442 = None
        permute_195: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_443, [0, 2, 1, 3]);  view_443 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_196: "f32[768, 768]" = torch.ops.aten.permute.default(primals_144, [1, 0])
        mm_103: "f32[8192, 768]" = torch.ops.aten.mm.default(view_441, permute_196);  permute_196 = None
        view_445: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_103, [8, 1024, 768]);  mm_103 = None
        view_446: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_445, [8, 1024, -1, 64]);  view_445 = None
        permute_197: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_446, [0, 2, 1, 3]);  view_446 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_198: "f32[768, 768]" = torch.ops.aten.permute.default(primals_145, [1, 0])
        mm_104: "f32[8192, 768]" = torch.ops.aten.mm.default(view_441, permute_198);  permute_198 = None
        view_448: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_104, [8, 1024, 768]);  mm_104 = None
        view_449: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_448, [8, 1024, -1, 64]);  view_448 = None
        permute_199: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_449, [0, 2, 1, 3]);  view_449 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_200: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_197, [0, 1, 3, 2]);  permute_197 = None
        expand_75: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_195, [8, 12, 1024, 64]);  permute_195 = None
        clone_73: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
        view_450: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_73, [96, 1024, 64]);  clone_73 = None
        expand_76: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_200, [8, 12, 64, 1024]);  permute_200 = None
        clone_74: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
        view_451: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_74, [96, 64, 1024]);  clone_74 = None
        bmm_36: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_450, view_451)
        view_452: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_36, [8, 12, 1024, 1024]);  bmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_102: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_452, add_76);  view_452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_18: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_102, [-1], True)
        sub_20: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_102, amax_18);  add_102 = amax_18 = None
        exp_18: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
        sum_19: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_22: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_69: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 69)
        inductor_random_default_54: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_69, 'rand');  inductor_lookup_seed_default_69 = None
        gt_70: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_54, 0.1);  inductor_random_default_54 = None
        mul_211: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_70, div_22)
        mul_212: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_211, 1.1111111111111112);  mul_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_77: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_212, [8, 12, 1024, 1024]);  mul_212 = None
        view_456: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_77, [96, 1024, 1024]);  expand_77 = None
        expand_78: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_199, [8, 12, 1024, 64]);  permute_199 = None
        clone_75: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_78, memory_format = torch.contiguous_format);  expand_78 = None
        view_457: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_75, [96, 1024, 64]);  clone_75 = None
        bmm_37: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_456, view_457)
        view_458: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_37, [8, 12, 1024, 64]);  bmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_201: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_458, [0, 2, 1, 3]);  view_458 = None
        clone_76: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_201, memory_format = torch.contiguous_format);  permute_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_459: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_76, [8, 1024, -1]);  clone_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_202: "f32[768, 768]" = torch.ops.aten.permute.default(primals_146, [1, 0])
        view_460: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_459, [8192, 768]);  view_459 = None
        mm_105: "f32[8192, 768]" = torch.ops.aten.mm.default(view_460, permute_202);  permute_202 = None
        view_461: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_105, [8, 1024, 768]);  mm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_70: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 70)
        inductor_random_default_53: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_70, 'rand');  inductor_lookup_seed_default_70 = None
        gt_71: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_53, 0.1);  inductor_random_default_53 = None
        mul_213: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_71, view_461);  view_461 = None
        mul_214: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_213, 1.1111111111111112);  mul_213 = None
        add_103: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_100, mul_214);  mul_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_36: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_103, 2)
        mean_35: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_36, [-1], True);  pow_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_104: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_35, 1e-06);  mean_35 = None
        rsqrt_35: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        mul_215: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_103, rsqrt_35)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_216: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_147, mul_215);  mul_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_203: "f32[768, 768]" = torch.ops.aten.permute.default(primals_148, [1, 0])
        view_462: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_216, [8192, 768]);  mul_216 = None
        mm_106: "f32[8192, 768]" = torch.ops.aten.mm.default(view_462, permute_203);  permute_203 = None
        view_463: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_106, [8, 1024, 768]);  mm_106 = None
        view_464: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_463, [8, 1024, -1, 64]);  view_463 = None
        permute_204: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_464, [0, 2, 1, 3]);  view_464 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_205: "f32[768, 768]" = torch.ops.aten.permute.default(primals_149, [1, 0])
        view_465: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_107: "f32[8192, 768]" = torch.ops.aten.mm.default(view_465, permute_205);  view_465 = permute_205 = None
        view_466: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_107, [8, 1024, 768]);  mm_107 = None
        view_467: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_466, [8, 1024, -1, 64]);  view_466 = None
        permute_206: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_467, [0, 2, 1, 3]);  view_467 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_207: "f32[768, 768]" = torch.ops.aten.permute.default(primals_150, [1, 0])
        view_468: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_108: "f32[8192, 768]" = torch.ops.aten.mm.default(view_468, permute_207);  view_468 = permute_207 = None
        view_469: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_108, [8, 1024, 768]);  mm_108 = None
        view_470: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_469, [8, 1024, -1, 64]);  view_469 = None
        permute_208: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_470, [0, 2, 1, 3]);  view_470 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_209: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_206, [0, 1, 3, 2]);  permute_206 = None
        expand_79: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_204, [8, 12, 1024, 64]);  permute_204 = None
        clone_77: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_79, memory_format = torch.contiguous_format);  expand_79 = None
        view_471: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_77, [96, 1024, 64]);  clone_77 = None
        expand_80: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_209, [8, 12, 64, 1024]);  permute_209 = None
        clone_78: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
        view_472: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_78, [96, 64, 1024]);  clone_78 = None
        bmm_38: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_471, view_472)
        view_473: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_38, [8, 12, 1024, 1024]);  bmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_105: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_473, add_80);  view_473 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_19: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_105, [-1], True)
        sub_21: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_105, amax_19);  add_105 = amax_19 = None
        exp_19: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
        sum_20: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_23: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_71: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 71)
        inductor_random_default_52: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_71, 'rand');  inductor_lookup_seed_default_71 = None
        gt_72: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_52, 0.1);  inductor_random_default_52 = None
        mul_217: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_72, div_23)
        mul_218: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_217, 1.1111111111111112);  mul_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_81: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_218, [8, 12, 1024, 1024]);  mul_218 = None
        view_477: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_81, [96, 1024, 1024]);  expand_81 = None
        expand_82: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_208, [8, 12, 1024, 64]);  permute_208 = None
        clone_79: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_82, memory_format = torch.contiguous_format);  expand_82 = None
        view_478: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_79, [96, 1024, 64]);  clone_79 = None
        bmm_39: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_477, view_478)
        view_479: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_39, [8, 12, 1024, 64]);  bmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_210: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_479, [0, 2, 1, 3]);  view_479 = None
        clone_80: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_480: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_80, [8, 1024, -1]);  clone_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_211: "f32[768, 768]" = torch.ops.aten.permute.default(primals_151, [1, 0])
        view_481: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_480, [8192, 768]);  view_480 = None
        mm_109: "f32[8192, 768]" = torch.ops.aten.mm.default(view_481, permute_211);  permute_211 = None
        view_482: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_109, [8, 1024, 768]);  mm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_72: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 72)
        inductor_random_default_51: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_72, 'rand');  inductor_lookup_seed_default_72 = None
        gt_73: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_51, 0.1);  inductor_random_default_51 = None
        mul_219: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_73, view_482);  view_482 = None
        mul_220: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_219, 1.1111111111111112);  mul_219 = None
        add_106: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_103, mul_220);  mul_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_37: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_106, 2)
        mean_36: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_37, [-1], True);  pow_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_107: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_36, 1e-06);  mean_36 = None
        rsqrt_36: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_107);  add_107 = None
        mul_221: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_106, rsqrt_36)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_222: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_152, mul_221);  mul_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_212: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_153, [1, 0])
        view_483: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_222, [8192, 768]);  mul_222 = None
        mm_110: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_483, permute_212);  permute_212 = None
        view_484: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_110, [8, 1024, 3072]);  mm_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_15: "f32[8, 1024, 3072]" = torch.ops.aten.relu.default(view_484);  view_484 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_73: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 73)
        inductor_random_default_50: "f32[8, 1024, 3072]" = torch.ops.prims.inductor_random.default([8, 1024, 3072], inductor_lookup_seed_default_73, 'rand');  inductor_lookup_seed_default_73 = None
        gt_74: "b8[8, 1024, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_50, 0.1);  inductor_random_default_50 = None
        mul_223: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(gt_74, relu_15)
        mul_224: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_223, 1.1111111111111112);  mul_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_213: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_154, [1, 0])
        view_485: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_224, [8192, 3072]);  mul_224 = None
        mm_111: "f32[8192, 768]" = torch.ops.aten.mm.default(view_485, permute_213);  permute_213 = None
        view_486: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_111, [8, 1024, 768]);  mm_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_74: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 74)
        inductor_random_default_49: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_74, 'rand');  inductor_lookup_seed_default_74 = None
        gt_75: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_49, 0.1);  inductor_random_default_49 = None
        mul_225: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_75, view_486);  view_486 = None
        mul_226: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_225, 1.1111111111111112);  mul_225 = None
        add_108: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_106, mul_226);  mul_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_38: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_108, 2)
        mean_37: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_38, [-1], True);  pow_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_109: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_37, 1e-06);  mean_37 = None
        rsqrt_37: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
        mul_227: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_108, rsqrt_37)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_228: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_155, mul_227);  mul_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_214: "f32[768, 768]" = torch.ops.aten.permute.default(primals_156, [1, 0])
        view_487: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_228, [8192, 768]);  mul_228 = None
        mm_112: "f32[8192, 768]" = torch.ops.aten.mm.default(view_487, permute_214);  permute_214 = None
        view_488: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_112, [8, 1024, 768]);  mm_112 = None
        view_489: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_488, [8, 1024, -1, 64]);  view_488 = None
        permute_215: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_489, [0, 2, 1, 3]);  view_489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_216: "f32[768, 768]" = torch.ops.aten.permute.default(primals_157, [1, 0])
        mm_113: "f32[8192, 768]" = torch.ops.aten.mm.default(view_487, permute_216);  permute_216 = None
        view_491: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_113, [8, 1024, 768]);  mm_113 = None
        view_492: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_491, [8, 1024, -1, 64]);  view_491 = None
        permute_217: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_492, [0, 2, 1, 3]);  view_492 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_218: "f32[768, 768]" = torch.ops.aten.permute.default(primals_158, [1, 0])
        mm_114: "f32[8192, 768]" = torch.ops.aten.mm.default(view_487, permute_218);  permute_218 = None
        view_494: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_114, [8, 1024, 768]);  mm_114 = None
        view_495: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_494, [8, 1024, -1, 64]);  view_494 = None
        permute_219: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_495, [0, 2, 1, 3]);  view_495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_220: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_217, [0, 1, 3, 2]);  permute_217 = None
        expand_83: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_215, [8, 12, 1024, 64]);  permute_215 = None
        clone_81: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_83, memory_format = torch.contiguous_format);  expand_83 = None
        view_496: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_81, [96, 1024, 64]);  clone_81 = None
        expand_84: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_220, [8, 12, 64, 1024]);  permute_220 = None
        clone_82: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
        view_497: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_82, [96, 64, 1024]);  clone_82 = None
        bmm_40: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_496, view_497)
        view_498: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_40, [8, 12, 1024, 1024]);  bmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_110: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_498, add_76);  view_498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_20: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_110, [-1], True)
        sub_22: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_110, amax_20);  add_110 = amax_20 = None
        exp_20: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_21: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_24: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_75: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 75)
        inductor_random_default_48: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_75, 'rand');  inductor_lookup_seed_default_75 = None
        gt_76: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_48, 0.1);  inductor_random_default_48 = None
        mul_229: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_76, div_24)
        mul_230: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_229, 1.1111111111111112);  mul_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_85: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_230, [8, 12, 1024, 1024]);  mul_230 = None
        view_502: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_85, [96, 1024, 1024]);  expand_85 = None
        expand_86: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_219, [8, 12, 1024, 64]);  permute_219 = None
        clone_83: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_86, memory_format = torch.contiguous_format);  expand_86 = None
        view_503: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_83, [96, 1024, 64]);  clone_83 = None
        bmm_41: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_502, view_503)
        view_504: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_41, [8, 12, 1024, 64]);  bmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_221: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_504, [0, 2, 1, 3]);  view_504 = None
        clone_84: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_221, memory_format = torch.contiguous_format);  permute_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_505: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_84, [8, 1024, -1]);  clone_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_222: "f32[768, 768]" = torch.ops.aten.permute.default(primals_159, [1, 0])
        view_506: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_505, [8192, 768]);  view_505 = None
        mm_115: "f32[8192, 768]" = torch.ops.aten.mm.default(view_506, permute_222);  permute_222 = None
        view_507: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_115, [8, 1024, 768]);  mm_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_76: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 76)
        inductor_random_default_47: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_76, 'rand');  inductor_lookup_seed_default_76 = None
        gt_77: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_47, 0.1);  inductor_random_default_47 = None
        mul_231: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_77, view_507);  view_507 = None
        mul_232: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_231, 1.1111111111111112);  mul_231 = None
        add_111: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_108, mul_232);  mul_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_39: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_111, 2)
        mean_38: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_39, [-1], True);  pow_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_112: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_38, 1e-06);  mean_38 = None
        rsqrt_38: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
        mul_233: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_111, rsqrt_38)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_234: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_160, mul_233);  mul_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_223: "f32[768, 768]" = torch.ops.aten.permute.default(primals_161, [1, 0])
        view_508: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_234, [8192, 768]);  mul_234 = None
        mm_116: "f32[8192, 768]" = torch.ops.aten.mm.default(view_508, permute_223);  permute_223 = None
        view_509: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_116, [8, 1024, 768]);  mm_116 = None
        view_510: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_509, [8, 1024, -1, 64]);  view_509 = None
        permute_224: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_510, [0, 2, 1, 3]);  view_510 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_225: "f32[768, 768]" = torch.ops.aten.permute.default(primals_162, [1, 0])
        view_511: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_117: "f32[8192, 768]" = torch.ops.aten.mm.default(view_511, permute_225);  view_511 = permute_225 = None
        view_512: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_117, [8, 1024, 768]);  mm_117 = None
        view_513: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_512, [8, 1024, -1, 64]);  view_512 = None
        permute_226: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_513, [0, 2, 1, 3]);  view_513 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_227: "f32[768, 768]" = torch.ops.aten.permute.default(primals_163, [1, 0])
        view_514: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_118: "f32[8192, 768]" = torch.ops.aten.mm.default(view_514, permute_227);  view_514 = permute_227 = None
        view_515: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_118, [8, 1024, 768]);  mm_118 = None
        view_516: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_515, [8, 1024, -1, 64]);  view_515 = None
        permute_228: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_516, [0, 2, 1, 3]);  view_516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_229: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_226, [0, 1, 3, 2]);  permute_226 = None
        expand_87: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_224, [8, 12, 1024, 64]);  permute_224 = None
        clone_85: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_87, memory_format = torch.contiguous_format);  expand_87 = None
        view_517: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_85, [96, 1024, 64]);  clone_85 = None
        expand_88: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_229, [8, 12, 64, 1024]);  permute_229 = None
        clone_86: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
        view_518: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_86, [96, 64, 1024]);  clone_86 = None
        bmm_42: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_517, view_518)
        view_519: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_42, [8, 12, 1024, 1024]);  bmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_113: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_519, add_80);  view_519 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_21: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_113, [-1], True)
        sub_23: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_113, amax_21);  add_113 = amax_21 = None
        exp_21: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_23);  sub_23 = None
        sum_22: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_25: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_77: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 77)
        inductor_random_default_46: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_77, 'rand');  inductor_lookup_seed_default_77 = None
        gt_78: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_46, 0.1);  inductor_random_default_46 = None
        mul_235: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_78, div_25)
        mul_236: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_235, 1.1111111111111112);  mul_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_89: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_236, [8, 12, 1024, 1024]);  mul_236 = None
        view_523: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_89, [96, 1024, 1024]);  expand_89 = None
        expand_90: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_228, [8, 12, 1024, 64]);  permute_228 = None
        clone_87: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_90, memory_format = torch.contiguous_format);  expand_90 = None
        view_524: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_87, [96, 1024, 64]);  clone_87 = None
        bmm_43: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_523, view_524)
        view_525: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_43, [8, 12, 1024, 64]);  bmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_230: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_525, [0, 2, 1, 3]);  view_525 = None
        clone_88: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_230, memory_format = torch.contiguous_format);  permute_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_526: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_88, [8, 1024, -1]);  clone_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_231: "f32[768, 768]" = torch.ops.aten.permute.default(primals_164, [1, 0])
        view_527: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_526, [8192, 768]);  view_526 = None
        mm_119: "f32[8192, 768]" = torch.ops.aten.mm.default(view_527, permute_231);  permute_231 = None
        view_528: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_119, [8, 1024, 768]);  mm_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_78: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 78)
        inductor_random_default_45: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_78, 'rand');  inductor_lookup_seed_default_78 = None
        gt_79: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_45, 0.1);  inductor_random_default_45 = None
        mul_237: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_79, view_528);  view_528 = None
        mul_238: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_237, 1.1111111111111112);  mul_237 = None
        add_114: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_111, mul_238);  mul_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_40: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_114, 2)
        mean_39: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_40, [-1], True);  pow_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_115: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_39, 1e-06);  mean_39 = None
        rsqrt_39: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
        mul_239: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_114, rsqrt_39)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_240: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_165, mul_239);  mul_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_232: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_166, [1, 0])
        view_529: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_240, [8192, 768]);  mul_240 = None
        mm_120: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_529, permute_232);  permute_232 = None
        view_530: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_120, [8, 1024, 3072]);  mm_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_16: "f32[8, 1024, 3072]" = torch.ops.aten.relu.default(view_530);  view_530 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_79: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 79)
        inductor_random_default_44: "f32[8, 1024, 3072]" = torch.ops.prims.inductor_random.default([8, 1024, 3072], inductor_lookup_seed_default_79, 'rand');  inductor_lookup_seed_default_79 = None
        gt_80: "b8[8, 1024, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_44, 0.1);  inductor_random_default_44 = None
        mul_241: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(gt_80, relu_16)
        mul_242: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_241, 1.1111111111111112);  mul_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_233: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_167, [1, 0])
        view_531: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_242, [8192, 3072]);  mul_242 = None
        mm_121: "f32[8192, 768]" = torch.ops.aten.mm.default(view_531, permute_233);  permute_233 = None
        view_532: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_121, [8, 1024, 768]);  mm_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_80: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 80)
        inductor_random_default_43: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_80, 'rand');  inductor_lookup_seed_default_80 = None
        gt_81: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_43, 0.1);  inductor_random_default_43 = None
        mul_243: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_81, view_532);  view_532 = None
        mul_244: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_243, 1.1111111111111112);  mul_243 = None
        add_116: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_114, mul_244);  mul_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_41: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_116, 2)
        mean_40: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_41, [-1], True);  pow_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_117: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_40, 1e-06);  mean_40 = None
        rsqrt_40: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
        mul_245: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_116, rsqrt_40)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_246: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_168, mul_245);  mul_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_234: "f32[768, 768]" = torch.ops.aten.permute.default(primals_169, [1, 0])
        view_533: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_246, [8192, 768]);  mul_246 = None
        mm_122: "f32[8192, 768]" = torch.ops.aten.mm.default(view_533, permute_234);  permute_234 = None
        view_534: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_122, [8, 1024, 768]);  mm_122 = None
        view_535: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_534, [8, 1024, -1, 64]);  view_534 = None
        permute_235: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_535, [0, 2, 1, 3]);  view_535 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_236: "f32[768, 768]" = torch.ops.aten.permute.default(primals_170, [1, 0])
        mm_123: "f32[8192, 768]" = torch.ops.aten.mm.default(view_533, permute_236);  permute_236 = None
        view_537: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_123, [8, 1024, 768]);  mm_123 = None
        view_538: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_537, [8, 1024, -1, 64]);  view_537 = None
        permute_237: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_538, [0, 2, 1, 3]);  view_538 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_238: "f32[768, 768]" = torch.ops.aten.permute.default(primals_171, [1, 0])
        mm_124: "f32[8192, 768]" = torch.ops.aten.mm.default(view_533, permute_238);  permute_238 = None
        view_540: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_124, [8, 1024, 768]);  mm_124 = None
        view_541: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_540, [8, 1024, -1, 64]);  view_540 = None
        permute_239: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_541, [0, 2, 1, 3]);  view_541 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_240: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_237, [0, 1, 3, 2]);  permute_237 = None
        expand_91: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_235, [8, 12, 1024, 64]);  permute_235 = None
        clone_89: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_91, memory_format = torch.contiguous_format);  expand_91 = None
        view_542: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_89, [96, 1024, 64]);  clone_89 = None
        expand_92: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_240, [8, 12, 64, 1024]);  permute_240 = None
        clone_90: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
        view_543: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_90, [96, 64, 1024]);  clone_90 = None
        bmm_44: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_542, view_543)
        view_544: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_44, [8, 12, 1024, 1024]);  bmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_118: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_544, add_76);  view_544 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_22: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_118, [-1], True)
        sub_24: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_118, amax_22);  add_118 = amax_22 = None
        exp_22: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_24);  sub_24 = None
        sum_23: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_26: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_81: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 81)
        inductor_random_default_42: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_81, 'rand');  inductor_lookup_seed_default_81 = None
        gt_82: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_42, 0.1);  inductor_random_default_42 = None
        mul_247: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_82, div_26)
        mul_248: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_247, 1.1111111111111112);  mul_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_93: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_248, [8, 12, 1024, 1024]);  mul_248 = None
        view_548: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_93, [96, 1024, 1024]);  expand_93 = None
        expand_94: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_239, [8, 12, 1024, 64]);  permute_239 = None
        clone_91: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_94, memory_format = torch.contiguous_format);  expand_94 = None
        view_549: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_91, [96, 1024, 64]);  clone_91 = None
        bmm_45: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_548, view_549)
        view_550: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_45, [8, 12, 1024, 64]);  bmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_241: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_550, [0, 2, 1, 3]);  view_550 = None
        clone_92: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_241, memory_format = torch.contiguous_format);  permute_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_551: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_92, [8, 1024, -1]);  clone_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_242: "f32[768, 768]" = torch.ops.aten.permute.default(primals_172, [1, 0])
        view_552: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_551, [8192, 768]);  view_551 = None
        mm_125: "f32[8192, 768]" = torch.ops.aten.mm.default(view_552, permute_242);  permute_242 = None
        view_553: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_125, [8, 1024, 768]);  mm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_82: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 82)
        inductor_random_default_41: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_82, 'rand');  inductor_lookup_seed_default_82 = None
        gt_83: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_41, 0.1);  inductor_random_default_41 = None
        mul_249: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_83, view_553);  view_553 = None
        mul_250: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_249, 1.1111111111111112);  mul_249 = None
        add_119: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_116, mul_250);  mul_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_42: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_119, 2)
        mean_41: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_42, [-1], True);  pow_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_120: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_41, 1e-06);  mean_41 = None
        rsqrt_41: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
        mul_251: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_119, rsqrt_41)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_252: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_173, mul_251);  mul_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_243: "f32[768, 768]" = torch.ops.aten.permute.default(primals_174, [1, 0])
        view_554: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_252, [8192, 768]);  mul_252 = None
        mm_126: "f32[8192, 768]" = torch.ops.aten.mm.default(view_554, permute_243);  permute_243 = None
        view_555: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_126, [8, 1024, 768]);  mm_126 = None
        view_556: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_555, [8, 1024, -1, 64]);  view_555 = None
        permute_244: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_556, [0, 2, 1, 3]);  view_556 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_245: "f32[768, 768]" = torch.ops.aten.permute.default(primals_175, [1, 0])
        view_557: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_127: "f32[8192, 768]" = torch.ops.aten.mm.default(view_557, permute_245);  view_557 = permute_245 = None
        view_558: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_127, [8, 1024, 768]);  mm_127 = None
        view_559: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_558, [8, 1024, -1, 64]);  view_558 = None
        permute_246: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_559, [0, 2, 1, 3]);  view_559 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_247: "f32[768, 768]" = torch.ops.aten.permute.default(primals_176, [1, 0])
        view_560: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_128: "f32[8192, 768]" = torch.ops.aten.mm.default(view_560, permute_247);  view_560 = permute_247 = None
        view_561: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_128, [8, 1024, 768]);  mm_128 = None
        view_562: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_561, [8, 1024, -1, 64]);  view_561 = None
        permute_248: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_562, [0, 2, 1, 3]);  view_562 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_249: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_246, [0, 1, 3, 2]);  permute_246 = None
        expand_95: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_244, [8, 12, 1024, 64]);  permute_244 = None
        clone_93: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_95, memory_format = torch.contiguous_format);  expand_95 = None
        view_563: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_93, [96, 1024, 64]);  clone_93 = None
        expand_96: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_249, [8, 12, 64, 1024]);  permute_249 = None
        clone_94: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_96, memory_format = torch.contiguous_format);  expand_96 = None
        view_564: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_94, [96, 64, 1024]);  clone_94 = None
        bmm_46: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_563, view_564)
        view_565: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_46, [8, 12, 1024, 1024]);  bmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_121: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_565, add_80);  view_565 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_23: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_121, [-1], True)
        sub_25: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_121, amax_23);  add_121 = amax_23 = None
        exp_23: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_24: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_27: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_83: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 83)
        inductor_random_default_40: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_83, 'rand');  inductor_lookup_seed_default_83 = None
        gt_84: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_40, 0.1);  inductor_random_default_40 = None
        mul_253: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_84, div_27)
        mul_254: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_253, 1.1111111111111112);  mul_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_97: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_254, [8, 12, 1024, 1024]);  mul_254 = None
        view_569: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_97, [96, 1024, 1024]);  expand_97 = None
        expand_98: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_248, [8, 12, 1024, 64]);  permute_248 = None
        clone_95: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_98, memory_format = torch.contiguous_format);  expand_98 = None
        view_570: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_95, [96, 1024, 64]);  clone_95 = None
        bmm_47: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_569, view_570)
        view_571: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_47, [8, 12, 1024, 64]);  bmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_250: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_571, [0, 2, 1, 3]);  view_571 = None
        clone_96: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_250, memory_format = torch.contiguous_format);  permute_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_572: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_96, [8, 1024, -1]);  clone_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_251: "f32[768, 768]" = torch.ops.aten.permute.default(primals_177, [1, 0])
        view_573: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_572, [8192, 768]);  view_572 = None
        mm_129: "f32[8192, 768]" = torch.ops.aten.mm.default(view_573, permute_251);  permute_251 = None
        view_574: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_129, [8, 1024, 768]);  mm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_84: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 84)
        inductor_random_default_39: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_84, 'rand');  inductor_lookup_seed_default_84 = None
        gt_85: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_39, 0.1);  inductor_random_default_39 = None
        mul_255: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_85, view_574);  view_574 = None
        mul_256: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_255, 1.1111111111111112);  mul_255 = None
        add_122: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_119, mul_256);  mul_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_43: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_122, 2)
        mean_42: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_43, [-1], True);  pow_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_123: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_42, 1e-06);  mean_42 = None
        rsqrt_42: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
        mul_257: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_122, rsqrt_42)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_258: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_178, mul_257);  mul_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_252: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_179, [1, 0])
        view_575: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_258, [8192, 768]);  mul_258 = None
        mm_130: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_575, permute_252);  permute_252 = None
        view_576: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_130, [8, 1024, 3072]);  mm_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_17: "f32[8, 1024, 3072]" = torch.ops.aten.relu.default(view_576);  view_576 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_85: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 85)
        inductor_random_default_38: "f32[8, 1024, 3072]" = torch.ops.prims.inductor_random.default([8, 1024, 3072], inductor_lookup_seed_default_85, 'rand');  inductor_lookup_seed_default_85 = None
        gt_86: "b8[8, 1024, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_38, 0.1);  inductor_random_default_38 = None
        mul_259: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(gt_86, relu_17)
        mul_260: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_259, 1.1111111111111112);  mul_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_253: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_180, [1, 0])
        view_577: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_260, [8192, 3072]);  mul_260 = None
        mm_131: "f32[8192, 768]" = torch.ops.aten.mm.default(view_577, permute_253);  permute_253 = None
        view_578: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_131, [8, 1024, 768]);  mm_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_86: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 86)
        inductor_random_default_37: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_86, 'rand');  inductor_lookup_seed_default_86 = None
        gt_87: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_37, 0.1);  inductor_random_default_37 = None
        mul_261: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_87, view_578);  view_578 = None
        mul_262: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_261, 1.1111111111111112);  mul_261 = None
        add_124: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_122, mul_262);  mul_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_44: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_124, 2)
        mean_43: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_44, [-1], True);  pow_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_125: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_43, 1e-06);  mean_43 = None
        rsqrt_43: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_125);  add_125 = None
        mul_263: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_124, rsqrt_43)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_264: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_181, mul_263);  mul_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_254: "f32[768, 768]" = torch.ops.aten.permute.default(primals_182, [1, 0])
        view_579: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_264, [8192, 768]);  mul_264 = None
        mm_132: "f32[8192, 768]" = torch.ops.aten.mm.default(view_579, permute_254);  permute_254 = None
        view_580: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_132, [8, 1024, 768]);  mm_132 = None
        view_581: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_580, [8, 1024, -1, 64]);  view_580 = None
        permute_255: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_581, [0, 2, 1, 3]);  view_581 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_256: "f32[768, 768]" = torch.ops.aten.permute.default(primals_183, [1, 0])
        mm_133: "f32[8192, 768]" = torch.ops.aten.mm.default(view_579, permute_256);  permute_256 = None
        view_583: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_133, [8, 1024, 768]);  mm_133 = None
        view_584: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_583, [8, 1024, -1, 64]);  view_583 = None
        permute_257: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_584, [0, 2, 1, 3]);  view_584 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_258: "f32[768, 768]" = torch.ops.aten.permute.default(primals_184, [1, 0])
        mm_134: "f32[8192, 768]" = torch.ops.aten.mm.default(view_579, permute_258);  permute_258 = None
        view_586: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_134, [8, 1024, 768]);  mm_134 = None
        view_587: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_586, [8, 1024, -1, 64]);  view_586 = None
        permute_259: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_587, [0, 2, 1, 3]);  view_587 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_260: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_257, [0, 1, 3, 2]);  permute_257 = None
        expand_99: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_255, [8, 12, 1024, 64]);  permute_255 = None
        clone_97: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_99, memory_format = torch.contiguous_format);  expand_99 = None
        view_588: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_97, [96, 1024, 64]);  clone_97 = None
        expand_100: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_260, [8, 12, 64, 1024]);  permute_260 = None
        clone_98: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_100, memory_format = torch.contiguous_format);  expand_100 = None
        view_589: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_98, [96, 64, 1024]);  clone_98 = None
        bmm_48: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_588, view_589)
        view_590: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_48, [8, 12, 1024, 1024]);  bmm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_126: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_590, add_76);  view_590 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_24: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_126, [-1], True)
        sub_26: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_126, amax_24);  add_126 = amax_24 = None
        exp_24: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
        sum_25: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [-1], True)
        div_28: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_24, sum_25);  exp_24 = sum_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_87: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 87)
        inductor_random_default_36: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_87, 'rand');  inductor_lookup_seed_default_87 = None
        gt_88: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_36, 0.1);  inductor_random_default_36 = None
        mul_265: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_88, div_28)
        mul_266: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_265, 1.1111111111111112);  mul_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_101: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_266, [8, 12, 1024, 1024]);  mul_266 = None
        view_594: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_101, [96, 1024, 1024]);  expand_101 = None
        expand_102: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_259, [8, 12, 1024, 64]);  permute_259 = None
        clone_99: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_102, memory_format = torch.contiguous_format);  expand_102 = None
        view_595: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_99, [96, 1024, 64]);  clone_99 = None
        bmm_49: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_594, view_595)
        view_596: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_49, [8, 12, 1024, 64]);  bmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_261: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_596, [0, 2, 1, 3]);  view_596 = None
        clone_100: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_261, memory_format = torch.contiguous_format);  permute_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_597: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_100, [8, 1024, -1]);  clone_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_262: "f32[768, 768]" = torch.ops.aten.permute.default(primals_185, [1, 0])
        view_598: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_597, [8192, 768]);  view_597 = None
        mm_135: "f32[8192, 768]" = torch.ops.aten.mm.default(view_598, permute_262);  permute_262 = None
        view_599: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_135, [8, 1024, 768]);  mm_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_88: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 88)
        inductor_random_default_35: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_88, 'rand');  inductor_lookup_seed_default_88 = None
        gt_89: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_35, 0.1);  inductor_random_default_35 = None
        mul_267: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_89, view_599);  view_599 = None
        mul_268: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_267, 1.1111111111111112);  mul_267 = None
        add_127: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_124, mul_268);  mul_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_45: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_127, 2)
        mean_44: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_45, [-1], True);  pow_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_128: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_44, 1e-06);  mean_44 = None
        rsqrt_44: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
        mul_269: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_127, rsqrt_44)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_270: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_186, mul_269);  mul_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_263: "f32[768, 768]" = torch.ops.aten.permute.default(primals_187, [1, 0])
        view_600: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_270, [8192, 768]);  mul_270 = None
        mm_136: "f32[8192, 768]" = torch.ops.aten.mm.default(view_600, permute_263);  permute_263 = None
        view_601: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_136, [8, 1024, 768]);  mm_136 = None
        view_602: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_601, [8, 1024, -1, 64]);  view_601 = None
        permute_264: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_602, [0, 2, 1, 3]);  view_602 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_265: "f32[768, 768]" = torch.ops.aten.permute.default(primals_188, [1, 0])
        view_603: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_137: "f32[8192, 768]" = torch.ops.aten.mm.default(view_603, permute_265);  view_603 = permute_265 = None
        view_604: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_137, [8, 1024, 768]);  mm_137 = None
        view_605: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_604, [8, 1024, -1, 64]);  view_604 = None
        permute_266: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_605, [0, 2, 1, 3]);  view_605 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_267: "f32[768, 768]" = torch.ops.aten.permute.default(primals_189, [1, 0])
        view_606: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_138: "f32[8192, 768]" = torch.ops.aten.mm.default(view_606, permute_267);  view_606 = permute_267 = None
        view_607: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_138, [8, 1024, 768]);  mm_138 = None
        view_608: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_607, [8, 1024, -1, 64]);  view_607 = None
        permute_268: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_608, [0, 2, 1, 3]);  view_608 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_269: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_266, [0, 1, 3, 2]);  permute_266 = None
        expand_103: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_264, [8, 12, 1024, 64]);  permute_264 = None
        clone_101: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_103, memory_format = torch.contiguous_format);  expand_103 = None
        view_609: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_101, [96, 1024, 64]);  clone_101 = None
        expand_104: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_269, [8, 12, 64, 1024]);  permute_269 = None
        clone_102: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_104, memory_format = torch.contiguous_format);  expand_104 = None
        view_610: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_102, [96, 64, 1024]);  clone_102 = None
        bmm_50: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_609, view_610)
        view_611: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_50, [8, 12, 1024, 1024]);  bmm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_129: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_611, add_80);  view_611 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_25: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_129, [-1], True)
        sub_27: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_129, amax_25);  add_129 = amax_25 = None
        exp_25: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
        sum_26: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_25, [-1], True)
        div_29: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_25, sum_26);  exp_25 = sum_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_89: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 89)
        inductor_random_default_34: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_89, 'rand');  inductor_lookup_seed_default_89 = None
        gt_90: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_34, 0.1);  inductor_random_default_34 = None
        mul_271: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_90, div_29)
        mul_272: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_271, 1.1111111111111112);  mul_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_105: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_272, [8, 12, 1024, 1024]);  mul_272 = None
        view_615: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_105, [96, 1024, 1024]);  expand_105 = None
        expand_106: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_268, [8, 12, 1024, 64]);  permute_268 = None
        clone_103: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_106, memory_format = torch.contiguous_format);  expand_106 = None
        view_616: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_103, [96, 1024, 64]);  clone_103 = None
        bmm_51: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_615, view_616)
        view_617: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_51, [8, 12, 1024, 64]);  bmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_270: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_617, [0, 2, 1, 3]);  view_617 = None
        clone_104: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_270, memory_format = torch.contiguous_format);  permute_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_618: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_104, [8, 1024, -1]);  clone_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_271: "f32[768, 768]" = torch.ops.aten.permute.default(primals_190, [1, 0])
        view_619: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_618, [8192, 768]);  view_618 = None
        mm_139: "f32[8192, 768]" = torch.ops.aten.mm.default(view_619, permute_271);  permute_271 = None
        view_620: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_139, [8, 1024, 768]);  mm_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_90: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 90)
        inductor_random_default_33: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_90, 'rand');  inductor_lookup_seed_default_90 = None
        gt_91: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_33, 0.1);  inductor_random_default_33 = None
        mul_273: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_91, view_620);  view_620 = None
        mul_274: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_273, 1.1111111111111112);  mul_273 = None
        add_130: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_127, mul_274);  mul_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_46: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_130, 2)
        mean_45: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_46, [-1], True);  pow_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_131: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_45, 1e-06);  mean_45 = None
        rsqrt_45: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
        mul_275: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_130, rsqrt_45)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_276: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_191, mul_275);  mul_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_272: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_192, [1, 0])
        view_621: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_276, [8192, 768]);  mul_276 = None
        mm_140: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_621, permute_272);  permute_272 = None
        view_622: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_140, [8, 1024, 3072]);  mm_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_18: "f32[8, 1024, 3072]" = torch.ops.aten.relu.default(view_622);  view_622 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_91: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 91)
        inductor_random_default_32: "f32[8, 1024, 3072]" = torch.ops.prims.inductor_random.default([8, 1024, 3072], inductor_lookup_seed_default_91, 'rand');  inductor_lookup_seed_default_91 = None
        gt_92: "b8[8, 1024, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_32, 0.1);  inductor_random_default_32 = None
        mul_277: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(gt_92, relu_18)
        mul_278: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_277, 1.1111111111111112);  mul_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_273: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_193, [1, 0])
        view_623: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_278, [8192, 3072]);  mul_278 = None
        mm_141: "f32[8192, 768]" = torch.ops.aten.mm.default(view_623, permute_273);  permute_273 = None
        view_624: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_141, [8, 1024, 768]);  mm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_92: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 92)
        inductor_random_default_31: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_92, 'rand');  inductor_lookup_seed_default_92 = None
        gt_93: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_31, 0.1);  inductor_random_default_31 = None
        mul_279: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_93, view_624);  view_624 = None
        mul_280: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_279, 1.1111111111111112);  mul_279 = None
        add_132: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_130, mul_280);  mul_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_47: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_132, 2)
        mean_46: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_47, [-1], True);  pow_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_133: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_46, 1e-06);  mean_46 = None
        rsqrt_46: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
        mul_281: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_132, rsqrt_46)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_282: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_194, mul_281);  mul_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_274: "f32[768, 768]" = torch.ops.aten.permute.default(primals_195, [1, 0])
        view_625: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_282, [8192, 768]);  mul_282 = None
        mm_142: "f32[8192, 768]" = torch.ops.aten.mm.default(view_625, permute_274);  permute_274 = None
        view_626: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_142, [8, 1024, 768]);  mm_142 = None
        view_627: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_626, [8, 1024, -1, 64]);  view_626 = None
        permute_275: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_627, [0, 2, 1, 3]);  view_627 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_276: "f32[768, 768]" = torch.ops.aten.permute.default(primals_196, [1, 0])
        mm_143: "f32[8192, 768]" = torch.ops.aten.mm.default(view_625, permute_276);  permute_276 = None
        view_629: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_143, [8, 1024, 768]);  mm_143 = None
        view_630: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_629, [8, 1024, -1, 64]);  view_629 = None
        permute_277: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_630, [0, 2, 1, 3]);  view_630 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_278: "f32[768, 768]" = torch.ops.aten.permute.default(primals_197, [1, 0])
        mm_144: "f32[8192, 768]" = torch.ops.aten.mm.default(view_625, permute_278);  permute_278 = None
        view_632: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_144, [8, 1024, 768]);  mm_144 = None
        view_633: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_632, [8, 1024, -1, 64]);  view_632 = None
        permute_279: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_633, [0, 2, 1, 3]);  view_633 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_280: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_277, [0, 1, 3, 2]);  permute_277 = None
        expand_107: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_275, [8, 12, 1024, 64]);  permute_275 = None
        clone_105: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_107, memory_format = torch.contiguous_format);  expand_107 = None
        view_634: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_105, [96, 1024, 64]);  clone_105 = None
        expand_108: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_280, [8, 12, 64, 1024]);  permute_280 = None
        clone_106: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_108, memory_format = torch.contiguous_format);  expand_108 = None
        view_635: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_106, [96, 64, 1024]);  clone_106 = None
        bmm_52: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_634, view_635)
        view_636: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_52, [8, 12, 1024, 1024]);  bmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_134: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_636, add_76);  view_636 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_26: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_134, [-1], True)
        sub_28: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_134, amax_26);  add_134 = amax_26 = None
        exp_26: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_27: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_26, [-1], True)
        div_30: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_26, sum_27);  exp_26 = sum_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_93: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 93)
        inductor_random_default_30: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_93, 'rand');  inductor_lookup_seed_default_93 = None
        gt_94: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_30, 0.1);  inductor_random_default_30 = None
        mul_283: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_94, div_30)
        mul_284: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_283, 1.1111111111111112);  mul_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_109: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_284, [8, 12, 1024, 1024]);  mul_284 = None
        view_640: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_109, [96, 1024, 1024]);  expand_109 = None
        expand_110: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_279, [8, 12, 1024, 64]);  permute_279 = None
        clone_107: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_110, memory_format = torch.contiguous_format);  expand_110 = None
        view_641: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_107, [96, 1024, 64]);  clone_107 = None
        bmm_53: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_640, view_641)
        view_642: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_53, [8, 12, 1024, 64]);  bmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_281: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_642, [0, 2, 1, 3]);  view_642 = None
        clone_108: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_281, memory_format = torch.contiguous_format);  permute_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_643: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_108, [8, 1024, -1]);  clone_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_282: "f32[768, 768]" = torch.ops.aten.permute.default(primals_198, [1, 0])
        view_644: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_643, [8192, 768]);  view_643 = None
        mm_145: "f32[8192, 768]" = torch.ops.aten.mm.default(view_644, permute_282);  permute_282 = None
        view_645: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_145, [8, 1024, 768]);  mm_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_94: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 94)
        inductor_random_default_29: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_94, 'rand');  inductor_lookup_seed_default_94 = None
        gt_95: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_29, 0.1);  inductor_random_default_29 = None
        mul_285: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_95, view_645);  view_645 = None
        mul_286: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_285, 1.1111111111111112);  mul_285 = None
        add_135: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_132, mul_286);  mul_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_48: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_135, 2)
        mean_47: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_48, [-1], True);  pow_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_136: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_47, 1e-06);  mean_47 = None
        rsqrt_47: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
        mul_287: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_135, rsqrt_47)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_288: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_199, mul_287);  mul_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_283: "f32[768, 768]" = torch.ops.aten.permute.default(primals_200, [1, 0])
        view_646: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_288, [8192, 768]);  mul_288 = None
        mm_146: "f32[8192, 768]" = torch.ops.aten.mm.default(view_646, permute_283);  permute_283 = None
        view_647: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_146, [8, 1024, 768]);  mm_146 = None
        view_648: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_647, [8, 1024, -1, 64]);  view_647 = None
        permute_284: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_648, [0, 2, 1, 3]);  view_648 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_285: "f32[768, 768]" = torch.ops.aten.permute.default(primals_201, [1, 0])
        view_649: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_147: "f32[8192, 768]" = torch.ops.aten.mm.default(view_649, permute_285);  view_649 = permute_285 = None
        view_650: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_147, [8, 1024, 768]);  mm_147 = None
        view_651: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_650, [8, 1024, -1, 64]);  view_650 = None
        permute_286: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_651, [0, 2, 1, 3]);  view_651 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_287: "f32[768, 768]" = torch.ops.aten.permute.default(primals_202, [1, 0])
        view_652: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_148: "f32[8192, 768]" = torch.ops.aten.mm.default(view_652, permute_287);  view_652 = permute_287 = None
        view_653: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_148, [8, 1024, 768]);  mm_148 = None
        view_654: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_653, [8, 1024, -1, 64]);  view_653 = None
        permute_288: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_654, [0, 2, 1, 3]);  view_654 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_289: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_286, [0, 1, 3, 2]);  permute_286 = None
        expand_111: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_284, [8, 12, 1024, 64]);  permute_284 = None
        clone_109: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_111, memory_format = torch.contiguous_format);  expand_111 = None
        view_655: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_109, [96, 1024, 64]);  clone_109 = None
        expand_112: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_289, [8, 12, 64, 1024]);  permute_289 = None
        clone_110: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_112, memory_format = torch.contiguous_format);  expand_112 = None
        view_656: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_110, [96, 64, 1024]);  clone_110 = None
        bmm_54: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_655, view_656)
        view_657: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_54, [8, 12, 1024, 1024]);  bmm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_137: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_657, add_80);  view_657 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_27: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_137, [-1], True)
        sub_29: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_137, amax_27);  add_137 = amax_27 = None
        exp_27: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_29);  sub_29 = None
        sum_28: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_27, [-1], True)
        div_31: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_27, sum_28);  exp_27 = sum_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_95: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 95)
        inductor_random_default_28: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_95, 'rand');  inductor_lookup_seed_default_95 = None
        gt_96: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_28, 0.1);  inductor_random_default_28 = None
        mul_289: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_96, div_31)
        mul_290: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_289, 1.1111111111111112);  mul_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_113: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_290, [8, 12, 1024, 1024]);  mul_290 = None
        view_661: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_113, [96, 1024, 1024]);  expand_113 = None
        expand_114: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_288, [8, 12, 1024, 64]);  permute_288 = None
        clone_111: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_114, memory_format = torch.contiguous_format);  expand_114 = None
        view_662: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_111, [96, 1024, 64]);  clone_111 = None
        bmm_55: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_661, view_662)
        view_663: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_55, [8, 12, 1024, 64]);  bmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_290: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_663, [0, 2, 1, 3]);  view_663 = None
        clone_112: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_290, memory_format = torch.contiguous_format);  permute_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_664: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_112, [8, 1024, -1]);  clone_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_291: "f32[768, 768]" = torch.ops.aten.permute.default(primals_203, [1, 0])
        view_665: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_664, [8192, 768]);  view_664 = None
        mm_149: "f32[8192, 768]" = torch.ops.aten.mm.default(view_665, permute_291);  permute_291 = None
        view_666: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_149, [8, 1024, 768]);  mm_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_96: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 96)
        inductor_random_default_27: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_96, 'rand');  inductor_lookup_seed_default_96 = None
        gt_97: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_27, 0.1);  inductor_random_default_27 = None
        mul_291: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_97, view_666);  view_666 = None
        mul_292: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_291, 1.1111111111111112);  mul_291 = None
        add_138: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_135, mul_292);  mul_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_49: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_138, 2)
        mean_48: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_49, [-1], True);  pow_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_139: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_48, 1e-06);  mean_48 = None
        rsqrt_48: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
        mul_293: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_138, rsqrt_48)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_294: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_204, mul_293);  mul_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_292: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_205, [1, 0])
        view_667: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_294, [8192, 768]);  mul_294 = None
        mm_150: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_667, permute_292);  permute_292 = None
        view_668: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_150, [8, 1024, 3072]);  mm_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_19: "f32[8, 1024, 3072]" = torch.ops.aten.relu.default(view_668);  view_668 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_97: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 97)
        inductor_random_default_26: "f32[8, 1024, 3072]" = torch.ops.prims.inductor_random.default([8, 1024, 3072], inductor_lookup_seed_default_97, 'rand');  inductor_lookup_seed_default_97 = None
        gt_98: "b8[8, 1024, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_26, 0.1);  inductor_random_default_26 = None
        mul_295: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(gt_98, relu_19)
        mul_296: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_295, 1.1111111111111112);  mul_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_293: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_206, [1, 0])
        view_669: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_296, [8192, 3072]);  mul_296 = None
        mm_151: "f32[8192, 768]" = torch.ops.aten.mm.default(view_669, permute_293);  permute_293 = None
        view_670: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_151, [8, 1024, 768]);  mm_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_98: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 98)
        inductor_random_default_25: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_98, 'rand');  inductor_lookup_seed_default_98 = None
        gt_99: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_25, 0.1);  inductor_random_default_25 = None
        mul_297: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_99, view_670);  view_670 = None
        mul_298: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_297, 1.1111111111111112);  mul_297 = None
        add_140: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_138, mul_298);  mul_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_50: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_140, 2)
        mean_49: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_50, [-1], True);  pow_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_141: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_49, 1e-06);  mean_49 = None
        rsqrt_49: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
        mul_299: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_140, rsqrt_49)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_300: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_207, mul_299);  mul_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_294: "f32[768, 768]" = torch.ops.aten.permute.default(primals_208, [1, 0])
        view_671: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_300, [8192, 768]);  mul_300 = None
        mm_152: "f32[8192, 768]" = torch.ops.aten.mm.default(view_671, permute_294);  permute_294 = None
        view_672: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_152, [8, 1024, 768]);  mm_152 = None
        view_673: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_672, [8, 1024, -1, 64]);  view_672 = None
        permute_295: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_673, [0, 2, 1, 3]);  view_673 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_296: "f32[768, 768]" = torch.ops.aten.permute.default(primals_209, [1, 0])
        mm_153: "f32[8192, 768]" = torch.ops.aten.mm.default(view_671, permute_296);  permute_296 = None
        view_675: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_153, [8, 1024, 768]);  mm_153 = None
        view_676: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_675, [8, 1024, -1, 64]);  view_675 = None
        permute_297: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_676, [0, 2, 1, 3]);  view_676 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_298: "f32[768, 768]" = torch.ops.aten.permute.default(primals_210, [1, 0])
        mm_154: "f32[8192, 768]" = torch.ops.aten.mm.default(view_671, permute_298);  permute_298 = None
        view_678: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_154, [8, 1024, 768]);  mm_154 = None
        view_679: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_678, [8, 1024, -1, 64]);  view_678 = None
        permute_299: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_679, [0, 2, 1, 3]);  view_679 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_300: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_297, [0, 1, 3, 2]);  permute_297 = None
        expand_115: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_295, [8, 12, 1024, 64]);  permute_295 = None
        clone_113: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_115, memory_format = torch.contiguous_format);  expand_115 = None
        view_680: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_113, [96, 1024, 64]);  clone_113 = None
        expand_116: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_300, [8, 12, 64, 1024]);  permute_300 = None
        clone_114: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_116, memory_format = torch.contiguous_format);  expand_116 = None
        view_681: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_114, [96, 64, 1024]);  clone_114 = None
        bmm_56: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_680, view_681)
        view_682: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_56, [8, 12, 1024, 1024]);  bmm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_142: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_682, add_76);  view_682 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_28: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_142, [-1], True)
        sub_30: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_142, amax_28);  add_142 = amax_28 = None
        exp_28: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_30);  sub_30 = None
        sum_29: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_28, [-1], True)
        div_32: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_28, sum_29);  exp_28 = sum_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_99: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 99)
        inductor_random_default_24: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_99, 'rand');  inductor_lookup_seed_default_99 = None
        gt_100: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_24, 0.1);  inductor_random_default_24 = None
        mul_301: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_100, div_32)
        mul_302: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_301, 1.1111111111111112);  mul_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_117: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_302, [8, 12, 1024, 1024]);  mul_302 = None
        view_686: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_117, [96, 1024, 1024]);  expand_117 = None
        expand_118: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_299, [8, 12, 1024, 64]);  permute_299 = None
        clone_115: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_118, memory_format = torch.contiguous_format);  expand_118 = None
        view_687: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_115, [96, 1024, 64]);  clone_115 = None
        bmm_57: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_686, view_687)
        view_688: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_57, [8, 12, 1024, 64]);  bmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_301: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_688, [0, 2, 1, 3]);  view_688 = None
        clone_116: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_301, memory_format = torch.contiguous_format);  permute_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_689: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_116, [8, 1024, -1]);  clone_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_302: "f32[768, 768]" = torch.ops.aten.permute.default(primals_211, [1, 0])
        view_690: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_689, [8192, 768]);  view_689 = None
        mm_155: "f32[8192, 768]" = torch.ops.aten.mm.default(view_690, permute_302);  permute_302 = None
        view_691: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_155, [8, 1024, 768]);  mm_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_100: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 100)
        inductor_random_default_23: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_100, 'rand');  inductor_lookup_seed_default_100 = None
        gt_101: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_23, 0.1);  inductor_random_default_23 = None
        mul_303: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_101, view_691);  view_691 = None
        mul_304: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_303, 1.1111111111111112);  mul_303 = None
        add_143: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_140, mul_304);  mul_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_51: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_143, 2)
        mean_50: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_51, [-1], True);  pow_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_144: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_50, 1e-06);  mean_50 = None
        rsqrt_50: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
        mul_305: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_143, rsqrt_50)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_306: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_212, mul_305);  mul_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_303: "f32[768, 768]" = torch.ops.aten.permute.default(primals_213, [1, 0])
        view_692: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_306, [8192, 768]);  mul_306 = None
        mm_156: "f32[8192, 768]" = torch.ops.aten.mm.default(view_692, permute_303);  permute_303 = None
        view_693: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_156, [8, 1024, 768]);  mm_156 = None
        view_694: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_693, [8, 1024, -1, 64]);  view_693 = None
        permute_304: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_694, [0, 2, 1, 3]);  view_694 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_305: "f32[768, 768]" = torch.ops.aten.permute.default(primals_214, [1, 0])
        view_695: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_157: "f32[8192, 768]" = torch.ops.aten.mm.default(view_695, permute_305);  view_695 = permute_305 = None
        view_696: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_157, [8, 1024, 768]);  mm_157 = None
        view_697: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_696, [8, 1024, -1, 64]);  view_696 = None
        permute_306: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_697, [0, 2, 1, 3]);  view_697 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_307: "f32[768, 768]" = torch.ops.aten.permute.default(primals_215, [1, 0])
        view_698: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_158: "f32[8192, 768]" = torch.ops.aten.mm.default(view_698, permute_307);  view_698 = permute_307 = None
        view_699: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_158, [8, 1024, 768]);  mm_158 = None
        view_700: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_699, [8, 1024, -1, 64]);  view_699 = None
        permute_308: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_700, [0, 2, 1, 3]);  view_700 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_309: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_306, [0, 1, 3, 2]);  permute_306 = None
        expand_119: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_304, [8, 12, 1024, 64]);  permute_304 = None
        clone_117: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_119, memory_format = torch.contiguous_format);  expand_119 = None
        view_701: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_117, [96, 1024, 64]);  clone_117 = None
        expand_120: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_309, [8, 12, 64, 1024]);  permute_309 = None
        clone_118: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_120, memory_format = torch.contiguous_format);  expand_120 = None
        view_702: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_118, [96, 64, 1024]);  clone_118 = None
        bmm_58: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_701, view_702)
        view_703: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_58, [8, 12, 1024, 1024]);  bmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_145: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_703, add_80);  view_703 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_29: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_145, [-1], True)
        sub_31: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_145, amax_29);  add_145 = amax_29 = None
        exp_29: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_30: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_29, [-1], True)
        div_33: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_29, sum_30);  exp_29 = sum_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_101: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 101)
        inductor_random_default_22: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_101, 'rand');  inductor_lookup_seed_default_101 = None
        gt_102: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_22, 0.1);  inductor_random_default_22 = None
        mul_307: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_102, div_33)
        mul_308: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_307, 1.1111111111111112);  mul_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_121: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_308, [8, 12, 1024, 1024]);  mul_308 = None
        view_707: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_121, [96, 1024, 1024]);  expand_121 = None
        expand_122: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_308, [8, 12, 1024, 64]);  permute_308 = None
        clone_119: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_122, memory_format = torch.contiguous_format);  expand_122 = None
        view_708: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_119, [96, 1024, 64]);  clone_119 = None
        bmm_59: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_707, view_708)
        view_709: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_59, [8, 12, 1024, 64]);  bmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_310: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_709, [0, 2, 1, 3]);  view_709 = None
        clone_120: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_310, memory_format = torch.contiguous_format);  permute_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_710: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_120, [8, 1024, -1]);  clone_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_311: "f32[768, 768]" = torch.ops.aten.permute.default(primals_216, [1, 0])
        view_711: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_710, [8192, 768]);  view_710 = None
        mm_159: "f32[8192, 768]" = torch.ops.aten.mm.default(view_711, permute_311);  permute_311 = None
        view_712: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_159, [8, 1024, 768]);  mm_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_102: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 102)
        inductor_random_default_21: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_102, 'rand');  inductor_lookup_seed_default_102 = None
        gt_103: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_21, 0.1);  inductor_random_default_21 = None
        mul_309: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_103, view_712);  view_712 = None
        mul_310: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_309, 1.1111111111111112);  mul_309 = None
        add_146: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_143, mul_310);  mul_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_52: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_146, 2)
        mean_51: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_52, [-1], True);  pow_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_147: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_51, 1e-06);  mean_51 = None
        rsqrt_51: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
        mul_311: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_146, rsqrt_51)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_312: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_217, mul_311);  mul_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_312: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_218, [1, 0])
        view_713: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_312, [8192, 768]);  mul_312 = None
        mm_160: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_713, permute_312);  permute_312 = None
        view_714: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_160, [8, 1024, 3072]);  mm_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_20: "f32[8, 1024, 3072]" = torch.ops.aten.relu.default(view_714);  view_714 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_103: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 103)
        inductor_random_default_20: "f32[8, 1024, 3072]" = torch.ops.prims.inductor_random.default([8, 1024, 3072], inductor_lookup_seed_default_103, 'rand');  inductor_lookup_seed_default_103 = None
        gt_104: "b8[8, 1024, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_20, 0.1);  inductor_random_default_20 = None
        mul_313: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(gt_104, relu_20)
        mul_314: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_313, 1.1111111111111112);  mul_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_313: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_219, [1, 0])
        view_715: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_314, [8192, 3072]);  mul_314 = None
        mm_161: "f32[8192, 768]" = torch.ops.aten.mm.default(view_715, permute_313);  permute_313 = None
        view_716: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_161, [8, 1024, 768]);  mm_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_104: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 104)
        inductor_random_default_19: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_104, 'rand');  inductor_lookup_seed_default_104 = None
        gt_105: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_19, 0.1);  inductor_random_default_19 = None
        mul_315: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_105, view_716);  view_716 = None
        mul_316: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_315, 1.1111111111111112);  mul_315 = None
        add_148: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_146, mul_316);  mul_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_53: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_148, 2)
        mean_52: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_53, [-1], True);  pow_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_149: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_52, 1e-06);  mean_52 = None
        rsqrt_52: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
        mul_317: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_148, rsqrt_52)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_318: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_220, mul_317);  mul_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_314: "f32[768, 768]" = torch.ops.aten.permute.default(primals_221, [1, 0])
        view_717: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_318, [8192, 768]);  mul_318 = None
        mm_162: "f32[8192, 768]" = torch.ops.aten.mm.default(view_717, permute_314);  permute_314 = None
        view_718: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_162, [8, 1024, 768]);  mm_162 = None
        view_719: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_718, [8, 1024, -1, 64]);  view_718 = None
        permute_315: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_719, [0, 2, 1, 3]);  view_719 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_316: "f32[768, 768]" = torch.ops.aten.permute.default(primals_222, [1, 0])
        mm_163: "f32[8192, 768]" = torch.ops.aten.mm.default(view_717, permute_316);  permute_316 = None
        view_721: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_163, [8, 1024, 768]);  mm_163 = None
        view_722: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_721, [8, 1024, -1, 64]);  view_721 = None
        permute_317: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_722, [0, 2, 1, 3]);  view_722 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_318: "f32[768, 768]" = torch.ops.aten.permute.default(primals_223, [1, 0])
        mm_164: "f32[8192, 768]" = torch.ops.aten.mm.default(view_717, permute_318);  permute_318 = None
        view_724: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_164, [8, 1024, 768]);  mm_164 = None
        view_725: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_724, [8, 1024, -1, 64]);  view_724 = None
        permute_319: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_725, [0, 2, 1, 3]);  view_725 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_320: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_317, [0, 1, 3, 2]);  permute_317 = None
        expand_123: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_315, [8, 12, 1024, 64]);  permute_315 = None
        clone_121: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_123, memory_format = torch.contiguous_format);  expand_123 = None
        view_726: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_121, [96, 1024, 64]);  clone_121 = None
        expand_124: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_320, [8, 12, 64, 1024]);  permute_320 = None
        clone_122: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_124, memory_format = torch.contiguous_format);  expand_124 = None
        view_727: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_122, [96, 64, 1024]);  clone_122 = None
        bmm_60: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_726, view_727)
        view_728: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_60, [8, 12, 1024, 1024]);  bmm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_150: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_728, add_76);  view_728 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_30: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_150, [-1], True)
        sub_32: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_150, amax_30);  add_150 = amax_30 = None
        exp_30: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_32);  sub_32 = None
        sum_31: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_30, [-1], True)
        div_34: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_30, sum_31);  exp_30 = sum_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_105: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 105)
        inductor_random_default_18: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_105, 'rand');  inductor_lookup_seed_default_105 = None
        gt_106: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_18, 0.1);  inductor_random_default_18 = None
        mul_319: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_106, div_34)
        mul_320: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_319, 1.1111111111111112);  mul_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_125: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_320, [8, 12, 1024, 1024]);  mul_320 = None
        view_732: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_125, [96, 1024, 1024]);  expand_125 = None
        expand_126: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_319, [8, 12, 1024, 64]);  permute_319 = None
        clone_123: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_126, memory_format = torch.contiguous_format);  expand_126 = None
        view_733: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_123, [96, 1024, 64]);  clone_123 = None
        bmm_61: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_732, view_733)
        view_734: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_61, [8, 12, 1024, 64]);  bmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_321: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_734, [0, 2, 1, 3]);  view_734 = None
        clone_124: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_321, memory_format = torch.contiguous_format);  permute_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_735: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_124, [8, 1024, -1]);  clone_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_322: "f32[768, 768]" = torch.ops.aten.permute.default(primals_224, [1, 0])
        view_736: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_735, [8192, 768]);  view_735 = None
        mm_165: "f32[8192, 768]" = torch.ops.aten.mm.default(view_736, permute_322);  permute_322 = None
        view_737: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_165, [8, 1024, 768]);  mm_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_106: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 106)
        inductor_random_default_17: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_106, 'rand');  inductor_lookup_seed_default_106 = None
        gt_107: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_17, 0.1);  inductor_random_default_17 = None
        mul_321: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_107, view_737);  view_737 = None
        mul_322: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_321, 1.1111111111111112);  mul_321 = None
        add_151: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_148, mul_322);  mul_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_54: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_151, 2)
        mean_53: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_54, [-1], True);  pow_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_152: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_53, 1e-06);  mean_53 = None
        rsqrt_53: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
        mul_323: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_151, rsqrt_53)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_324: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_225, mul_323);  mul_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_323: "f32[768, 768]" = torch.ops.aten.permute.default(primals_226, [1, 0])
        view_738: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_324, [8192, 768]);  mul_324 = None
        mm_166: "f32[8192, 768]" = torch.ops.aten.mm.default(view_738, permute_323);  permute_323 = None
        view_739: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_166, [8, 1024, 768]);  mm_166 = None
        view_740: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_739, [8, 1024, -1, 64]);  view_739 = None
        permute_324: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_740, [0, 2, 1, 3]);  view_740 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_325: "f32[768, 768]" = torch.ops.aten.permute.default(primals_227, [1, 0])
        view_741: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_167: "f32[8192, 768]" = torch.ops.aten.mm.default(view_741, permute_325);  view_741 = permute_325 = None
        view_742: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_167, [8, 1024, 768]);  mm_167 = None
        view_743: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_742, [8, 1024, -1, 64]);  view_742 = None
        permute_326: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_743, [0, 2, 1, 3]);  view_743 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_327: "f32[768, 768]" = torch.ops.aten.permute.default(primals_228, [1, 0])
        view_744: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_168: "f32[8192, 768]" = torch.ops.aten.mm.default(view_744, permute_327);  view_744 = permute_327 = None
        view_745: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_168, [8, 1024, 768]);  mm_168 = None
        view_746: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_745, [8, 1024, -1, 64]);  view_745 = None
        permute_328: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_746, [0, 2, 1, 3]);  view_746 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_329: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_326, [0, 1, 3, 2]);  permute_326 = None
        expand_127: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_324, [8, 12, 1024, 64]);  permute_324 = None
        clone_125: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_127, memory_format = torch.contiguous_format);  expand_127 = None
        view_747: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_125, [96, 1024, 64]);  clone_125 = None
        expand_128: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_329, [8, 12, 64, 1024]);  permute_329 = None
        clone_126: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_128, memory_format = torch.contiguous_format);  expand_128 = None
        view_748: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_126, [96, 64, 1024]);  clone_126 = None
        bmm_62: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_747, view_748)
        view_749: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_62, [8, 12, 1024, 1024]);  bmm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_153: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_749, add_80);  view_749 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_31: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_153, [-1], True)
        sub_33: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_153, amax_31);  add_153 = amax_31 = None
        exp_31: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_33);  sub_33 = None
        sum_32: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_31, [-1], True)
        div_35: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_31, sum_32);  exp_31 = sum_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_107: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 107)
        inductor_random_default_16: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_107, 'rand');  inductor_lookup_seed_default_107 = None
        gt_108: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_16, 0.1);  inductor_random_default_16 = None
        mul_325: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_108, div_35)
        mul_326: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_325, 1.1111111111111112);  mul_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_129: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_326, [8, 12, 1024, 1024]);  mul_326 = None
        view_753: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_129, [96, 1024, 1024]);  expand_129 = None
        expand_130: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_328, [8, 12, 1024, 64]);  permute_328 = None
        clone_127: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_130, memory_format = torch.contiguous_format);  expand_130 = None
        view_754: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_127, [96, 1024, 64]);  clone_127 = None
        bmm_63: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_753, view_754)
        view_755: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_63, [8, 12, 1024, 64]);  bmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_330: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_755, [0, 2, 1, 3]);  view_755 = None
        clone_128: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_330, memory_format = torch.contiguous_format);  permute_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_756: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_128, [8, 1024, -1]);  clone_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_331: "f32[768, 768]" = torch.ops.aten.permute.default(primals_229, [1, 0])
        view_757: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_756, [8192, 768]);  view_756 = None
        mm_169: "f32[8192, 768]" = torch.ops.aten.mm.default(view_757, permute_331);  permute_331 = None
        view_758: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_169, [8, 1024, 768]);  mm_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_108: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 108)
        inductor_random_default_15: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_108, 'rand');  inductor_lookup_seed_default_108 = None
        gt_109: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_15, 0.1);  inductor_random_default_15 = None
        mul_327: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_109, view_758);  view_758 = None
        mul_328: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_327, 1.1111111111111112);  mul_327 = None
        add_154: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_151, mul_328);  mul_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_55: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_154, 2)
        mean_54: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_55, [-1], True);  pow_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_155: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_54, 1e-06);  mean_54 = None
        rsqrt_54: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
        mul_329: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_154, rsqrt_54)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_330: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_230, mul_329);  mul_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_332: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_231, [1, 0])
        view_759: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_330, [8192, 768]);  mul_330 = None
        mm_170: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_759, permute_332);  permute_332 = None
        view_760: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_170, [8, 1024, 3072]);  mm_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_21: "f32[8, 1024, 3072]" = torch.ops.aten.relu.default(view_760);  view_760 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_109: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 109)
        inductor_random_default_14: "f32[8, 1024, 3072]" = torch.ops.prims.inductor_random.default([8, 1024, 3072], inductor_lookup_seed_default_109, 'rand');  inductor_lookup_seed_default_109 = None
        gt_110: "b8[8, 1024, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_14, 0.1);  inductor_random_default_14 = None
        mul_331: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(gt_110, relu_21)
        mul_332: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_331, 1.1111111111111112);  mul_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_333: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_232, [1, 0])
        view_761: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_332, [8192, 3072]);  mul_332 = None
        mm_171: "f32[8192, 768]" = torch.ops.aten.mm.default(view_761, permute_333);  permute_333 = None
        view_762: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_171, [8, 1024, 768]);  mm_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_110: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 110)
        inductor_random_default_13: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_110, 'rand');  inductor_lookup_seed_default_110 = None
        gt_111: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_13, 0.1);  inductor_random_default_13 = None
        mul_333: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_111, view_762);  view_762 = None
        mul_334: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_333, 1.1111111111111112);  mul_333 = None
        add_156: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_154, mul_334);  mul_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_56: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_156, 2)
        mean_55: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_56, [-1], True);  pow_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_157: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_55, 1e-06);  mean_55 = None
        rsqrt_55: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
        mul_335: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_156, rsqrt_55)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_336: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_233, mul_335);  mul_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_334: "f32[768, 768]" = torch.ops.aten.permute.default(primals_234, [1, 0])
        view_763: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_336, [8192, 768]);  mul_336 = None
        mm_172: "f32[8192, 768]" = torch.ops.aten.mm.default(view_763, permute_334);  permute_334 = None
        view_764: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_172, [8, 1024, 768]);  mm_172 = None
        view_765: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_764, [8, 1024, -1, 64]);  view_764 = None
        permute_335: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_765, [0, 2, 1, 3]);  view_765 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_336: "f32[768, 768]" = torch.ops.aten.permute.default(primals_235, [1, 0])
        mm_173: "f32[8192, 768]" = torch.ops.aten.mm.default(view_763, permute_336);  permute_336 = None
        view_767: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_173, [8, 1024, 768]);  mm_173 = None
        view_768: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_767, [8, 1024, -1, 64]);  view_767 = None
        permute_337: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_768, [0, 2, 1, 3]);  view_768 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_338: "f32[768, 768]" = torch.ops.aten.permute.default(primals_236, [1, 0])
        mm_174: "f32[8192, 768]" = torch.ops.aten.mm.default(view_763, permute_338);  permute_338 = None
        view_770: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_174, [8, 1024, 768]);  mm_174 = None
        view_771: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_770, [8, 1024, -1, 64]);  view_770 = None
        permute_339: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_771, [0, 2, 1, 3]);  view_771 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_340: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_337, [0, 1, 3, 2]);  permute_337 = None
        expand_131: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_335, [8, 12, 1024, 64]);  permute_335 = None
        clone_129: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_131, memory_format = torch.contiguous_format);  expand_131 = None
        view_772: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_129, [96, 1024, 64]);  clone_129 = None
        expand_132: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_340, [8, 12, 64, 1024]);  permute_340 = None
        clone_130: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_132, memory_format = torch.contiguous_format);  expand_132 = None
        view_773: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_130, [96, 64, 1024]);  clone_130 = None
        bmm_64: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_772, view_773)
        view_774: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_64, [8, 12, 1024, 1024]);  bmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_158: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_774, add_76);  view_774 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_32: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_158, [-1], True)
        sub_34: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_158, amax_32);  add_158 = amax_32 = None
        exp_32: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_33: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_32, [-1], True)
        div_36: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_32, sum_33);  exp_32 = sum_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_111: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 111)
        inductor_random_default_12: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_111, 'rand');  inductor_lookup_seed_default_111 = None
        gt_112: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_12, 0.1);  inductor_random_default_12 = None
        mul_337: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_112, div_36)
        mul_338: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_337, 1.1111111111111112);  mul_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_133: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_338, [8, 12, 1024, 1024]);  mul_338 = None
        view_778: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_133, [96, 1024, 1024]);  expand_133 = None
        expand_134: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_339, [8, 12, 1024, 64]);  permute_339 = None
        clone_131: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_134, memory_format = torch.contiguous_format);  expand_134 = None
        view_779: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_131, [96, 1024, 64]);  clone_131 = None
        bmm_65: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_778, view_779)
        view_780: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_65, [8, 12, 1024, 64]);  bmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_341: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_780, [0, 2, 1, 3]);  view_780 = None
        clone_132: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_341, memory_format = torch.contiguous_format);  permute_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_781: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_132, [8, 1024, -1]);  clone_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_342: "f32[768, 768]" = torch.ops.aten.permute.default(primals_237, [1, 0])
        view_782: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_781, [8192, 768]);  view_781 = None
        mm_175: "f32[8192, 768]" = torch.ops.aten.mm.default(view_782, permute_342);  permute_342 = None
        view_783: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_175, [8, 1024, 768]);  mm_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_112: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 112)
        inductor_random_default_11: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_112, 'rand');  inductor_lookup_seed_default_112 = None
        gt_113: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_11, 0.1);  inductor_random_default_11 = None
        mul_339: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_113, view_783);  view_783 = None
        mul_340: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_339, 1.1111111111111112);  mul_339 = None
        add_159: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_156, mul_340);  mul_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_57: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_159, 2)
        mean_56: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_57, [-1], True);  pow_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_160: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_56, 1e-06);  mean_56 = None
        rsqrt_56: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
        mul_341: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_159, rsqrt_56)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_342: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_238, mul_341);  mul_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_343: "f32[768, 768]" = torch.ops.aten.permute.default(primals_239, [1, 0])
        view_784: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_342, [8192, 768]);  mul_342 = None
        mm_176: "f32[8192, 768]" = torch.ops.aten.mm.default(view_784, permute_343);  permute_343 = None
        view_785: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_176, [8, 1024, 768]);  mm_176 = None
        view_786: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_785, [8, 1024, -1, 64]);  view_785 = None
        permute_344: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_786, [0, 2, 1, 3]);  view_786 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_345: "f32[768, 768]" = torch.ops.aten.permute.default(primals_240, [1, 0])
        view_787: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_177: "f32[8192, 768]" = torch.ops.aten.mm.default(view_787, permute_345);  view_787 = permute_345 = None
        view_788: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_177, [8, 1024, 768]);  mm_177 = None
        view_789: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_788, [8, 1024, -1, 64]);  view_788 = None
        permute_346: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_789, [0, 2, 1, 3]);  view_789 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_347: "f32[768, 768]" = torch.ops.aten.permute.default(primals_241, [1, 0])
        view_790: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_178: "f32[8192, 768]" = torch.ops.aten.mm.default(view_790, permute_347);  view_790 = permute_347 = None
        view_791: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_178, [8, 1024, 768]);  mm_178 = None
        view_792: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_791, [8, 1024, -1, 64]);  view_791 = None
        permute_348: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_792, [0, 2, 1, 3]);  view_792 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_349: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_346, [0, 1, 3, 2]);  permute_346 = None
        expand_135: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_344, [8, 12, 1024, 64]);  permute_344 = None
        clone_133: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_135, memory_format = torch.contiguous_format);  expand_135 = None
        view_793: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_133, [96, 1024, 64]);  clone_133 = None
        expand_136: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_349, [8, 12, 64, 1024]);  permute_349 = None
        clone_134: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_136, memory_format = torch.contiguous_format);  expand_136 = None
        view_794: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_134, [96, 64, 1024]);  clone_134 = None
        bmm_66: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_793, view_794)
        view_795: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_66, [8, 12, 1024, 1024]);  bmm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_161: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_795, add_80);  view_795 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_33: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_161, [-1], True)
        sub_35: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_161, amax_33);  add_161 = amax_33 = None
        exp_33: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_35);  sub_35 = None
        sum_34: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_33, [-1], True)
        div_37: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_33, sum_34);  exp_33 = sum_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_113: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 113)
        inductor_random_default_10: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_113, 'rand');  inductor_lookup_seed_default_113 = None
        gt_114: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_10, 0.1);  inductor_random_default_10 = None
        mul_343: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_114, div_37)
        mul_344: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_343, 1.1111111111111112);  mul_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_137: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_344, [8, 12, 1024, 1024]);  mul_344 = None
        view_799: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_137, [96, 1024, 1024]);  expand_137 = None
        expand_138: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_348, [8, 12, 1024, 64]);  permute_348 = None
        clone_135: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_138, memory_format = torch.contiguous_format);  expand_138 = None
        view_800: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_135, [96, 1024, 64]);  clone_135 = None
        bmm_67: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_799, view_800)
        view_801: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_67, [8, 12, 1024, 64]);  bmm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_350: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_801, [0, 2, 1, 3]);  view_801 = None
        clone_136: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_350, memory_format = torch.contiguous_format);  permute_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_802: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_136, [8, 1024, -1]);  clone_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_351: "f32[768, 768]" = torch.ops.aten.permute.default(primals_242, [1, 0])
        view_803: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_802, [8192, 768]);  view_802 = None
        mm_179: "f32[8192, 768]" = torch.ops.aten.mm.default(view_803, permute_351);  permute_351 = None
        view_804: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_179, [8, 1024, 768]);  mm_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_114: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 114)
        inductor_random_default_9: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_114, 'rand');  inductor_lookup_seed_default_114 = None
        gt_115: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_9, 0.1);  inductor_random_default_9 = None
        mul_345: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_115, view_804);  view_804 = None
        mul_346: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_345, 1.1111111111111112);  mul_345 = None
        add_162: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_159, mul_346);  mul_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_58: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_162, 2)
        mean_57: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_58, [-1], True);  pow_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_163: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_57, 1e-06);  mean_57 = None
        rsqrt_57: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
        mul_347: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_162, rsqrt_57)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_348: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_243, mul_347);  mul_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_352: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_244, [1, 0])
        view_805: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_348, [8192, 768]);  mul_348 = None
        mm_180: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_805, permute_352);  permute_352 = None
        view_806: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_180, [8, 1024, 3072]);  mm_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_22: "f32[8, 1024, 3072]" = torch.ops.aten.relu.default(view_806);  view_806 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_115: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 115)
        inductor_random_default_8: "f32[8, 1024, 3072]" = torch.ops.prims.inductor_random.default([8, 1024, 3072], inductor_lookup_seed_default_115, 'rand');  inductor_lookup_seed_default_115 = None
        gt_116: "b8[8, 1024, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_8, 0.1);  inductor_random_default_8 = None
        mul_349: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(gt_116, relu_22)
        mul_350: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_349, 1.1111111111111112);  mul_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_353: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_245, [1, 0])
        view_807: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_350, [8192, 3072]);  mul_350 = None
        mm_181: "f32[8192, 768]" = torch.ops.aten.mm.default(view_807, permute_353);  permute_353 = None
        view_808: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_181, [8, 1024, 768]);  mm_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_116: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 116)
        inductor_random_default_7: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_116, 'rand');  inductor_lookup_seed_default_116 = None
        gt_117: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_7, 0.1);  inductor_random_default_7 = None
        mul_351: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_117, view_808);  view_808 = None
        mul_352: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_351, 1.1111111111111112);  mul_351 = None
        add_164: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_162, mul_352);  mul_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_59: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_164, 2)
        mean_58: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_59, [-1], True);  pow_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_165: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_58, 1e-06);  mean_58 = None
        rsqrt_58: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
        mul_353: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_164, rsqrt_58)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_354: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_246, mul_353);  mul_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_354: "f32[768, 768]" = torch.ops.aten.permute.default(primals_247, [1, 0])
        view_809: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_354, [8192, 768]);  mul_354 = None
        mm_182: "f32[8192, 768]" = torch.ops.aten.mm.default(view_809, permute_354);  permute_354 = None
        view_810: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_182, [8, 1024, 768]);  mm_182 = None
        view_811: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_810, [8, 1024, -1, 64]);  view_810 = None
        permute_355: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_811, [0, 2, 1, 3]);  view_811 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_356: "f32[768, 768]" = torch.ops.aten.permute.default(primals_248, [1, 0])
        mm_183: "f32[8192, 768]" = torch.ops.aten.mm.default(view_809, permute_356);  permute_356 = None
        view_813: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_183, [8, 1024, 768]);  mm_183 = None
        view_814: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_813, [8, 1024, -1, 64]);  view_813 = None
        permute_357: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_814, [0, 2, 1, 3]);  view_814 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_358: "f32[768, 768]" = torch.ops.aten.permute.default(primals_249, [1, 0])
        mm_184: "f32[8192, 768]" = torch.ops.aten.mm.default(view_809, permute_358);  permute_358 = None
        view_816: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_184, [8, 1024, 768]);  mm_184 = None
        view_817: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_816, [8, 1024, -1, 64]);  view_816 = None
        permute_359: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_817, [0, 2, 1, 3]);  view_817 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_360: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_357, [0, 1, 3, 2]);  permute_357 = None
        expand_139: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_355, [8, 12, 1024, 64]);  permute_355 = None
        clone_137: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_139, memory_format = torch.contiguous_format);  expand_139 = None
        view_818: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_137, [96, 1024, 64]);  clone_137 = None
        expand_140: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_360, [8, 12, 64, 1024]);  permute_360 = None
        clone_138: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_140, memory_format = torch.contiguous_format);  expand_140 = None
        view_819: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_138, [96, 64, 1024]);  clone_138 = None
        bmm_68: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_818, view_819)
        view_820: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_68, [8, 12, 1024, 1024]);  bmm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_166: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_820, add_76);  view_820 = add_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_34: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_166, [-1], True)
        sub_36: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_166, amax_34);  add_166 = amax_34 = None
        exp_34: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_36);  sub_36 = None
        sum_35: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_34, [-1], True)
        div_38: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_34, sum_35);  exp_34 = sum_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_117: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 117)
        inductor_random_default_6: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_117, 'rand');  inductor_lookup_seed_default_117 = None
        gt_118: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_6, 0.1);  inductor_random_default_6 = None
        mul_355: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_118, div_38)
        mul_356: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_355, 1.1111111111111112);  mul_355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_141: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_356, [8, 12, 1024, 1024]);  mul_356 = None
        view_824: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_141, [96, 1024, 1024]);  expand_141 = None
        expand_142: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_359, [8, 12, 1024, 64]);  permute_359 = None
        clone_139: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_142, memory_format = torch.contiguous_format);  expand_142 = None
        view_825: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_139, [96, 1024, 64]);  clone_139 = None
        bmm_69: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_824, view_825)
        view_826: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_69, [8, 12, 1024, 64]);  bmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_361: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_826, [0, 2, 1, 3]);  view_826 = None
        clone_140: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_361, memory_format = torch.contiguous_format);  permute_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_827: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_140, [8, 1024, -1]);  clone_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_362: "f32[768, 768]" = torch.ops.aten.permute.default(primals_250, [1, 0])
        view_828: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_827, [8192, 768]);  view_827 = None
        mm_185: "f32[8192, 768]" = torch.ops.aten.mm.default(view_828, permute_362);  permute_362 = None
        view_829: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_185, [8, 1024, 768]);  mm_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_118: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 118)
        inductor_random_default_5: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_118, 'rand');  inductor_lookup_seed_default_118 = None
        gt_119: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_5, 0.1);  inductor_random_default_5 = None
        mul_357: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_119, view_829);  view_829 = None
        mul_358: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_357, 1.1111111111111112);  mul_357 = None
        add_167: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_164, mul_358);  mul_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_60: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_167, 2)
        mean_59: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_60, [-1], True);  pow_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_168: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_59, 1e-06);  mean_59 = None
        rsqrt_59: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
        mul_359: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_167, rsqrt_59)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_360: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_251, mul_359);  mul_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_363: "f32[768, 768]" = torch.ops.aten.permute.default(primals_252, [1, 0])
        view_830: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_360, [8192, 768]);  mul_360 = None
        mm_186: "f32[8192, 768]" = torch.ops.aten.mm.default(view_830, permute_363);  permute_363 = None
        view_831: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_186, [8, 1024, 768]);  mm_186 = None
        view_832: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_831, [8, 1024, -1, 64]);  view_831 = None
        permute_364: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_832, [0, 2, 1, 3]);  view_832 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_365: "f32[768, 768]" = torch.ops.aten.permute.default(primals_253, [1, 0])
        view_833: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_187: "f32[8192, 768]" = torch.ops.aten.mm.default(view_833, permute_365);  view_833 = permute_365 = None
        view_834: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_187, [8, 1024, 768]);  mm_187 = None
        view_835: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_834, [8, 1024, -1, 64]);  view_834 = None
        permute_366: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_835, [0, 2, 1, 3]);  view_835 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_367: "f32[768, 768]" = torch.ops.aten.permute.default(primals_254, [1, 0])
        view_836: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_188: "f32[8192, 768]" = torch.ops.aten.mm.default(view_836, permute_367);  view_836 = permute_367 = None
        view_837: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_188, [8, 1024, 768]);  mm_188 = None
        view_838: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_837, [8, 1024, -1, 64]);  view_837 = None
        permute_368: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_838, [0, 2, 1, 3]);  view_838 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_369: "f32[8, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_366, [0, 1, 3, 2]);  permute_366 = None
        expand_143: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_364, [8, 12, 1024, 64]);  permute_364 = None
        clone_141: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_143, memory_format = torch.contiguous_format);  expand_143 = None
        view_839: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_141, [96, 1024, 64]);  clone_141 = None
        expand_144: "f32[8, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_369, [8, 12, 64, 1024]);  permute_369 = None
        clone_142: "f32[8, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_144, memory_format = torch.contiguous_format);  expand_144 = None
        view_840: "f32[96, 64, 1024]" = torch.ops.aten.reshape.default(clone_142, [96, 64, 1024]);  clone_142 = None
        bmm_70: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_839, view_840)
        view_841: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_70, [8, 12, 1024, 1024]);  bmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_169: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_841, add_80);  view_841 = add_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_35: "f32[8, 12, 1024, 1]" = torch.ops.aten.amax.default(add_169, [-1], True)
        sub_37: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_169, amax_35);  add_169 = amax_35 = None
        exp_35: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
        sum_36: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_35, [-1], True)
        div_39: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_35, sum_36);  exp_35 = sum_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_119: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 119)
        inductor_random_default_4: "f32[8, 12, 1024, 1024]" = torch.ops.prims.inductor_random.default([8, 12, 1024, 1024], inductor_lookup_seed_default_119, 'rand');  inductor_lookup_seed_default_119 = None
        gt_120: "b8[8, 12, 1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_4, 0.1);  inductor_random_default_4 = None
        mul_361: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_120, div_39)
        mul_362: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_361, 1.1111111111111112);  mul_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_145: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(mul_362, [8, 12, 1024, 1024]);  mul_362 = None
        view_845: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(expand_145, [96, 1024, 1024]);  expand_145 = None
        expand_146: "f32[8, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_368, [8, 12, 1024, 64]);  permute_368 = None
        clone_143: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_146, memory_format = torch.contiguous_format);  expand_146 = None
        view_846: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_143, [96, 1024, 64]);  clone_143 = None
        bmm_71: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_845, view_846)
        view_847: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_71, [8, 12, 1024, 64]);  bmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_370: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_847, [0, 2, 1, 3]);  view_847 = None
        clone_144: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_370, memory_format = torch.contiguous_format);  permute_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_848: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_144, [8, 1024, -1]);  clone_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_371: "f32[768, 768]" = torch.ops.aten.permute.default(primals_255, [1, 0])
        view_849: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_848, [8192, 768]);  view_848 = None
        mm_189: "f32[8192, 768]" = torch.ops.aten.mm.default(view_849, permute_371);  permute_371 = None
        view_850: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_189, [8, 1024, 768]);  mm_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_120: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 120)
        inductor_random_default_3: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_120, 'rand');  inductor_lookup_seed_default_120 = None
        gt_121: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_3, 0.1);  inductor_random_default_3 = None
        mul_363: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_121, view_850);  view_850 = None
        mul_364: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_363, 1.1111111111111112);  mul_363 = None
        add_170: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_167, mul_364);  mul_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_61: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_170, 2)
        mean_60: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_61, [-1], True);  pow_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_171: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_60, 1e-06);  mean_60 = None
        rsqrt_60: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
        mul_365: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_170, rsqrt_60)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_366: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_256, mul_365);  mul_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_372: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_257, [1, 0])
        view_851: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_366, [8192, 768]);  mul_366 = None
        mm_190: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_851, permute_372);  permute_372 = None
        view_852: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_190, [8, 1024, 3072]);  mm_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_23: "f32[8, 1024, 3072]" = torch.ops.aten.relu.default(view_852);  view_852 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_121: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 121)
        inductor_random_default_2: "f32[8, 1024, 3072]" = torch.ops.prims.inductor_random.default([8, 1024, 3072], inductor_lookup_seed_default_121, 'rand');  inductor_lookup_seed_default_121 = None
        gt_122: "b8[8, 1024, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_2, 0.1);  inductor_random_default_2 = None
        mul_367: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(gt_122, relu_23)
        mul_368: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_367, 1.1111111111111112);  mul_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_373: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_258, [1, 0])
        view_853: "f32[8192, 3072]" = torch.ops.aten.reshape.default(mul_368, [8192, 3072]);  mul_368 = None
        mm_191: "f32[8192, 768]" = torch.ops.aten.mm.default(view_853, permute_373);  permute_373 = None
        view_854: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_191, [8, 1024, 768]);  mm_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_122: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 122)
        inductor_random_default_1: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_122, 'rand');  inductor_lookup_seed_default_122 = None
        gt_123: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.1);  inductor_random_default_1 = None
        mul_369: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_123, view_854);  view_854 = None
        mul_370: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_369, 1.1111111111111112);  mul_369 = None
        add_172: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_170, mul_370);  mul_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_62: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_172, 2)
        mean_61: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_62, [-1], True);  pow_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_173: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_61, 1e-06);  mean_61 = None
        rsqrt_61: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_173);  add_173 = None
        mul_371: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_172, rsqrt_61)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_372: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(primals_259, mul_371);  mul_371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:755 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_123: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 123);  inductor_seeds_default = None
        inductor_random_default: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default_123, 'rand');  inductor_lookup_seed_default_123 = None
        gt_124: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_373: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_124, mul_372);  mul_372 = None
        mul_374: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_373, 1.1111111111111112);  mul_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1095 in forward, code: sequence_output = sequence_output * (self.model_dim**-0.5)
        mul_375: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_374, 0.03608439182435161);  mul_374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1097 in forward, code: lm_logits = self.lm_head(sequence_output)
        permute_374: "f32[768, 32128]" = torch.ops.aten.permute.default(primals_2, [1, 0])
        view_855: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_375, [8192, 768]);  mul_375 = None
        mm_192: "f32[8192, 32128]" = torch.ops.aten.mm.default(view_855, permute_374);  permute_374 = None
        view_856: "f32[8, 1024, 32128]" = torch.ops.aten.reshape.default(mm_192, [8, 1024, 32128]);  mm_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1104 in forward, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        view_857: "f32[8192, 32128]" = torch.ops.aten.reshape.default(view_856, [-1, 32128])
        view_858: "i64[8192]" = torch.ops.aten.reshape.default(primals_101, [-1])
        amax_36: "f32[8192, 1]" = torch.ops.aten.amax.default(view_857, [1], True)
        sub_38: "f32[8192, 32128]" = torch.ops.aten.sub.Tensor(view_857, amax_36);  view_857 = None
        exp_36: "f32[8192, 32128]" = torch.ops.aten.exp.default(sub_38)
        sum_37: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(exp_36, [1], True);  exp_36 = None
        log_2: "f32[8192, 1]" = torch.ops.aten.log.default(sum_37);  sum_37 = None
        sub_39: "f32[8192, 32128]" = torch.ops.aten.sub.Tensor(sub_38, log_2);  sub_38 = None
        ne: "b8[8192]" = torch.ops.aten.ne.Scalar(view_858, -100)
        where_6: "i64[8192]" = torch.ops.aten.where.self(ne, view_858, full_default_4);  view_858 = full_default_4 = None
        unsqueeze_18: "i64[8192, 1]" = torch.ops.aten.unsqueeze.default(where_6, 1);  where_6 = None
        gather: "f32[8192, 1]" = torch.ops.aten.gather.default(sub_39, 1, unsqueeze_18);  sub_39 = unsqueeze_18 = None
        squeeze: "f32[8192]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg_1: "f32[8192]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        where_7: "f32[8192]" = torch.ops.aten.where.self(ne, neg_1, full_default);  neg_1 = full_default = None
        sum_38: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type_5: "f32[]" = torch.ops.prims.convert_element_type.default(sum_38, torch.float32);  sum_38 = None
        sum_39: "f32[]" = torch.ops.aten.sum.default(where_7);  where_7 = None
        div_40: "f32[]" = torch.ops.aten.div.Tensor(sum_39, convert_element_type_5);  sum_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_1: "b8[8, 1024, 3072]" = torch.ops.aten.le.Scalar(relu_23, 0);  relu_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_392: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_845, [0, 2, 1]);  view_845 = None
        permute_393: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_846, [0, 2, 1]);  view_846 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_394: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_839, [0, 2, 1]);  view_839 = None
        permute_395: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_840, [0, 2, 1]);  view_840 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_417: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_824, [0, 2, 1]);  view_824 = None
        permute_418: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_825, [0, 2, 1]);  view_825 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_419: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_818, [0, 2, 1]);  view_818 = None
        permute_420: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_819, [0, 2, 1]);  view_819 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_2: "b8[8, 1024, 3072]" = torch.ops.aten.le.Scalar(relu_22, 0);  relu_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_450: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_799, [0, 2, 1]);  view_799 = None
        permute_451: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_800, [0, 2, 1]);  view_800 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_452: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_793, [0, 2, 1]);  view_793 = None
        permute_453: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_794, [0, 2, 1]);  view_794 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_475: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_778, [0, 2, 1]);  view_778 = None
        permute_476: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_779, [0, 2, 1]);  view_779 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_477: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_772, [0, 2, 1]);  view_772 = None
        permute_478: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_773, [0, 2, 1]);  view_773 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_3: "b8[8, 1024, 3072]" = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_508: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_753, [0, 2, 1]);  view_753 = None
        permute_509: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_754, [0, 2, 1]);  view_754 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_510: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_747, [0, 2, 1]);  view_747 = None
        permute_511: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_748, [0, 2, 1]);  view_748 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_533: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_732, [0, 2, 1]);  view_732 = None
        permute_534: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_733, [0, 2, 1]);  view_733 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_535: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_726, [0, 2, 1]);  view_726 = None
        permute_536: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_727, [0, 2, 1]);  view_727 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_4: "b8[8, 1024, 3072]" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_566: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_707, [0, 2, 1]);  view_707 = None
        permute_567: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_708, [0, 2, 1]);  view_708 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_568: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_701, [0, 2, 1]);  view_701 = None
        permute_569: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_702, [0, 2, 1]);  view_702 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_591: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_686, [0, 2, 1]);  view_686 = None
        permute_592: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_687, [0, 2, 1]);  view_687 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_593: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_680, [0, 2, 1]);  view_680 = None
        permute_594: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_681, [0, 2, 1]);  view_681 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_5: "b8[8, 1024, 3072]" = torch.ops.aten.le.Scalar(relu_19, 0);  relu_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_624: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_661, [0, 2, 1]);  view_661 = None
        permute_625: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_662, [0, 2, 1]);  view_662 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_626: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_655, [0, 2, 1]);  view_655 = None
        permute_627: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_656, [0, 2, 1]);  view_656 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_649: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_640, [0, 2, 1]);  view_640 = None
        permute_650: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_641, [0, 2, 1]);  view_641 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_651: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_634, [0, 2, 1]);  view_634 = None
        permute_652: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_635, [0, 2, 1]);  view_635 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_6: "b8[8, 1024, 3072]" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_682: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_615, [0, 2, 1]);  view_615 = None
        permute_683: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_616, [0, 2, 1]);  view_616 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_684: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_609, [0, 2, 1]);  view_609 = None
        permute_685: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_610, [0, 2, 1]);  view_610 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_707: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_594, [0, 2, 1]);  view_594 = None
        permute_708: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_595, [0, 2, 1]);  view_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_709: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_588, [0, 2, 1]);  view_588 = None
        permute_710: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_589, [0, 2, 1]);  view_589 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_7: "b8[8, 1024, 3072]" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_740: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_569, [0, 2, 1]);  view_569 = None
        permute_741: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_570, [0, 2, 1]);  view_570 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_742: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_563, [0, 2, 1]);  view_563 = None
        permute_743: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_564, [0, 2, 1]);  view_564 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_765: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_548, [0, 2, 1]);  view_548 = None
        permute_766: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_549, [0, 2, 1]);  view_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_767: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_542, [0, 2, 1]);  view_542 = None
        permute_768: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_543, [0, 2, 1]);  view_543 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_8: "b8[8, 1024, 3072]" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_798: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_523, [0, 2, 1]);  view_523 = None
        permute_799: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_524, [0, 2, 1]);  view_524 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_800: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_517, [0, 2, 1]);  view_517 = None
        permute_801: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_518, [0, 2, 1]);  view_518 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_823: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_502, [0, 2, 1]);  view_502 = None
        permute_824: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_503, [0, 2, 1]);  view_503 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_825: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_496, [0, 2, 1]);  view_496 = None
        permute_826: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_497, [0, 2, 1]);  view_497 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_9: "b8[8, 1024, 3072]" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_856: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_477, [0, 2, 1]);  view_477 = None
        permute_857: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_478, [0, 2, 1]);  view_478 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_858: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_471, [0, 2, 1]);  view_471 = None
        permute_859: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_472, [0, 2, 1]);  view_472 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_881: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_456, [0, 2, 1]);  view_456 = None
        permute_882: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_457, [0, 2, 1]);  view_457 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_883: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_450, [0, 2, 1]);  view_450 = None
        permute_884: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_451, [0, 2, 1]);  view_451 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_10: "b8[8, 1024, 3072]" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_914: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_431, [0, 2, 1]);  view_431 = None
        permute_915: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_432, [0, 2, 1]);  view_432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_916: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_425, [0, 2, 1]);  view_425 = None
        permute_917: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_426, [0, 2, 1]);  view_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_939: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_410, [0, 2, 1]);  view_410 = None
        permute_940: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_411, [0, 2, 1]);  view_411 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_941: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_404, [0, 2, 1]);  view_404 = None
        permute_942: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_405, [0, 2, 1]);  view_405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_11: "b8[8, 1024, 3072]" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_972: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_385, [0, 2, 1]);  view_385 = None
        permute_973: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_386, [0, 2, 1]);  view_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_974: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_379, [0, 2, 1]);  view_379 = None
        permute_975: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_380, [0, 2, 1]);  view_380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_997: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_364, [0, 2, 1]);  view_364 = None
        permute_998: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_365, [0, 2, 1]);  view_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_999: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_358, [0, 2, 1]);  view_358 = None
        permute_1000: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_359, [0, 2, 1]);  view_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_12: "b8[8, 1024, 3072]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_1030: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_339, [0, 2, 1]);  view_339 = None
        permute_1031: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_340, [0, 2, 1]);  view_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_1032: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_333, [0, 2, 1]);  view_333 = None
        permute_1033: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_334, [0, 2, 1]);  view_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_1055: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_318, [0, 2, 1]);  view_318 = None
        permute_1056: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_319, [0, 2, 1]);  view_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_1058: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_312, [0, 2, 1]);  view_312 = None
        permute_1059: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_313, [0, 2, 1]);  view_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_13: "b8[8, 1024, 3072]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_1089: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_291, [0, 2, 1]);  view_291 = None
        permute_1090: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_292, [0, 2, 1]);  view_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_1091: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_285, [0, 2, 1]);  view_285 = None
        permute_1092: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_286, [0, 2, 1]);  view_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_14: "b8[8, 1024, 3072]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_1122: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_266, [0, 2, 1]);  view_266 = None
        permute_1123: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_267, [0, 2, 1]);  view_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_1124: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_260, [0, 2, 1]);  view_260 = None
        permute_1125: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_261, [0, 2, 1]);  view_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_15: "b8[8, 1024, 3072]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_1155: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_241, [0, 2, 1]);  view_241 = None
        permute_1156: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_242, [0, 2, 1]);  view_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_1157: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_235, [0, 2, 1]);  view_235 = None
        permute_1158: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_236, [0, 2, 1]);  view_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_16: "b8[8, 1024, 3072]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_1188: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_216, [0, 2, 1]);  view_216 = None
        permute_1189: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_217, [0, 2, 1]);  view_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_1190: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
        permute_1191: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_211, [0, 2, 1]);  view_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_17: "b8[8, 1024, 3072]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_1221: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_191, [0, 2, 1]);  view_191 = None
        permute_1222: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_192, [0, 2, 1]);  view_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_1223: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_185, [0, 2, 1]);  view_185 = None
        permute_1224: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1]);  view_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_18: "b8[8, 1024, 3072]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_1254: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
        permute_1255: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_1256: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_160, [0, 2, 1]);  view_160 = None
        permute_1257: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_161, [0, 2, 1]);  view_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_19: "b8[8, 1024, 3072]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_1287: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_141, [0, 2, 1]);  view_141 = None
        permute_1288: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_142, [0, 2, 1]);  view_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_1289: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_135, [0, 2, 1]);  view_135 = None
        permute_1290: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1]);  view_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_20: "b8[8, 1024, 3072]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_1320: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_116, [0, 2, 1]);  view_116 = None
        permute_1321: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_117, [0, 2, 1]);  view_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_1322: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_110, [0, 2, 1]);  view_110 = None
        permute_1323: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_111, [0, 2, 1]);  view_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_21: "b8[8, 1024, 3072]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_1353: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_91, [0, 2, 1]);  view_91 = None
        permute_1354: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_92, [0, 2, 1]);  view_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_1355: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_85, [0, 2, 1]);  view_85 = None
        permute_1356: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_86, [0, 2, 1]);  view_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_22: "b8[8, 1024, 3072]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_1386: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_66, [0, 2, 1]);  view_66 = None
        permute_1387: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_67, [0, 2, 1]);  view_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_1388: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_60, [0, 2, 1]);  view_60 = None
        permute_1389: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_61, [0, 2, 1]);  view_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_23: "b8[8, 1024, 3072]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_1419: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_41, [0, 2, 1]);  view_41 = None
        permute_1420: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_42, [0, 2, 1]);  view_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_1421: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_35, [0, 2, 1]);  view_35 = None
        permute_1422: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_36, [0, 2, 1]);  view_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_24: "b8[8, 1024, 3072]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_1452: "f32[96, 1024, 1024]" = torch.ops.aten.permute.default(view_16, [0, 2, 1]);  view_16 = None
        permute_1453: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_17, [0, 2, 1]);  view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_1455: "f32[96, 64, 1024]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
        permute_1456: "f32[96, 1024, 64]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
        return (div_40, view_856, mul_151, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, embedding, ge, gt, rsqrt, view_1, bmm, add_6, embedding_1, amax, sum_1, gt_2, view_20, gt_3, add_9, rsqrt_1, view_22, gt_4, view_24, gt_5, add_11, rsqrt_2, view_26, div_3, gt_6, view_45, gt_7, add_14, rsqrt_3, view_47, gt_8, view_49, gt_9, add_16, rsqrt_4, view_51, div_4, gt_10, view_70, gt_11, add_19, rsqrt_5, view_72, gt_12, view_74, gt_13, add_21, rsqrt_6, view_76, div_5, gt_14, view_95, gt_15, add_24, rsqrt_7, view_97, gt_16, view_99, gt_17, add_26, rsqrt_8, view_101, div_6, gt_18, view_120, gt_19, add_29, rsqrt_9, view_122, gt_20, view_124, gt_21, add_31, rsqrt_10, view_126, div_7, gt_22, view_145, gt_23, add_34, rsqrt_11, view_147, gt_24, view_149, gt_25, add_36, rsqrt_12, view_151, div_8, gt_26, view_170, gt_27, add_39, rsqrt_13, view_172, gt_28, view_174, gt_29, add_41, rsqrt_14, view_176, div_9, gt_30, view_195, gt_31, add_44, rsqrt_15, view_197, gt_32, view_199, gt_33, add_46, rsqrt_16, view_201, div_10, gt_34, view_220, gt_35, add_49, rsqrt_17, view_222, gt_36, view_224, gt_37, add_51, rsqrt_18, view_226, div_11, gt_38, view_245, gt_39, add_54, rsqrt_19, view_247, gt_40, view_249, gt_41, add_56, rsqrt_20, view_251, div_12, gt_42, view_270, gt_43, add_59, rsqrt_21, view_272, gt_44, view_274, gt_45, add_61, rsqrt_22, view_276, div_13, gt_46, view_295, gt_47, add_64, rsqrt_23, view_297, gt_48, view_299, gt_49, add_66, rsqrt_24, gt_50, mul_151, where_2, embedding_2, gt_51, rsqrt_25, view_303, add_75, div_16, gt_52, view_322, gt_53, add_78, rsqrt_26, view_324, div_17, gt_54, view_343, gt_55, add_82, rsqrt_27, view_345, gt_56, view_347, gt_57, add_84, rsqrt_28, view_349, div_18, gt_58, view_368, gt_59, add_87, rsqrt_29, view_370, div_19, gt_60, view_389, gt_61, add_90, rsqrt_30, view_391, gt_62, view_393, gt_63, add_92, rsqrt_31, view_395, div_20, gt_64, view_414, gt_65, add_95, rsqrt_32, view_416, div_21, gt_66, view_435, gt_67, add_98, rsqrt_33, view_437, gt_68, view_439, gt_69, add_100, rsqrt_34, view_441, div_22, gt_70, view_460, gt_71, add_103, rsqrt_35, view_462, div_23, gt_72, view_481, gt_73, add_106, rsqrt_36, view_483, gt_74, view_485, gt_75, add_108, rsqrt_37, view_487, div_24, gt_76, view_506, gt_77, add_111, rsqrt_38, view_508, div_25, gt_78, view_527, gt_79, add_114, rsqrt_39, view_529, gt_80, view_531, gt_81, add_116, rsqrt_40, view_533, div_26, gt_82, view_552, gt_83, add_119, rsqrt_41, view_554, div_27, gt_84, view_573, gt_85, add_122, rsqrt_42, view_575, gt_86, view_577, gt_87, add_124, rsqrt_43, view_579, div_28, gt_88, view_598, gt_89, add_127, rsqrt_44, view_600, div_29, gt_90, view_619, gt_91, add_130, rsqrt_45, view_621, gt_92, view_623, gt_93, add_132, rsqrt_46, view_625, div_30, gt_94, view_644, gt_95, add_135, rsqrt_47, view_646, div_31, gt_96, view_665, gt_97, add_138, rsqrt_48, view_667, gt_98, view_669, gt_99, add_140, rsqrt_49, view_671, div_32, gt_100, view_690, gt_101, add_143, rsqrt_50, view_692, div_33, gt_102, view_711, gt_103, add_146, rsqrt_51, view_713, gt_104, view_715, gt_105, add_148, rsqrt_52, view_717, div_34, gt_106, view_736, gt_107, add_151, rsqrt_53, view_738, div_35, gt_108, view_757, gt_109, add_154, rsqrt_54, view_759, gt_110, view_761, gt_111, add_156, rsqrt_55, view_763, div_36, gt_112, view_782, gt_113, add_159, rsqrt_56, view_784, div_37, gt_114, view_803, gt_115, add_162, rsqrt_57, view_805, gt_116, view_807, gt_117, add_164, rsqrt_58, view_809, div_38, gt_118, view_828, gt_119, add_167, rsqrt_59, view_830, div_39, gt_120, view_849, gt_121, add_170, rsqrt_60, view_851, gt_122, view_853, gt_123, add_172, rsqrt_61, gt_124, view_855, view_856, amax_36, log_2, convert_element_type_5, le_1, permute_392, permute_393, permute_394, permute_395, permute_417, permute_418, permute_419, permute_420, le_2, permute_450, permute_451, permute_452, permute_453, permute_475, permute_476, permute_477, permute_478, le_3, permute_508, permute_509, permute_510, permute_511, permute_533, permute_534, permute_535, permute_536, le_4, permute_566, permute_567, permute_568, permute_569, permute_591, permute_592, permute_593, permute_594, le_5, permute_624, permute_625, permute_626, permute_627, permute_649, permute_650, permute_651, permute_652, le_6, permute_682, permute_683, permute_684, permute_685, permute_707, permute_708, permute_709, permute_710, le_7, permute_740, permute_741, permute_742, permute_743, permute_765, permute_766, permute_767, permute_768, le_8, permute_798, permute_799, permute_800, permute_801, permute_823, permute_824, permute_825, permute_826, le_9, permute_856, permute_857, permute_858, permute_859, permute_881, permute_882, permute_883, permute_884, le_10, permute_914, permute_915, permute_916, permute_917, permute_939, permute_940, permute_941, permute_942, le_11, permute_972, permute_973, permute_974, permute_975, permute_997, permute_998, permute_999, permute_1000, le_12, permute_1030, permute_1031, permute_1032, permute_1033, permute_1055, permute_1056, permute_1058, permute_1059, le_13, permute_1089, permute_1090, permute_1091, permute_1092, le_14, permute_1122, permute_1123, permute_1124, permute_1125, le_15, permute_1155, permute_1156, permute_1157, permute_1158, le_16, permute_1188, permute_1189, permute_1190, permute_1191, le_17, permute_1221, permute_1222, permute_1223, permute_1224, le_18, permute_1254, permute_1255, permute_1256, permute_1257, le_19, permute_1287, permute_1288, permute_1289, permute_1290, le_20, permute_1320, permute_1321, permute_1322, permute_1323, le_21, permute_1353, permute_1354, permute_1355, permute_1356, le_22, permute_1386, permute_1387, permute_1388, permute_1389, le_23, permute_1419, permute_1420, permute_1421, permute_1422, le_24, permute_1452, permute_1453, permute_1455, permute_1456)
