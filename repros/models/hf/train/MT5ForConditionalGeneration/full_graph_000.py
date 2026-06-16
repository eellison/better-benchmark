class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[32, 128][128, 1]cuda:0", primals_2: "f32[250112, 512][512, 1]cuda:0", primals_3: "f32[512][1]cuda:0", primals_4: "f32[384, 512][512, 1]cuda:0", primals_5: "f32[384, 512][512, 1]cuda:0", primals_6: "f32[384, 512][512, 1]cuda:0", primals_7: "f32[32, 6][6, 1]cuda:0", primals_8: "f32[512, 384][384, 1]cuda:0", primals_9: "f32[512][1]cuda:0", primals_10: "f32[1024, 512][512, 1]cuda:0", primals_11: "f32[1024, 512][512, 1]cuda:0", primals_12: "f32[512, 1024][1024, 1]cuda:0", primals_13: "f32[512][1]cuda:0", primals_14: "f32[384, 512][512, 1]cuda:0", primals_15: "f32[384, 512][512, 1]cuda:0", primals_16: "f32[384, 512][512, 1]cuda:0", primals_17: "f32[512, 384][384, 1]cuda:0", primals_18: "f32[512][1]cuda:0", primals_19: "f32[1024, 512][512, 1]cuda:0", primals_20: "f32[1024, 512][512, 1]cuda:0", primals_21: "f32[512, 1024][1024, 1]cuda:0", primals_22: "f32[512][1]cuda:0", primals_23: "f32[384, 512][512, 1]cuda:0", primals_24: "f32[384, 512][512, 1]cuda:0", primals_25: "f32[384, 512][512, 1]cuda:0", primals_26: "f32[512, 384][384, 1]cuda:0", primals_27: "f32[512][1]cuda:0", primals_28: "f32[1024, 512][512, 1]cuda:0", primals_29: "f32[1024, 512][512, 1]cuda:0", primals_30: "f32[512, 1024][1024, 1]cuda:0", primals_31: "f32[512][1]cuda:0", primals_32: "f32[384, 512][512, 1]cuda:0", primals_33: "f32[384, 512][512, 1]cuda:0", primals_34: "f32[384, 512][512, 1]cuda:0", primals_35: "f32[512, 384][384, 1]cuda:0", primals_36: "f32[512][1]cuda:0", primals_37: "f32[1024, 512][512, 1]cuda:0", primals_38: "f32[1024, 512][512, 1]cuda:0", primals_39: "f32[512, 1024][1024, 1]cuda:0", primals_40: "f32[512][1]cuda:0", primals_41: "f32[384, 512][512, 1]cuda:0", primals_42: "f32[384, 512][512, 1]cuda:0", primals_43: "f32[384, 512][512, 1]cuda:0", primals_44: "f32[512, 384][384, 1]cuda:0", primals_45: "f32[512][1]cuda:0", primals_46: "f32[1024, 512][512, 1]cuda:0", primals_47: "f32[1024, 512][512, 1]cuda:0", primals_48: "f32[512, 1024][1024, 1]cuda:0", primals_49: "f32[512][1]cuda:0", primals_50: "f32[384, 512][512, 1]cuda:0", primals_51: "f32[384, 512][512, 1]cuda:0", primals_52: "f32[384, 512][512, 1]cuda:0", primals_53: "f32[512, 384][384, 1]cuda:0", primals_54: "f32[512][1]cuda:0", primals_55: "f32[1024, 512][512, 1]cuda:0", primals_56: "f32[1024, 512][512, 1]cuda:0", primals_57: "f32[512, 1024][1024, 1]cuda:0", primals_58: "f32[512][1]cuda:0", primals_59: "f32[384, 512][512, 1]cuda:0", primals_60: "f32[384, 512][512, 1]cuda:0", primals_61: "f32[384, 512][512, 1]cuda:0", primals_62: "f32[512, 384][384, 1]cuda:0", primals_63: "f32[512][1]cuda:0", primals_64: "f32[1024, 512][512, 1]cuda:0", primals_65: "f32[1024, 512][512, 1]cuda:0", primals_66: "f32[512, 1024][1024, 1]cuda:0", primals_67: "f32[512][1]cuda:0", primals_68: "f32[384, 512][512, 1]cuda:0", primals_69: "f32[384, 512][512, 1]cuda:0", primals_70: "f32[384, 512][512, 1]cuda:0", primals_71: "f32[512, 384][384, 1]cuda:0", primals_72: "f32[512][1]cuda:0", primals_73: "f32[1024, 512][512, 1]cuda:0", primals_74: "f32[1024, 512][512, 1]cuda:0", primals_75: "f32[512, 1024][1024, 1]cuda:0", primals_76: "f32[512][1]cuda:0", primals_77: "i64[32, 128][128, 1]cuda:0", primals_78: "f32[512][1]cuda:0", primals_79: "f32[384, 512][512, 1]cuda:0", primals_80: "f32[384, 512][512, 1]cuda:0", primals_81: "f32[384, 512][512, 1]cuda:0", primals_82: "f32[32, 6][6, 1]cuda:0", primals_83: "f32[512, 384][384, 1]cuda:0", primals_84: "f32[512][1]cuda:0", primals_85: "f32[384, 512][512, 1]cuda:0", primals_86: "f32[384, 512][512, 1]cuda:0", primals_87: "f32[384, 512][512, 1]cuda:0", primals_88: "f32[512, 384][384, 1]cuda:0", primals_89: "f32[512][1]cuda:0", primals_90: "f32[1024, 512][512, 1]cuda:0", primals_91: "f32[1024, 512][512, 1]cuda:0", primals_92: "f32[512, 1024][1024, 1]cuda:0", primals_93: "f32[512][1]cuda:0", primals_94: "f32[384, 512][512, 1]cuda:0", primals_95: "f32[384, 512][512, 1]cuda:0", primals_96: "f32[384, 512][512, 1]cuda:0", primals_97: "f32[512, 384][384, 1]cuda:0", primals_98: "f32[512][1]cuda:0", primals_99: "f32[384, 512][512, 1]cuda:0", primals_100: "f32[384, 512][512, 1]cuda:0", primals_101: "f32[384, 512][512, 1]cuda:0", primals_102: "f32[512, 384][384, 1]cuda:0", primals_103: "f32[512][1]cuda:0", primals_104: "f32[1024, 512][512, 1]cuda:0", primals_105: "f32[1024, 512][512, 1]cuda:0", primals_106: "f32[512, 1024][1024, 1]cuda:0", primals_107: "f32[512][1]cuda:0", primals_108: "f32[384, 512][512, 1]cuda:0", primals_109: "f32[384, 512][512, 1]cuda:0", primals_110: "f32[384, 512][512, 1]cuda:0", primals_111: "f32[512, 384][384, 1]cuda:0", primals_112: "f32[512][1]cuda:0", primals_113: "f32[384, 512][512, 1]cuda:0", primals_114: "f32[384, 512][512, 1]cuda:0", primals_115: "f32[384, 512][512, 1]cuda:0", primals_116: "f32[512, 384][384, 1]cuda:0", primals_117: "f32[512][1]cuda:0", primals_118: "f32[1024, 512][512, 1]cuda:0", primals_119: "f32[1024, 512][512, 1]cuda:0", primals_120: "f32[512, 1024][1024, 1]cuda:0", primals_121: "f32[512][1]cuda:0", primals_122: "f32[384, 512][512, 1]cuda:0", primals_123: "f32[384, 512][512, 1]cuda:0", primals_124: "f32[384, 512][512, 1]cuda:0", primals_125: "f32[512, 384][384, 1]cuda:0", primals_126: "f32[512][1]cuda:0", primals_127: "f32[384, 512][512, 1]cuda:0", primals_128: "f32[384, 512][512, 1]cuda:0", primals_129: "f32[384, 512][512, 1]cuda:0", primals_130: "f32[512, 384][384, 1]cuda:0", primals_131: "f32[512][1]cuda:0", primals_132: "f32[1024, 512][512, 1]cuda:0", primals_133: "f32[1024, 512][512, 1]cuda:0", primals_134: "f32[512, 1024][1024, 1]cuda:0", primals_135: "f32[512][1]cuda:0", primals_136: "f32[384, 512][512, 1]cuda:0", primals_137: "f32[384, 512][512, 1]cuda:0", primals_138: "f32[384, 512][512, 1]cuda:0", primals_139: "f32[512, 384][384, 1]cuda:0", primals_140: "f32[512][1]cuda:0", primals_141: "f32[384, 512][512, 1]cuda:0", primals_142: "f32[384, 512][512, 1]cuda:0", primals_143: "f32[384, 512][512, 1]cuda:0", primals_144: "f32[512, 384][384, 1]cuda:0", primals_145: "f32[512][1]cuda:0", primals_146: "f32[1024, 512][512, 1]cuda:0", primals_147: "f32[1024, 512][512, 1]cuda:0", primals_148: "f32[512, 1024][1024, 1]cuda:0", primals_149: "f32[512][1]cuda:0", primals_150: "f32[384, 512][512, 1]cuda:0", primals_151: "f32[384, 512][512, 1]cuda:0", primals_152: "f32[384, 512][512, 1]cuda:0", primals_153: "f32[512, 384][384, 1]cuda:0", primals_154: "f32[512][1]cuda:0", primals_155: "f32[384, 512][512, 1]cuda:0", primals_156: "f32[384, 512][512, 1]cuda:0", primals_157: "f32[384, 512][512, 1]cuda:0", primals_158: "f32[512, 384][384, 1]cuda:0", primals_159: "f32[512][1]cuda:0", primals_160: "f32[1024, 512][512, 1]cuda:0", primals_161: "f32[1024, 512][512, 1]cuda:0", primals_162: "f32[512, 1024][1024, 1]cuda:0", primals_163: "f32[512][1]cuda:0", primals_164: "f32[384, 512][512, 1]cuda:0", primals_165: "f32[384, 512][512, 1]cuda:0", primals_166: "f32[384, 512][512, 1]cuda:0", primals_167: "f32[512, 384][384, 1]cuda:0", primals_168: "f32[512][1]cuda:0", primals_169: "f32[384, 512][512, 1]cuda:0", primals_170: "f32[384, 512][512, 1]cuda:0", primals_171: "f32[384, 512][512, 1]cuda:0", primals_172: "f32[512, 384][384, 1]cuda:0", primals_173: "f32[512][1]cuda:0", primals_174: "f32[1024, 512][512, 1]cuda:0", primals_175: "f32[1024, 512][512, 1]cuda:0", primals_176: "f32[512, 1024][1024, 1]cuda:0", primals_177: "f32[512][1]cuda:0", primals_178: "f32[384, 512][512, 1]cuda:0", primals_179: "f32[384, 512][512, 1]cuda:0", primals_180: "f32[384, 512][512, 1]cuda:0", primals_181: "f32[512, 384][384, 1]cuda:0", primals_182: "f32[512][1]cuda:0", primals_183: "f32[384, 512][512, 1]cuda:0", primals_184: "f32[384, 512][512, 1]cuda:0", primals_185: "f32[384, 512][512, 1]cuda:0", primals_186: "f32[512, 384][384, 1]cuda:0", primals_187: "f32[512][1]cuda:0", primals_188: "f32[1024, 512][512, 1]cuda:0", primals_189: "f32[1024, 512][512, 1]cuda:0", primals_190: "f32[512, 1024][1024, 1]cuda:0", primals_191: "f32[512][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:680 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.embedding.default(primals_2, primals_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default: "f32[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([32, 1, 128, 128], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[84][1]cuda:0" = torch.ops.prims.inductor_seeds.default(84, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:732 in forward, code: hidden_states = self.dropout(inputs_embeds)
        inductor_lookup_seed_default: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_83: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_83, 0.1);  inductor_random_default_83 = None
        mul: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt, embedding)
        mul_1: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_1: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(mul_1, 2)
        mean: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_2: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_2: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1, rsqrt)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_3: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_3, mul_2);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_4, torch.bfloat16);  primals_4 = None
        convert_element_type_1: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        permute: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type, [1, 0]);  convert_element_type = None
        view_1: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1, [4096, 512]);  convert_element_type_1 = None
        mm: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_1, permute)
        view_2: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [32, 128, 384]);  mm = None
        view_3: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_2, [32, 128, -1, 64]);  view_2 = None
        permute_1: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_4: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_5, torch.bfloat16);  primals_5 = None
        permute_2: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_4, [1, 0]);  convert_element_type_4 = None
        mm_1: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_1, permute_2)
        view_5: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_1, [32, 128, 384]);  mm_1 = None
        view_6: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_5, [32, 128, -1, 64]);  view_5 = None
        permute_3: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_8: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_6, torch.bfloat16);  primals_6 = None
        permute_4: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_8, [1, 0]);  convert_element_type_8 = None
        mm_2: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_1, permute_4)
        view_8: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [32, 128, 384]);  mm_2 = None
        view_9: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_8, [32, 128, -1, 64]);  view_8 = None
        permute_5: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_6: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        expand_1: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_1, [32, 6, 128, 64]);  permute_1 = None
        clone: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_10: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone, [192, 128, 64]);  clone = None
        expand_2: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_6, [32, 6, 64, 128]);  permute_6 = None
        clone_1: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_11: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_1, [192, 64, 128]);  clone_1 = None
        bmm: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_10, view_11)
        view_12: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm, [32, 6, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:232 in compute_bias, code: context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None] + past_seen_tokens
        iota_4: "i64[128][1]cuda:0" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_3: "i64[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_4, 1)
        add_3: "i64[128, 1][1, 1]cuda:0" = torch.ops.aten.add.Tensor(unsqueeze_3, 0);  unsqueeze_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:233 in compute_bias, code: memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        unsqueeze_4: "i64[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_4, 0)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:234 in compute_bias, code: relative_position = memory_position - context_position  # shape (query_length, key_length)
        sub: "i64[128, 128][128, 1]cuda:0" = torch.ops.aten.sub.Tensor(unsqueeze_4, add_3);  unsqueeze_4 = add_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:205 in _relative_position_bucket, code: relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        gt_1: "b8[128, 128][128, 1]cuda:0" = torch.ops.aten.gt.Scalar(sub, 0)
        convert_element_type_14: "i64[128, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_1, torch.int64);  gt_1 = None
        mul_4: "i64[128, 128][128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_14, 16);  convert_element_type_14 = None
        add_4: "i64[128, 128][128, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_4, 0);  mul_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:206 in _relative_position_bucket, code: relative_position = torch.abs(relative_position)
        abs_1: "i64[128, 128][128, 1]cuda:0" = torch.ops.aten.abs.default(sub)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:213 in _relative_position_bucket, code: is_small = relative_position < max_exact
        lt: "b8[128, 128][128, 1]cuda:0" = torch.ops.aten.lt.Scalar(abs_1, 8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:217 in _relative_position_bucket, code: torch.log(relative_position.float() / max_exact)
        convert_element_type_15: "f32[128, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(abs_1, torch.float32)
        div: "f32[128, 128][128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_15, 8);  convert_element_type_15 = None
        log: "f32[128, 128][128, 1]cuda:0" = torch.ops.aten.log.default(div);  div = None
        div_1: "f32[128, 128][128, 1]cuda:0" = torch.ops.aten.div.Tensor(log, 2.772588722239781);  log = None
        mul_5: "f32[128, 128][128, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_1, 8);  div_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:220 in _relative_position_bucket, code: ).to(torch.long)
        convert_element_type_16: "i64[128, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_5, torch.int64);  mul_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:216 in _relative_position_bucket, code: relative_position_if_large = max_exact + (
        add_5: "i64[128, 128][128, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_16, 8);  convert_element_type_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:222 in _relative_position_bucket, code: relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        full: "i64[128, 128][128, 1]cuda:0" = torch.ops.aten.full.default([128, 128], 15, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:221 in _relative_position_bucket, code: relative_position_if_large = torch.min(
        minimum: "i64[128, 128][128, 1]cuda:0" = torch.ops.aten.minimum.default(add_5, full);  add_5 = full = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:225 in _relative_position_bucket, code: relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        where_1: "i64[128, 128][128, 1]cuda:0" = torch.ops.aten.where.self(lt, abs_1, minimum);  lt = abs_1 = minimum = None
        add_6: "i64[128, 128][128, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4, where_1);  add_4 = where_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:241 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        embedding_1: "f32[128, 128, 6][768, 6, 1]cuda:0" = torch.ops.aten.embedding.default(primals_7, add_6);  primals_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:242 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        permute_7: "f32[6, 128, 128][1, 768, 6]cuda:0" = torch.ops.aten.permute.default(embedding_1, [2, 0, 1])
        unsqueeze_5: "f32[1, 6, 128, 128][6, 1, 768, 6]cuda:0" = torch.ops.aten.unsqueeze.default(permute_7, 0);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:317 in forward, code: position_bias = position_bias + causal_mask
        add_7: "f32[32, 6, 128, 128][98304, 1, 768, 6]cuda:0" = torch.ops.aten.add.Tensor(unsqueeze_5, full_default);  unsqueeze_5 = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_8: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_12, add_7);  view_12 = None
        convert_element_type_17: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_8, torch.bfloat16);  add_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_18: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_17, torch.float32);  convert_element_type_17 = None
        amax: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_18, [-1], True)
        sub_1: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_18, amax);  convert_element_type_18 = None
        exp: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_2: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        convert_element_type_19: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_1: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_82: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 6, 128, 128], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        convert_element_type_default_79: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_82, torch.bfloat16);  inductor_random_default_82 = None
        gt_2: "b8[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_79, 0.1);  convert_element_type_default_79 = None
        mul_6: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_2, convert_element_type_19);  convert_element_type_19 = None
        mul_7: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_6, 1.1111111111111112);  mul_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_3: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(mul_7, [32, 6, 128, 128]);  mul_7 = None
        view_16: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_3, [192, 128, 128]);  expand_3 = None
        expand_4: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_5, [32, 6, 128, 64]);  permute_5 = None
        clone_2: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_17: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [192, 128, 64]);  clone_2 = None
        bmm_1: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_16, view_17)
        view_18: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_1, [32, 6, 128, 64]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_8: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        clone_3: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_19: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_3, [32, 128, -1]);  clone_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_22: "bf16[512, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_8, torch.bfloat16);  primals_8 = None
        permute_9: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_22, [1, 0]);  convert_element_type_22 = None
        view_20: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_19, [4096, 384]);  view_19 = None
        mm_3: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_20, permute_9)
        view_21: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_3, [32, 128, 512]);  mm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_2: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_81: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        convert_element_type_default_78: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_81, torch.bfloat16);  inductor_random_default_81 = None
        gt_3: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_78, 0.1);  convert_element_type_default_78 = None
        mul_8: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_3, view_21);  view_21 = None
        mul_9: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_8, 1.1111111111111112);  mul_8 = None
        add_9: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, mul_9);  mul_1 = mul_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_2: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_9, 2)
        mean_1: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_10: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        rsqrt_1: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_10: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_9, rsqrt_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_11: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_9, mul_10);  mul_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        convert_element_type_25: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_10, torch.bfloat16);  primals_10 = None
        convert_element_type_26: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_11, torch.bfloat16);  mul_11 = None
        permute_10: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_25, [1, 0]);  convert_element_type_25 = None
        view_22: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_26, [4096, 512]);  convert_element_type_26 = None
        mm_4: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_22, permute_10)
        view_23: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [32, 128, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_12: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_23, 0.5)
        convert_element_type_29: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_23, torch.float32)
        pow_3: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_29, 3.0);  convert_element_type_29 = None
        mul_13: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_11: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_23, mul_13);  view_23 = mul_13 = None
        mul_14: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_11, 0.7978845608028654);  add_11 = None
        tanh: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
        add_12: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_15: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_12, add_12);  mul_12 = add_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        convert_element_type_30: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_11, torch.bfloat16);  primals_11 = None
        permute_11: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_30, [1, 0]);  convert_element_type_30 = None
        mm_5: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_22, permute_11)
        view_25: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_5, [32, 128, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_16: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_15, view_25);  mul_15 = view_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:109 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_3: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_80: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 1024], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        gt_4: "b8[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_80, 0.1);  inductor_random_default_80 = None
        mul_17: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_4, mul_16);  mul_16 = None
        mul_18: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_17, 1.1111111111111112);  mul_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        convert_element_type_34: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_12, torch.bfloat16);  primals_12 = None
        convert_element_type_35: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_18, torch.bfloat16);  mul_18 = None
        permute_12: "bf16[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_34, [1, 0]);  convert_element_type_34 = None
        view_26: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_35, [4096, 1024]);  convert_element_type_35 = None
        mm_6: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_26, permute_12)
        view_27: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [32, 128, 512]);  mm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_4: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_random_default_79: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        convert_element_type_default_77: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_79, torch.bfloat16);  inductor_random_default_79 = None
        gt_5: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_77, 0.1);  convert_element_type_default_77 = None
        mul_19: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_5, view_27);  view_27 = None
        mul_20: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_19, 1.1111111111111112);  mul_19 = None
        add_13: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_9, mul_20);  mul_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_4: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_13, 2)
        mean_2: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_14: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
        rsqrt_2: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        mul_21: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_13, rsqrt_2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_22: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_13, mul_21);  mul_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_38: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
        convert_element_type_39: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_22, torch.bfloat16);  mul_22 = None
        permute_13: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_38, [1, 0]);  convert_element_type_38 = None
        view_28: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_39, [4096, 512]);  convert_element_type_39 = None
        mm_7: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_28, permute_13)
        view_29: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_7, [32, 128, 384]);  mm_7 = None
        view_30: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_29, [32, 128, -1, 64]);  view_29 = None
        permute_14: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_42: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16);  primals_15 = None
        permute_15: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_42, [1, 0]);  convert_element_type_42 = None
        mm_8: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_28, permute_15)
        view_32: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [32, 128, 384]);  mm_8 = None
        view_33: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_32, [32, 128, -1, 64]);  view_32 = None
        permute_16: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_46: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_16, torch.bfloat16);  primals_16 = None
        permute_17: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_46, [1, 0]);  convert_element_type_46 = None
        mm_9: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_28, permute_17)
        view_35: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_9, [32, 128, 384]);  mm_9 = None
        view_36: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_35, [32, 128, -1, 64]);  view_35 = None
        permute_18: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_19: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_16, [0, 1, 3, 2]);  permute_16 = None
        expand_5: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_14, [32, 6, 128, 64]);  permute_14 = None
        clone_4: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_37: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_4, [192, 128, 64]);  clone_4 = None
        expand_6: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_19, [32, 6, 64, 128]);  permute_19 = None
        clone_5: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_38: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_5, [192, 64, 128]);  clone_5 = None
        bmm_2: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_37, view_38)
        view_39: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_2, [32, 6, 128, 128]);  bmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_15: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_39, add_7);  view_39 = None
        convert_element_type_52: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_15, torch.bfloat16);  add_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_53: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_52, torch.float32)
        amax_1: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_53, [-1], True)
        sub_2: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_53, amax_1);  convert_element_type_53 = None
        exp_1: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_2: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_3: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = None
        convert_element_type_54: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_5: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_random_default_78: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 6, 128, 128], inductor_lookup_seed_default_5, 'rand');  inductor_lookup_seed_default_5 = None
        convert_element_type_default_76: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_78, torch.bfloat16);  inductor_random_default_78 = None
        gt_6: "b8[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_76, 0.1);  convert_element_type_default_76 = None
        mul_23: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_6, convert_element_type_54);  convert_element_type_54 = None
        mul_24: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_23, 1.1111111111111112);  mul_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_7: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(mul_24, [32, 6, 128, 128]);  mul_24 = None
        view_43: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_7, [192, 128, 128]);  expand_7 = None
        expand_8: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_18, [32, 6, 128, 64]);  permute_18 = None
        clone_6: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_44: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_6, [192, 128, 64]);  clone_6 = None
        bmm_3: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_43, view_44)
        view_45: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_3, [32, 6, 128, 64]);  bmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_20: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_45, [0, 2, 1, 3]);  view_45 = None
        clone_7: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_20, memory_format = torch.contiguous_format);  permute_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_46: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_7, [32, 128, -1]);  clone_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_57: "bf16[512, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_17, torch.bfloat16);  primals_17 = None
        permute_21: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_57, [1, 0]);  convert_element_type_57 = None
        view_47: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_46, [4096, 384]);  view_46 = None
        mm_10: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_47, permute_21)
        view_48: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [32, 128, 512]);  mm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_6: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6)
        inductor_random_default_77: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_6, 'rand');  inductor_lookup_seed_default_6 = None
        convert_element_type_default_75: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_77, torch.bfloat16);  inductor_random_default_77 = None
        gt_7: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_75, 0.1);  convert_element_type_default_75 = None
        mul_25: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_7, view_48);  view_48 = None
        mul_26: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_25, 1.1111111111111112);  mul_25 = None
        add_16: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_13, mul_26);  mul_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_5: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_16, 2)
        mean_3: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_17: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None
        rsqrt_3: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        mul_27: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_16, rsqrt_3)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_28: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_18, mul_27);  mul_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        convert_element_type_60: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_19, torch.bfloat16);  primals_19 = None
        convert_element_type_61: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_28, torch.bfloat16);  mul_28 = None
        permute_22: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_60, [1, 0]);  convert_element_type_60 = None
        view_49: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_61, [4096, 512]);  convert_element_type_61 = None
        mm_11: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_49, permute_22)
        view_50: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_11, [32, 128, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_29: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_50, 0.5)
        convert_element_type_64: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_50, torch.float32)
        pow_6: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_64, 3.0);  convert_element_type_64 = None
        mul_30: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_18: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_50, mul_30);  view_50 = mul_30 = None
        mul_31: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_18, 0.7978845608028654);  add_18 = None
        tanh_1: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.tanh.default(mul_31);  mul_31 = None
        add_19: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_32: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_29, add_19);  mul_29 = add_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        convert_element_type_65: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_20, torch.bfloat16);  primals_20 = None
        permute_23: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_65, [1, 0]);  convert_element_type_65 = None
        mm_12: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_49, permute_23)
        view_52: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_12, [32, 128, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_33: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_32, view_52);  mul_32 = view_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:109 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_7: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 7)
        inductor_random_default_76: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 1024], inductor_lookup_seed_default_7, 'rand');  inductor_lookup_seed_default_7 = None
        gt_8: "b8[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_76, 0.1);  inductor_random_default_76 = None
        mul_34: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_8, mul_33);  mul_33 = None
        mul_35: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_34, 1.1111111111111112);  mul_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        convert_element_type_69: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_21, torch.bfloat16);  primals_21 = None
        convert_element_type_70: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_35, torch.bfloat16);  mul_35 = None
        permute_24: "bf16[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_69, [1, 0]);  convert_element_type_69 = None
        view_53: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_70, [4096, 1024]);  convert_element_type_70 = None
        mm_13: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_53, permute_24)
        view_54: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_13, [32, 128, 512]);  mm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_8: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 8)
        inductor_random_default_75: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_8, 'rand');  inductor_lookup_seed_default_8 = None
        convert_element_type_default_74: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_75, torch.bfloat16);  inductor_random_default_75 = None
        gt_9: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_74, 0.1);  convert_element_type_default_74 = None
        mul_36: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_9, view_54);  view_54 = None
        mul_37: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_36, 1.1111111111111112);  mul_36 = None
        add_20: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_16, mul_37);  mul_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_7: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_20, 2)
        mean_4: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_21: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_4, 1e-06);  mean_4 = None
        rsqrt_4: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        mul_38: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_20, rsqrt_4)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_39: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_22, mul_38);  mul_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_73: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_23, torch.bfloat16);  primals_23 = None
        convert_element_type_74: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_39, torch.bfloat16);  mul_39 = None
        permute_25: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_73, [1, 0]);  convert_element_type_73 = None
        view_55: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_74, [4096, 512]);  convert_element_type_74 = None
        mm_14: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_55, permute_25)
        view_56: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_14, [32, 128, 384]);  mm_14 = None
        view_57: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_56, [32, 128, -1, 64]);  view_56 = None
        permute_26: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_57, [0, 2, 1, 3]);  view_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_77: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_24, torch.bfloat16);  primals_24 = None
        permute_27: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_77, [1, 0]);  convert_element_type_77 = None
        mm_15: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_55, permute_27)
        view_59: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_15, [32, 128, 384]);  mm_15 = None
        view_60: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_59, [32, 128, -1, 64]);  view_59 = None
        permute_28: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_81: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_25, torch.bfloat16);  primals_25 = None
        permute_29: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_81, [1, 0]);  convert_element_type_81 = None
        mm_16: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_55, permute_29)
        view_62: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_16, [32, 128, 384]);  mm_16 = None
        view_63: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_62, [32, 128, -1, 64]);  view_62 = None
        permute_30: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_63, [0, 2, 1, 3]);  view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_31: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_28, [0, 1, 3, 2]);  permute_28 = None
        expand_9: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_26, [32, 6, 128, 64]);  permute_26 = None
        clone_8: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_64: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_8, [192, 128, 64]);  clone_8 = None
        expand_10: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_31, [32, 6, 64, 128]);  permute_31 = None
        clone_9: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_65: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_9, [192, 64, 128]);  clone_9 = None
        bmm_4: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_64, view_65)
        view_66: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_4, [32, 6, 128, 128]);  bmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_22: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_66, add_7);  view_66 = None
        convert_element_type_87: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_22, torch.bfloat16);  add_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_88: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_87, torch.float32)
        amax_2: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_88, [-1], True)
        sub_3: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_88, amax_2);  convert_element_type_88 = None
        exp_2: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        sum_3: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_4: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = None
        convert_element_type_89: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_9: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 9)
        inductor_random_default_74: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 6, 128, 128], inductor_lookup_seed_default_9, 'rand');  inductor_lookup_seed_default_9 = None
        convert_element_type_default_73: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_74, torch.bfloat16);  inductor_random_default_74 = None
        gt_10: "b8[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_73, 0.1);  convert_element_type_default_73 = None
        mul_40: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_10, convert_element_type_89);  convert_element_type_89 = None
        mul_41: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_40, 1.1111111111111112);  mul_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_11: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(mul_41, [32, 6, 128, 128]);  mul_41 = None
        view_70: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_11, [192, 128, 128]);  expand_11 = None
        expand_12: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_30, [32, 6, 128, 64]);  permute_30 = None
        clone_10: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_71: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_10, [192, 128, 64]);  clone_10 = None
        bmm_5: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_70, view_71)
        view_72: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_5, [32, 6, 128, 64]);  bmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_32: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        clone_11: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_73: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_11, [32, 128, -1]);  clone_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_92: "bf16[512, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_26, torch.bfloat16);  primals_26 = None
        permute_33: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_92, [1, 0]);  convert_element_type_92 = None
        view_74: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_73, [4096, 384]);  view_73 = None
        mm_17: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_74, permute_33)
        view_75: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_17, [32, 128, 512]);  mm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_10: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 10)
        inductor_random_default_73: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_10, 'rand');  inductor_lookup_seed_default_10 = None
        convert_element_type_default_72: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_73, torch.bfloat16);  inductor_random_default_73 = None
        gt_11: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_72, 0.1);  convert_element_type_default_72 = None
        mul_42: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_11, view_75);  view_75 = None
        mul_43: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, 1.1111111111111112);  mul_42 = None
        add_23: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_20, mul_43);  mul_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_8: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_23, 2)
        mean_5: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_8, [-1], True);  pow_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_24: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_5, 1e-06);  mean_5 = None
        rsqrt_5: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        mul_44: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_23, rsqrt_5)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_45: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_27, mul_44);  mul_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        convert_element_type_95: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_28, torch.bfloat16);  primals_28 = None
        convert_element_type_96: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_45, torch.bfloat16);  mul_45 = None
        permute_34: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_95, [1, 0]);  convert_element_type_95 = None
        view_76: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_96, [4096, 512]);  convert_element_type_96 = None
        mm_18: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_76, permute_34)
        view_77: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_18, [32, 128, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_46: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_77, 0.5)
        convert_element_type_99: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_77, torch.float32)
        pow_9: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_99, 3.0);  convert_element_type_99 = None
        mul_47: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_25: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_77, mul_47);  view_77 = mul_47 = None
        mul_48: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_25, 0.7978845608028654);  add_25 = None
        tanh_2: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.tanh.default(mul_48);  mul_48 = None
        add_26: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_49: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_46, add_26);  mul_46 = add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        convert_element_type_100: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_29, torch.bfloat16);  primals_29 = None
        permute_35: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_100, [1, 0]);  convert_element_type_100 = None
        mm_19: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_76, permute_35)
        view_79: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_19, [32, 128, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_50: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_49, view_79);  mul_49 = view_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:109 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_11: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 11)
        inductor_random_default_72: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 1024], inductor_lookup_seed_default_11, 'rand');  inductor_lookup_seed_default_11 = None
        gt_12: "b8[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_72, 0.1);  inductor_random_default_72 = None
        mul_51: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_12, mul_50);  mul_50 = None
        mul_52: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_51, 1.1111111111111112);  mul_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        convert_element_type_104: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_30, torch.bfloat16);  primals_30 = None
        convert_element_type_105: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_52, torch.bfloat16);  mul_52 = None
        permute_36: "bf16[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_104, [1, 0]);  convert_element_type_104 = None
        view_80: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_105, [4096, 1024]);  convert_element_type_105 = None
        mm_20: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_80, permute_36)
        view_81: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_20, [32, 128, 512]);  mm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_12: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 12)
        inductor_random_default_71: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_12, 'rand');  inductor_lookup_seed_default_12 = None
        convert_element_type_default_71: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_71, torch.bfloat16);  inductor_random_default_71 = None
        gt_13: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_71, 0.1);  convert_element_type_default_71 = None
        mul_53: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_13, view_81);  view_81 = None
        mul_54: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_53, 1.1111111111111112);  mul_53 = None
        add_27: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_23, mul_54);  mul_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_10: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_27, 2)
        mean_6: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_28: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_6, 1e-06);  mean_6 = None
        rsqrt_6: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_55: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_27, rsqrt_6)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_56: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_31, mul_55);  mul_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_108: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_32, torch.bfloat16);  primals_32 = None
        convert_element_type_109: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_56, torch.bfloat16);  mul_56 = None
        permute_37: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_108, [1, 0]);  convert_element_type_108 = None
        view_82: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_109, [4096, 512]);  convert_element_type_109 = None
        mm_21: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_82, permute_37)
        view_83: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_21, [32, 128, 384]);  mm_21 = None
        view_84: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_83, [32, 128, -1, 64]);  view_83 = None
        permute_38: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_112: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_33, torch.bfloat16);  primals_33 = None
        permute_39: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_112, [1, 0]);  convert_element_type_112 = None
        mm_22: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_82, permute_39)
        view_86: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_22, [32, 128, 384]);  mm_22 = None
        view_87: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_86, [32, 128, -1, 64]);  view_86 = None
        permute_40: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_116: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_34, torch.bfloat16);  primals_34 = None
        permute_41: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_116, [1, 0]);  convert_element_type_116 = None
        mm_23: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_82, permute_41)
        view_89: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_23, [32, 128, 384]);  mm_23 = None
        view_90: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_89, [32, 128, -1, 64]);  view_89 = None
        permute_42: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_43: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_40, [0, 1, 3, 2]);  permute_40 = None
        expand_13: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_38, [32, 6, 128, 64]);  permute_38 = None
        clone_12: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_91: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_12, [192, 128, 64]);  clone_12 = None
        expand_14: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_43, [32, 6, 64, 128]);  permute_43 = None
        clone_13: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_92: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_13, [192, 64, 128]);  clone_13 = None
        bmm_6: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_91, view_92)
        view_93: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_6, [32, 6, 128, 128]);  bmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_29: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_93, add_7);  view_93 = None
        convert_element_type_122: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_29, torch.bfloat16);  add_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_123: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_122, torch.float32)
        amax_3: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_123, [-1], True)
        sub_4: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_123, amax_3);  convert_element_type_123 = None
        exp_3: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_4: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_5: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = None
        convert_element_type_124: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_13: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 13)
        inductor_random_default_70: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 6, 128, 128], inductor_lookup_seed_default_13, 'rand');  inductor_lookup_seed_default_13 = None
        convert_element_type_default_70: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_70, torch.bfloat16);  inductor_random_default_70 = None
        gt_14: "b8[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_70, 0.1);  convert_element_type_default_70 = None
        mul_57: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_14, convert_element_type_124);  convert_element_type_124 = None
        mul_58: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_57, 1.1111111111111112);  mul_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_15: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(mul_58, [32, 6, 128, 128]);  mul_58 = None
        view_97: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_15, [192, 128, 128]);  expand_15 = None
        expand_16: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_42, [32, 6, 128, 64]);  permute_42 = None
        clone_14: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_98: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_14, [192, 128, 64]);  clone_14 = None
        bmm_7: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_97, view_98)
        view_99: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_7, [32, 6, 128, 64]);  bmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_44: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None
        clone_15: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_100: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_15, [32, 128, -1]);  clone_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_127: "bf16[512, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_35, torch.bfloat16);  primals_35 = None
        permute_45: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_127, [1, 0]);  convert_element_type_127 = None
        view_101: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_100, [4096, 384]);  view_100 = None
        mm_24: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_101, permute_45)
        view_102: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_24, [32, 128, 512]);  mm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_14: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 14)
        inductor_random_default_69: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_14, 'rand');  inductor_lookup_seed_default_14 = None
        convert_element_type_default_69: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_69, torch.bfloat16);  inductor_random_default_69 = None
        gt_15: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_69, 0.1);  convert_element_type_default_69 = None
        mul_59: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_15, view_102);  view_102 = None
        mul_60: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_59, 1.1111111111111112);  mul_59 = None
        add_30: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_27, mul_60);  mul_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_11: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_30, 2)
        mean_7: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_31: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_7, 1e-06);  mean_7 = None
        rsqrt_7: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        mul_61: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_30, rsqrt_7)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_62: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_36, mul_61);  mul_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        convert_element_type_130: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_37, torch.bfloat16);  primals_37 = None
        convert_element_type_131: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_62, torch.bfloat16);  mul_62 = None
        permute_46: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_130, [1, 0]);  convert_element_type_130 = None
        view_103: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_131, [4096, 512]);  convert_element_type_131 = None
        mm_25: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_103, permute_46)
        view_104: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_25, [32, 128, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_63: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_104, 0.5)
        convert_element_type_134: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_104, torch.float32)
        pow_12: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_134, 3.0);  convert_element_type_134 = None
        mul_64: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_32: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_104, mul_64);  view_104 = mul_64 = None
        mul_65: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_32, 0.7978845608028654);  add_32 = None
        tanh_3: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.tanh.default(mul_65);  mul_65 = None
        add_33: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_66: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_63, add_33);  mul_63 = add_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        convert_element_type_135: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_38, torch.bfloat16);  primals_38 = None
        permute_47: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_135, [1, 0]);  convert_element_type_135 = None
        mm_26: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_103, permute_47)
        view_106: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_26, [32, 128, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_67: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, view_106);  mul_66 = view_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:109 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_15: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 15)
        inductor_random_default_68: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 1024], inductor_lookup_seed_default_15, 'rand');  inductor_lookup_seed_default_15 = None
        gt_16: "b8[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_68, 0.1);  inductor_random_default_68 = None
        mul_68: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_16, mul_67);  mul_67 = None
        mul_69: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, 1.1111111111111112);  mul_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        convert_element_type_139: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_39, torch.bfloat16);  primals_39 = None
        convert_element_type_140: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_69, torch.bfloat16);  mul_69 = None
        permute_48: "bf16[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_139, [1, 0]);  convert_element_type_139 = None
        view_107: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_140, [4096, 1024]);  convert_element_type_140 = None
        mm_27: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_107, permute_48)
        view_108: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_27, [32, 128, 512]);  mm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_16: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 16)
        inductor_random_default_67: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_16, 'rand');  inductor_lookup_seed_default_16 = None
        convert_element_type_default_68: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_67, torch.bfloat16);  inductor_random_default_67 = None
        gt_17: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_68, 0.1);  convert_element_type_default_68 = None
        mul_70: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_17, view_108);  view_108 = None
        mul_71: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, 1.1111111111111112);  mul_70 = None
        add_34: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_30, mul_71);  mul_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_13: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_34, 2)
        mean_8: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_35: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_8, 1e-06);  mean_8 = None
        rsqrt_8: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        mul_72: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_34, rsqrt_8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_73: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_40, mul_72);  mul_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_143: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_41, torch.bfloat16);  primals_41 = None
        convert_element_type_144: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_73, torch.bfloat16);  mul_73 = None
        permute_49: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_143, [1, 0]);  convert_element_type_143 = None
        view_109: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_144, [4096, 512]);  convert_element_type_144 = None
        mm_28: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_109, permute_49)
        view_110: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_28, [32, 128, 384]);  mm_28 = None
        view_111: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_110, [32, 128, -1, 64]);  view_110 = None
        permute_50: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_111, [0, 2, 1, 3]);  view_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_147: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_42, torch.bfloat16);  primals_42 = None
        permute_51: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_147, [1, 0]);  convert_element_type_147 = None
        mm_29: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_109, permute_51)
        view_113: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_29, [32, 128, 384]);  mm_29 = None
        view_114: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_113, [32, 128, -1, 64]);  view_113 = None
        permute_52: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_151: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_43, torch.bfloat16);  primals_43 = None
        permute_53: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_151, [1, 0]);  convert_element_type_151 = None
        mm_30: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_109, permute_53)
        view_116: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_30, [32, 128, 384]);  mm_30 = None
        view_117: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_116, [32, 128, -1, 64]);  view_116 = None
        permute_54: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_55: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_52, [0, 1, 3, 2]);  permute_52 = None
        expand_17: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_50, [32, 6, 128, 64]);  permute_50 = None
        clone_16: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_118: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_16, [192, 128, 64]);  clone_16 = None
        expand_18: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_55, [32, 6, 64, 128]);  permute_55 = None
        clone_17: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_119: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_17, [192, 64, 128]);  clone_17 = None
        bmm_8: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_118, view_119)
        view_120: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_8, [32, 6, 128, 128]);  bmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_36: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_120, add_7);  view_120 = None
        convert_element_type_157: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_36, torch.bfloat16);  add_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_158: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_157, torch.float32)
        amax_4: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_158, [-1], True)
        sub_5: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_158, amax_4);  convert_element_type_158 = None
        exp_4: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        sum_5: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_6: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = None
        convert_element_type_159: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_17: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 17)
        inductor_random_default_66: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 6, 128, 128], inductor_lookup_seed_default_17, 'rand');  inductor_lookup_seed_default_17 = None
        convert_element_type_default_67: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_66, torch.bfloat16);  inductor_random_default_66 = None
        gt_18: "b8[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_67, 0.1);  convert_element_type_default_67 = None
        mul_74: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_18, convert_element_type_159);  convert_element_type_159 = None
        mul_75: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_74, 1.1111111111111112);  mul_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_19: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(mul_75, [32, 6, 128, 128]);  mul_75 = None
        view_124: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_19, [192, 128, 128]);  expand_19 = None
        expand_20: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_54, [32, 6, 128, 64]);  permute_54 = None
        clone_18: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_125: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_18, [192, 128, 64]);  clone_18 = None
        bmm_9: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_124, view_125)
        view_126: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_9, [32, 6, 128, 64]);  bmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_56: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
        clone_19: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_127: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_19, [32, 128, -1]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_162: "bf16[512, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16);  primals_44 = None
        permute_57: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_162, [1, 0]);  convert_element_type_162 = None
        view_128: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_127, [4096, 384]);  view_127 = None
        mm_31: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_128, permute_57)
        view_129: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_31, [32, 128, 512]);  mm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_18: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 18)
        inductor_random_default_65: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_18, 'rand');  inductor_lookup_seed_default_18 = None
        convert_element_type_default_66: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_65, torch.bfloat16);  inductor_random_default_65 = None
        gt_19: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_66, 0.1);  convert_element_type_default_66 = None
        mul_76: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_19, view_129);  view_129 = None
        mul_77: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_76, 1.1111111111111112);  mul_76 = None
        add_37: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_34, mul_77);  mul_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_14: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_37, 2)
        mean_9: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_14, [-1], True);  pow_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_38: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_9, 1e-06);  mean_9 = None
        rsqrt_9: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        mul_78: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_37, rsqrt_9)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_79: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_45, mul_78);  mul_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        convert_element_type_165: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_46, torch.bfloat16);  primals_46 = None
        convert_element_type_166: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_79, torch.bfloat16);  mul_79 = None
        permute_58: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_165, [1, 0]);  convert_element_type_165 = None
        view_130: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_166, [4096, 512]);  convert_element_type_166 = None
        mm_32: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_130, permute_58)
        view_131: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_32, [32, 128, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_80: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_131, 0.5)
        convert_element_type_169: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_131, torch.float32)
        pow_15: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_169, 3.0);  convert_element_type_169 = None
        mul_81: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
        add_39: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_131, mul_81);  view_131 = mul_81 = None
        mul_82: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_39, 0.7978845608028654);  add_39 = None
        tanh_4: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.tanh.default(mul_82);  mul_82 = None
        add_40: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_83: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_80, add_40);  mul_80 = add_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        convert_element_type_170: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_47, torch.bfloat16);  primals_47 = None
        permute_59: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_170, [1, 0]);  convert_element_type_170 = None
        mm_33: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_130, permute_59)
        view_133: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_33, [32, 128, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_84: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_83, view_133);  mul_83 = view_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:109 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_19: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 19)
        inductor_random_default_64: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 1024], inductor_lookup_seed_default_19, 'rand');  inductor_lookup_seed_default_19 = None
        gt_20: "b8[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_64, 0.1);  inductor_random_default_64 = None
        mul_85: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_20, mul_84);  mul_84 = None
        mul_86: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_85, 1.1111111111111112);  mul_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        convert_element_type_174: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_48, torch.bfloat16);  primals_48 = None
        convert_element_type_175: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_86, torch.bfloat16);  mul_86 = None
        permute_60: "bf16[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_174, [1, 0]);  convert_element_type_174 = None
        view_134: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_175, [4096, 1024]);  convert_element_type_175 = None
        mm_34: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_134, permute_60)
        view_135: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_34, [32, 128, 512]);  mm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_20: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 20)
        inductor_random_default_63: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_20, 'rand');  inductor_lookup_seed_default_20 = None
        convert_element_type_default_65: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_63, torch.bfloat16);  inductor_random_default_63 = None
        gt_21: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_65, 0.1);  convert_element_type_default_65 = None
        mul_87: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_21, view_135);  view_135 = None
        mul_88: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_87, 1.1111111111111112);  mul_87 = None
        add_41: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_37, mul_88);  mul_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_16: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_41, 2)
        mean_10: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_42: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_10, 1e-06);  mean_10 = None
        rsqrt_10: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_89: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_41, rsqrt_10)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_90: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_49, mul_89);  mul_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_178: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_50, torch.bfloat16);  primals_50 = None
        convert_element_type_179: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_90, torch.bfloat16);  mul_90 = None
        permute_61: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_178, [1, 0]);  convert_element_type_178 = None
        view_136: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_179, [4096, 512]);  convert_element_type_179 = None
        mm_35: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_136, permute_61)
        view_137: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_35, [32, 128, 384]);  mm_35 = None
        view_138: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_137, [32, 128, -1, 64]);  view_137 = None
        permute_62: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_182: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_51, torch.bfloat16);  primals_51 = None
        permute_63: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_182, [1, 0]);  convert_element_type_182 = None
        mm_36: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_136, permute_63)
        view_140: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_36, [32, 128, 384]);  mm_36 = None
        view_141: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_140, [32, 128, -1, 64]);  view_140 = None
        permute_64: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_141, [0, 2, 1, 3]);  view_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_186: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_52, torch.bfloat16);  primals_52 = None
        permute_65: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_186, [1, 0]);  convert_element_type_186 = None
        mm_37: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_136, permute_65)
        view_143: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_37, [32, 128, 384]);  mm_37 = None
        view_144: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_143, [32, 128, -1, 64]);  view_143 = None
        permute_66: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_144, [0, 2, 1, 3]);  view_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_67: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_64, [0, 1, 3, 2]);  permute_64 = None
        expand_21: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_62, [32, 6, 128, 64]);  permute_62 = None
        clone_20: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_145: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_20, [192, 128, 64]);  clone_20 = None
        expand_22: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_67, [32, 6, 64, 128]);  permute_67 = None
        clone_21: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_146: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_21, [192, 64, 128]);  clone_21 = None
        bmm_10: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_145, view_146)
        view_147: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_10, [32, 6, 128, 128]);  bmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_43: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_147, add_7);  view_147 = None
        convert_element_type_192: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_43, torch.bfloat16);  add_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_193: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_192, torch.float32)
        amax_5: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_193, [-1], True)
        sub_6: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_193, amax_5);  convert_element_type_193 = None
        exp_5: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_6: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_7: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = None
        convert_element_type_194: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_21: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 21)
        inductor_random_default_62: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 6, 128, 128], inductor_lookup_seed_default_21, 'rand');  inductor_lookup_seed_default_21 = None
        convert_element_type_default_64: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_62, torch.bfloat16);  inductor_random_default_62 = None
        gt_22: "b8[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_64, 0.1);  convert_element_type_default_64 = None
        mul_91: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_22, convert_element_type_194);  convert_element_type_194 = None
        mul_92: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_91, 1.1111111111111112);  mul_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_23: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(mul_92, [32, 6, 128, 128]);  mul_92 = None
        view_151: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_23, [192, 128, 128]);  expand_23 = None
        expand_24: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_66, [32, 6, 128, 64]);  permute_66 = None
        clone_22: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_152: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_22, [192, 128, 64]);  clone_22 = None
        bmm_11: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_151, view_152)
        view_153: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_11, [32, 6, 128, 64]);  bmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_68: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_153, [0, 2, 1, 3]);  view_153 = None
        clone_23: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_154: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_23, [32, 128, -1]);  clone_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_197: "bf16[512, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_53, torch.bfloat16);  primals_53 = None
        permute_69: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_197, [1, 0]);  convert_element_type_197 = None
        view_155: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_154, [4096, 384]);  view_154 = None
        mm_38: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_155, permute_69)
        view_156: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_38, [32, 128, 512]);  mm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_22: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 22)
        inductor_random_default_61: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_22, 'rand');  inductor_lookup_seed_default_22 = None
        convert_element_type_default_63: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_61, torch.bfloat16);  inductor_random_default_61 = None
        gt_23: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_63, 0.1);  convert_element_type_default_63 = None
        mul_93: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_23, view_156);  view_156 = None
        mul_94: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_93, 1.1111111111111112);  mul_93 = None
        add_44: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_41, mul_94);  mul_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_17: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_44, 2)
        mean_11: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_45: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_11, 1e-06);  mean_11 = None
        rsqrt_11: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        mul_95: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_44, rsqrt_11)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_96: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_54, mul_95);  mul_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        convert_element_type_200: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_55, torch.bfloat16);  primals_55 = None
        convert_element_type_201: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_96, torch.bfloat16);  mul_96 = None
        permute_70: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_200, [1, 0]);  convert_element_type_200 = None
        view_157: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_201, [4096, 512]);  convert_element_type_201 = None
        mm_39: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_157, permute_70)
        view_158: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_39, [32, 128, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_97: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_158, 0.5)
        convert_element_type_204: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_158, torch.float32)
        pow_18: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_204, 3.0);  convert_element_type_204 = None
        mul_98: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_18, 0.044715);  pow_18 = None
        add_46: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_158, mul_98);  view_158 = mul_98 = None
        mul_99: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_46, 0.7978845608028654);  add_46 = None
        tanh_5: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.tanh.default(mul_99);  mul_99 = None
        add_47: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_100: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_97, add_47);  mul_97 = add_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        convert_element_type_205: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_56, torch.bfloat16);  primals_56 = None
        permute_71: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_205, [1, 0]);  convert_element_type_205 = None
        mm_40: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_157, permute_71)
        view_160: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_40, [32, 128, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_101: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_100, view_160);  mul_100 = view_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:109 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_23: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 23)
        inductor_random_default_60: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 1024], inductor_lookup_seed_default_23, 'rand');  inductor_lookup_seed_default_23 = None
        gt_24: "b8[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_60, 0.1);  inductor_random_default_60 = None
        mul_102: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_24, mul_101);  mul_101 = None
        mul_103: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_102, 1.1111111111111112);  mul_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        convert_element_type_209: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_57, torch.bfloat16);  primals_57 = None
        convert_element_type_210: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_103, torch.bfloat16);  mul_103 = None
        permute_72: "bf16[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_209, [1, 0]);  convert_element_type_209 = None
        view_161: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_210, [4096, 1024]);  convert_element_type_210 = None
        mm_41: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_161, permute_72)
        view_162: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_41, [32, 128, 512]);  mm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_24: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 24)
        inductor_random_default_59: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_24, 'rand');  inductor_lookup_seed_default_24 = None
        convert_element_type_default_62: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_59, torch.bfloat16);  inductor_random_default_59 = None
        gt_25: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_62, 0.1);  convert_element_type_default_62 = None
        mul_104: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_25, view_162);  view_162 = None
        mul_105: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_104, 1.1111111111111112);  mul_104 = None
        add_48: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_44, mul_105);  mul_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_19: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_48, 2)
        mean_12: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_49: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_12, 1e-06);  mean_12 = None
        rsqrt_12: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        mul_106: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_48, rsqrt_12)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_107: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_58, mul_106);  mul_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_213: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_59, torch.bfloat16);  primals_59 = None
        convert_element_type_214: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_107, torch.bfloat16);  mul_107 = None
        permute_73: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_213, [1, 0]);  convert_element_type_213 = None
        view_163: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_214, [4096, 512]);  convert_element_type_214 = None
        mm_42: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_163, permute_73)
        view_164: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_42, [32, 128, 384]);  mm_42 = None
        view_165: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_164, [32, 128, -1, 64]);  view_164 = None
        permute_74: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_217: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_60, torch.bfloat16);  primals_60 = None
        permute_75: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_217, [1, 0]);  convert_element_type_217 = None
        mm_43: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_163, permute_75)
        view_167: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_43, [32, 128, 384]);  mm_43 = None
        view_168: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_167, [32, 128, -1, 64]);  view_167 = None
        permute_76: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_221: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_61, torch.bfloat16);  primals_61 = None
        permute_77: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_221, [1, 0]);  convert_element_type_221 = None
        mm_44: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_163, permute_77)
        view_170: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_44, [32, 128, 384]);  mm_44 = None
        view_171: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_170, [32, 128, -1, 64]);  view_170 = None
        permute_78: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_171, [0, 2, 1, 3]);  view_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_79: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_76, [0, 1, 3, 2]);  permute_76 = None
        expand_25: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_74, [32, 6, 128, 64]);  permute_74 = None
        clone_24: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_172: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_24, [192, 128, 64]);  clone_24 = None
        expand_26: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_79, [32, 6, 64, 128]);  permute_79 = None
        clone_25: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_173: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_25, [192, 64, 128]);  clone_25 = None
        bmm_12: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_172, view_173)
        view_174: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_12, [32, 6, 128, 128]);  bmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_50: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_174, add_7);  view_174 = None
        convert_element_type_227: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_50, torch.bfloat16);  add_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_228: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_227, torch.float32)
        amax_6: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_228, [-1], True)
        sub_7: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_228, amax_6);  convert_element_type_228 = None
        exp_6: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_7: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_8: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = None
        convert_element_type_229: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_25: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 25)
        inductor_random_default_58: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 6, 128, 128], inductor_lookup_seed_default_25, 'rand');  inductor_lookup_seed_default_25 = None
        convert_element_type_default_61: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_58, torch.bfloat16);  inductor_random_default_58 = None
        gt_26: "b8[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_61, 0.1);  convert_element_type_default_61 = None
        mul_108: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_26, convert_element_type_229);  convert_element_type_229 = None
        mul_109: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_108, 1.1111111111111112);  mul_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_27: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(mul_109, [32, 6, 128, 128]);  mul_109 = None
        view_178: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_27, [192, 128, 128]);  expand_27 = None
        expand_28: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_78, [32, 6, 128, 64]);  permute_78 = None
        clone_26: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_179: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_26, [192, 128, 64]);  clone_26 = None
        bmm_13: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_178, view_179)
        view_180: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_13, [32, 6, 128, 64]);  bmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_80: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_180, [0, 2, 1, 3]);  view_180 = None
        clone_27: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_181: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_27, [32, 128, -1]);  clone_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_232: "bf16[512, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_62, torch.bfloat16);  primals_62 = None
        permute_81: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_232, [1, 0]);  convert_element_type_232 = None
        view_182: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_181, [4096, 384]);  view_181 = None
        mm_45: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_182, permute_81)
        view_183: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_45, [32, 128, 512]);  mm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_26: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 26)
        inductor_random_default_57: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_26, 'rand');  inductor_lookup_seed_default_26 = None
        convert_element_type_default_60: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_57, torch.bfloat16);  inductor_random_default_57 = None
        gt_27: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_60, 0.1);  convert_element_type_default_60 = None
        mul_110: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_27, view_183);  view_183 = None
        mul_111: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_110, 1.1111111111111112);  mul_110 = None
        add_51: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_48, mul_111);  mul_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_20: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_51, 2)
        mean_13: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_20, [-1], True);  pow_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_52: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_13, 1e-06);  mean_13 = None
        rsqrt_13: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_112: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_51, rsqrt_13)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_113: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_63, mul_112);  mul_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        convert_element_type_235: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_64, torch.bfloat16);  primals_64 = None
        convert_element_type_236: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_113, torch.bfloat16);  mul_113 = None
        permute_82: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_235, [1, 0]);  convert_element_type_235 = None
        view_184: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_236, [4096, 512]);  convert_element_type_236 = None
        mm_46: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_184, permute_82)
        view_185: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_46, [32, 128, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_114: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_185, 0.5)
        convert_element_type_239: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_185, torch.float32)
        pow_21: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_239, 3.0);  convert_element_type_239 = None
        mul_115: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
        add_53: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_185, mul_115);  view_185 = mul_115 = None
        mul_116: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_53, 0.7978845608028654);  add_53 = None
        tanh_6: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.tanh.default(mul_116);  mul_116 = None
        add_54: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_117: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_114, add_54);  mul_114 = add_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        convert_element_type_240: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_65, torch.bfloat16);  primals_65 = None
        permute_83: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_240, [1, 0]);  convert_element_type_240 = None
        mm_47: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_184, permute_83)
        view_187: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_47, [32, 128, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_118: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_117, view_187);  mul_117 = view_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:109 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_27: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 27)
        inductor_random_default_56: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 1024], inductor_lookup_seed_default_27, 'rand');  inductor_lookup_seed_default_27 = None
        gt_28: "b8[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_56, 0.1);  inductor_random_default_56 = None
        mul_119: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_28, mul_118);  mul_118 = None
        mul_120: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_119, 1.1111111111111112);  mul_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        convert_element_type_244: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_66, torch.bfloat16);  primals_66 = None
        convert_element_type_245: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_120, torch.bfloat16);  mul_120 = None
        permute_84: "bf16[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_244, [1, 0]);  convert_element_type_244 = None
        view_188: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_245, [4096, 1024]);  convert_element_type_245 = None
        mm_48: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_188, permute_84)
        view_189: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_48, [32, 128, 512]);  mm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_28: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 28)
        inductor_random_default_55: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_28, 'rand');  inductor_lookup_seed_default_28 = None
        convert_element_type_default_59: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_55, torch.bfloat16);  inductor_random_default_55 = None
        gt_29: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_59, 0.1);  convert_element_type_default_59 = None
        mul_121: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_29, view_189);  view_189 = None
        mul_122: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_121, 1.1111111111111112);  mul_121 = None
        add_55: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_51, mul_122);  mul_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_22: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_55, 2)
        mean_14: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_22, [-1], True);  pow_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_56: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_14, 1e-06);  mean_14 = None
        rsqrt_14: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        mul_123: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_55, rsqrt_14)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_124: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_67, mul_123);  mul_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_248: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_68, torch.bfloat16);  primals_68 = None
        convert_element_type_249: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_124, torch.bfloat16);  mul_124 = None
        permute_85: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_248, [1, 0]);  convert_element_type_248 = None
        view_190: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_249, [4096, 512]);  convert_element_type_249 = None
        mm_49: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_190, permute_85)
        view_191: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_49, [32, 128, 384]);  mm_49 = None
        view_192: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_191, [32, 128, -1, 64]);  view_191 = None
        permute_86: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_252: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_69, torch.bfloat16);  primals_69 = None
        permute_87: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_252, [1, 0]);  convert_element_type_252 = None
        mm_50: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_190, permute_87)
        view_194: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_50, [32, 128, 384]);  mm_50 = None
        view_195: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_194, [32, 128, -1, 64]);  view_194 = None
        permute_88: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_195, [0, 2, 1, 3]);  view_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_256: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_70, torch.bfloat16);  primals_70 = None
        permute_89: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_256, [1, 0]);  convert_element_type_256 = None
        mm_51: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_190, permute_89)
        view_197: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_51, [32, 128, 384]);  mm_51 = None
        view_198: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_197, [32, 128, -1, 64]);  view_197 = None
        permute_90: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_198, [0, 2, 1, 3]);  view_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_91: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_88, [0, 1, 3, 2]);  permute_88 = None
        expand_29: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_86, [32, 6, 128, 64]);  permute_86 = None
        clone_28: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_199: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_28, [192, 128, 64]);  clone_28 = None
        expand_30: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_91, [32, 6, 64, 128]);  permute_91 = None
        clone_29: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_200: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_29, [192, 64, 128]);  clone_29 = None
        bmm_14: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_199, view_200)
        view_201: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_14, [32, 6, 128, 128]);  bmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_57: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_201, add_7);  view_201 = add_7 = None
        convert_element_type_262: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_57, torch.bfloat16);  add_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_263: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_262, torch.float32)
        amax_7: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_263, [-1], True)
        sub_8: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_263, amax_7);  convert_element_type_263 = None
        exp_7: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
        sum_8: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_9: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = None
        convert_element_type_264: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_29: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 29)
        inductor_random_default_54: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 6, 128, 128], inductor_lookup_seed_default_29, 'rand');  inductor_lookup_seed_default_29 = None
        convert_element_type_default_58: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_54, torch.bfloat16);  inductor_random_default_54 = None
        gt_30: "b8[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_58, 0.1);  convert_element_type_default_58 = None
        mul_125: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_30, convert_element_type_264);  convert_element_type_264 = None
        mul_126: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_125, 1.1111111111111112);  mul_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_31: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(mul_126, [32, 6, 128, 128]);  mul_126 = None
        view_205: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_31, [192, 128, 128]);  expand_31 = None
        expand_32: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_90, [32, 6, 128, 64]);  permute_90 = None
        clone_30: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_206: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_30, [192, 128, 64]);  clone_30 = None
        bmm_15: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_205, view_206)
        view_207: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_15, [32, 6, 128, 64]);  bmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_92: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None
        clone_31: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_208: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_31, [32, 128, -1]);  clone_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_267: "bf16[512, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_71, torch.bfloat16);  primals_71 = None
        permute_93: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_267, [1, 0]);  convert_element_type_267 = None
        view_209: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_208, [4096, 384]);  view_208 = None
        mm_52: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_209, permute_93)
        view_210: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_52, [32, 128, 512]);  mm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_30: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 30)
        inductor_random_default_53: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_30, 'rand');  inductor_lookup_seed_default_30 = None
        convert_element_type_default_57: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_53, torch.bfloat16);  inductor_random_default_53 = None
        gt_31: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_57, 0.1);  convert_element_type_default_57 = None
        mul_127: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_31, view_210);  view_210 = None
        mul_128: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_127, 1.1111111111111112);  mul_127 = None
        add_58: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_55, mul_128);  mul_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_23: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_58, 2)
        mean_15: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_23, [-1], True);  pow_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_59: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_15, 1e-06);  mean_15 = None
        rsqrt_15: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
        mul_129: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_58, rsqrt_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_130: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_72, mul_129);  mul_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        convert_element_type_270: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_73, torch.bfloat16);  primals_73 = None
        convert_element_type_271: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_130, torch.bfloat16);  mul_130 = None
        permute_94: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_270, [1, 0]);  convert_element_type_270 = None
        view_211: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_271, [4096, 512]);  convert_element_type_271 = None
        mm_53: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_211, permute_94)
        view_212: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_53, [32, 128, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_131: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_212, 0.5)
        convert_element_type_274: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_212, torch.float32)
        pow_24: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_274, 3.0);  convert_element_type_274 = None
        mul_132: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_24, 0.044715);  pow_24 = None
        add_60: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_212, mul_132);  view_212 = mul_132 = None
        mul_133: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_60, 0.7978845608028654);  add_60 = None
        tanh_7: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.tanh.default(mul_133);  mul_133 = None
        add_61: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_134: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_131, add_61);  mul_131 = add_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        convert_element_type_275: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_74, torch.bfloat16);  primals_74 = None
        permute_95: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_275, [1, 0]);  convert_element_type_275 = None
        mm_54: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_211, permute_95)
        view_214: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_54, [32, 128, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_135: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_134, view_214);  mul_134 = view_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:109 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_31: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 31)
        inductor_random_default_52: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 1024], inductor_lookup_seed_default_31, 'rand');  inductor_lookup_seed_default_31 = None
        gt_32: "b8[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_52, 0.1);  inductor_random_default_52 = None
        mul_136: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_32, mul_135);  mul_135 = None
        mul_137: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_136, 1.1111111111111112);  mul_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        convert_element_type_279: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_75, torch.bfloat16);  primals_75 = None
        convert_element_type_280: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_137, torch.bfloat16);  mul_137 = None
        permute_96: "bf16[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_279, [1, 0]);  convert_element_type_279 = None
        view_215: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_280, [4096, 1024]);  convert_element_type_280 = None
        mm_55: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_215, permute_96)
        view_216: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_55, [32, 128, 512]);  mm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_32: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 32)
        inductor_random_default_51: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_32, 'rand');  inductor_lookup_seed_default_32 = None
        convert_element_type_default_56: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_51, torch.bfloat16);  inductor_random_default_51 = None
        gt_33: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_56, 0.1);  convert_element_type_default_56 = None
        mul_138: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_33, view_216);  view_216 = None
        mul_139: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_138, 1.1111111111111112);  mul_138 = None
        add_62: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_58, mul_139);  mul_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_25: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_62, 2)
        mean_16: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_63: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_16, 1e-06);  mean_16 = None
        rsqrt_16: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
        mul_140: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_62, rsqrt_16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_141: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_76, mul_140);  mul_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:766 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_33: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 33)
        inductor_random_default_50: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_33, 'rand');  inductor_lookup_seed_default_33 = None
        gt_34: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_50, 0.1);  inductor_random_default_50 = None
        mul_142: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_34, mul_141);  mul_141 = None
        mul_143: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_142, 1.1111111111111112);  mul_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        add_64: "i64[128][1]cuda:0" = torch.ops.aten.add.Tensor(iota_4, 0);  iota_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_6: "i64[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_64, 0);  add_64 = None
        unsqueeze_7: "i64[1, 1, 128][128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 1);  unsqueeze_6 = None
        unsqueeze_8: "i64[1, 1, 128, 1][128, 128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_11: "i64[1, 1, 1, 128][128, 128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.le.Tensor(unsqueeze_11, unsqueeze_8);  unsqueeze_11 = unsqueeze_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_33: "b8[32, 1, 128, 128][0, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(le, [32, -1, 128, 128]);  le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default_1: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_33, full_default_1, full_default_2);  expand_33 = full_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:732 in forward, code: hidden_states = self.dropout(inputs_embeds)
        inductor_lookup_seed_default_34: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 34)
        inductor_random_default_49: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_34, 'rand');  inductor_lookup_seed_default_34 = None
        gt_35: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_49, 0.1);  inductor_random_default_49 = None
        mul_144: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_35, embedding)
        mul_145: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_144, 1.1111111111111112);  mul_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_26: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(mul_145, 2)
        mean_17: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_26, [-1], True);  pow_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_68: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_17, 1e-06);  mean_17 = None
        rsqrt_17: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        mul_146: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_145, rsqrt_17)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_147: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_78, mul_146);  mul_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_283: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_79, torch.bfloat16);  primals_79 = None
        convert_element_type_284: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_147, torch.bfloat16);  mul_147 = None
        permute_97: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_283, [1, 0]);  convert_element_type_283 = None
        view_218: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_284, [4096, 512]);  convert_element_type_284 = None
        mm_56: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_218, permute_97)
        view_219: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_56, [32, 128, 384]);  mm_56 = None
        view_220: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_219, [32, 128, -1, 64]);  view_219 = None
        permute_98: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_220, [0, 2, 1, 3]);  view_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_287: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_80, torch.bfloat16);  primals_80 = None
        permute_99: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_287, [1, 0]);  convert_element_type_287 = None
        mm_57: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_218, permute_99)
        view_222: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_57, [32, 128, 384]);  mm_57 = None
        view_223: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_222, [32, 128, -1, 64]);  view_222 = None
        permute_100: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_223, [0, 2, 1, 3]);  view_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_291: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_81, torch.bfloat16);  primals_81 = None
        permute_101: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_291, [1, 0]);  convert_element_type_291 = None
        mm_58: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_218, permute_101)
        view_225: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_58, [32, 128, 384]);  mm_58 = None
        view_226: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_225, [32, 128, -1, 64]);  view_225 = None
        permute_102: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_103: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_100, [0, 1, 3, 2]);  permute_100 = None
        expand_35: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_98, [32, 6, 128, 64]);  permute_98 = None
        clone_32: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_227: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_32, [192, 128, 64]);  clone_32 = None
        expand_36: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_103, [32, 6, 64, 128]);  permute_103 = None
        clone_33: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_228: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_33, [192, 64, 128]);  clone_33 = None
        bmm_16: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_227, view_228)
        view_229: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_16, [32, 6, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:208 in _relative_position_bucket, code: relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        full_1: "i64[128, 128][128, 1]cuda:0" = torch.ops.aten.full.default([128, 128], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_1: "i64[128, 128][128, 1]cuda:0" = torch.ops.aten.minimum.default(sub, full_1);  sub = full_1 = None
        neg: "i64[128, 128][128, 1]cuda:0" = torch.ops.aten.neg.default(minimum_1);  minimum_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:213 in _relative_position_bucket, code: is_small = relative_position < max_exact
        lt_1: "b8[128, 128][128, 1]cuda:0" = torch.ops.aten.lt.Scalar(neg, 16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:217 in _relative_position_bucket, code: torch.log(relative_position.float() / max_exact)
        convert_element_type_297: "f32[128, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(neg, torch.float32)
        div_10: "f32[128, 128][128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_297, 16);  convert_element_type_297 = None
        log_1: "f32[128, 128][128, 1]cuda:0" = torch.ops.aten.log.default(div_10);  div_10 = None
        div_11: "f32[128, 128][128, 1]cuda:0" = torch.ops.aten.div.Tensor(log_1, 2.0794415416798357);  log_1 = None
        mul_148: "f32[128, 128][128, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_11, 16);  div_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:220 in _relative_position_bucket, code: ).to(torch.long)
        convert_element_type_298: "i64[128, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_148, torch.int64);  mul_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:216 in _relative_position_bucket, code: relative_position_if_large = max_exact + (
        add_70: "i64[128, 128][128, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_298, 16);  convert_element_type_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:222 in _relative_position_bucket, code: relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        full_2: "i64[128, 128][128, 1]cuda:0" = torch.ops.aten.full.default([128, 128], 31, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:221 in _relative_position_bucket, code: relative_position_if_large = torch.min(
        minimum_2: "i64[128, 128][128, 1]cuda:0" = torch.ops.aten.minimum.default(add_70, full_2);  add_70 = full_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:225 in _relative_position_bucket, code: relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        where_4: "i64[128, 128][128, 1]cuda:0" = torch.ops.aten.where.self(lt_1, neg, minimum_2);  lt_1 = neg = minimum_2 = None
        add_71: "i64[128, 128][128, 1]cuda:0" = torch.ops.aten.add.Tensor(where_4, 0);  where_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:241 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        embedding_3: "f32[128, 128, 6][768, 6, 1]cuda:0" = torch.ops.aten.embedding.default(primals_82, add_71);  primals_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:242 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        permute_104: "f32[6, 128, 128][1, 768, 6]cuda:0" = torch.ops.aten.permute.default(embedding_3, [2, 0, 1])
        unsqueeze_17: "f32[1, 6, 128, 128][6, 1, 768, 6]cuda:0" = torch.ops.aten.unsqueeze.default(permute_104, 0);  permute_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:317 in forward, code: position_bias = position_bias + causal_mask
        add_72: "f32[32, 6, 128, 128][98304, 1, 768, 6]cuda:0" = torch.ops.aten.add.Tensor(unsqueeze_17, where_2);  unsqueeze_17 = where_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_73: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_229, add_72);  view_229 = None
        convert_element_type_299: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_73, torch.bfloat16);  add_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_300: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_299, torch.float32);  convert_element_type_299 = None
        amax_8: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_300, [-1], True)
        sub_10: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_300, amax_8);  convert_element_type_300 = None
        exp_8: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_9: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_12: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = None
        convert_element_type_301: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_12, torch.bfloat16);  div_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_35: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 35)
        inductor_random_default_48: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 6, 128, 128], inductor_lookup_seed_default_35, 'rand');  inductor_lookup_seed_default_35 = None
        convert_element_type_default_55: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_48, torch.bfloat16);  inductor_random_default_48 = None
        gt_36: "b8[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_55, 0.1);  convert_element_type_default_55 = None
        mul_149: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_36, convert_element_type_301);  convert_element_type_301 = None
        mul_150: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_149, 1.1111111111111112);  mul_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_37: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(mul_150, [32, 6, 128, 128]);  mul_150 = None
        view_233: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_37, [192, 128, 128]);  expand_37 = None
        expand_38: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_102, [32, 6, 128, 64]);  permute_102 = None
        clone_34: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_234: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_34, [192, 128, 64]);  clone_34 = None
        bmm_17: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_233, view_234)
        view_235: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_17, [32, 6, 128, 64]);  bmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_105: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_235, [0, 2, 1, 3]);  view_235 = None
        clone_35: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_105, memory_format = torch.contiguous_format);  permute_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_236: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_35, [32, 128, -1]);  clone_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_304: "bf16[512, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_83, torch.bfloat16);  primals_83 = None
        permute_106: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_304, [1, 0]);  convert_element_type_304 = None
        view_237: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_236, [4096, 384]);  view_236 = None
        mm_59: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_237, permute_106)
        view_238: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_59, [32, 128, 512]);  mm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_36: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 36)
        inductor_random_default_47: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_36, 'rand');  inductor_lookup_seed_default_36 = None
        convert_element_type_default_54: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_47, torch.bfloat16);  inductor_random_default_47 = None
        gt_37: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_54, 0.1);  convert_element_type_default_54 = None
        mul_151: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_37, view_238);  view_238 = None
        mul_152: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_151, 1.1111111111111112);  mul_151 = None
        add_74: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_145, mul_152);  mul_145 = mul_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_27: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_74, 2)
        mean_18: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_27, [-1], True);  pow_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_75: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_18, 1e-06);  mean_18 = None
        rsqrt_18: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        mul_153: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_74, rsqrt_18)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_154: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_84, mul_153);  mul_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_307: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_85, torch.bfloat16);  primals_85 = None
        convert_element_type_308: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_154, torch.bfloat16);  mul_154 = None
        permute_107: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_307, [1, 0]);  convert_element_type_307 = None
        view_239: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_308, [4096, 512]);  convert_element_type_308 = None
        mm_60: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_239, permute_107)
        view_240: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_60, [32, 128, 384]);  mm_60 = None
        view_241: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_240, [32, 128, -1, 64]);  view_240 = None
        permute_108: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_241, [0, 2, 1, 3]);  view_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_311: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_86, torch.bfloat16);  primals_86 = None
        convert_element_type_312: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_143, torch.bfloat16)
        permute_109: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_311, [1, 0]);  convert_element_type_311 = None
        view_242: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_312, [4096, 512]);  convert_element_type_312 = None
        mm_61: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_242, permute_109)
        view_243: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_61, [32, 128, 384]);  mm_61 = None
        view_244: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_243, [32, 128, -1, 64]);  view_243 = None
        permute_110: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_315: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_87, torch.bfloat16);  primals_87 = None
        permute_111: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_315, [1, 0]);  convert_element_type_315 = None
        mm_62: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_242, permute_111)
        view_246: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_62, [32, 128, 384]);  mm_62 = None
        view_247: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_246, [32, 128, -1, 64]);  view_246 = None
        permute_112: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_247, [0, 2, 1, 3]);  view_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_113: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_110, [0, 1, 3, 2]);  permute_110 = None
        expand_39: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_108, [32, 6, 128, 64]);  permute_108 = None
        clone_36: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_248: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_36, [192, 128, 64]);  clone_36 = None
        expand_40: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_113, [32, 6, 64, 128]);  permute_113 = None
        clone_37: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_249: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_37, [192, 64, 128]);  clone_37 = None
        bmm_18: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_248, view_249)
        view_250: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_18, [32, 6, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        convert_element_type_default_15: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_250, torch.bfloat16);  view_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_322: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_default_15, torch.float32);  convert_element_type_default_15 = None
        amax_9: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_322, [-1], True)
        sub_11: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_322, amax_9);  convert_element_type_322 = None
        exp_9: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        sum_10: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_13: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = None
        convert_element_type_323: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_37: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 37)
        inductor_random_default_46: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 6, 128, 128], inductor_lookup_seed_default_37, 'rand');  inductor_lookup_seed_default_37 = None
        convert_element_type_default_53: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_46, torch.bfloat16);  inductor_random_default_46 = None
        gt_38: "b8[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_53, 0.1);  convert_element_type_default_53 = None
        mul_155: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_38, convert_element_type_323);  convert_element_type_323 = None
        mul_156: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_155, 1.1111111111111112);  mul_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_41: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(mul_156, [32, 6, 128, 128]);  mul_156 = None
        view_254: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_41, [192, 128, 128]);  expand_41 = None
        expand_42: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_112, [32, 6, 128, 64]);  permute_112 = None
        clone_38: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_255: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_38, [192, 128, 64]);  clone_38 = None
        bmm_19: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_254, view_255)
        view_256: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_19, [32, 6, 128, 64]);  bmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_114: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
        clone_39: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_257: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_39, [32, 128, -1]);  clone_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_326: "bf16[512, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_88, torch.bfloat16);  primals_88 = None
        permute_115: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_326, [1, 0]);  convert_element_type_326 = None
        view_258: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_257, [4096, 384]);  view_257 = None
        mm_63: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_258, permute_115)
        view_259: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_63, [32, 128, 512]);  mm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:400 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_38: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 38)
        inductor_random_default_45: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_38, 'rand');  inductor_lookup_seed_default_38 = None
        convert_element_type_default_52: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_45, torch.bfloat16);  inductor_random_default_45 = None
        gt_39: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_52, 0.1);  convert_element_type_default_52 = None
        mul_157: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_39, view_259);  view_259 = None
        mul_158: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_157, 1.1111111111111112);  mul_157 = None
        add_78: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_74, mul_158);  mul_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_28: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_78, 2)
        mean_19: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_28, [-1], True);  pow_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_79: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_19, 1e-06);  mean_19 = None
        rsqrt_19: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
        mul_159: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_78, rsqrt_19)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_160: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_89, mul_159);  mul_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        convert_element_type_329: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_90, torch.bfloat16);  primals_90 = None
        convert_element_type_330: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_160, torch.bfloat16);  mul_160 = None
        permute_116: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_329, [1, 0]);  convert_element_type_329 = None
        view_260: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_330, [4096, 512]);  convert_element_type_330 = None
        mm_64: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_260, permute_116)
        view_261: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_64, [32, 128, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_161: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_261, 0.5)
        convert_element_type_333: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_261, torch.float32)
        pow_29: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_333, 3.0);  convert_element_type_333 = None
        mul_162: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_29, 0.044715);  pow_29 = None
        add_80: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_261, mul_162);  view_261 = mul_162 = None
        mul_163: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_80, 0.7978845608028654);  add_80 = None
        tanh_8: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.tanh.default(mul_163);  mul_163 = None
        add_81: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_164: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_161, add_81);  mul_161 = add_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        convert_element_type_334: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_91, torch.bfloat16);  primals_91 = None
        permute_117: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_334, [1, 0]);  convert_element_type_334 = None
        mm_65: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_260, permute_117)
        view_263: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_65, [32, 128, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_165: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_164, view_263);  mul_164 = view_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:109 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_39: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 39)
        inductor_random_default_44: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 1024], inductor_lookup_seed_default_39, 'rand');  inductor_lookup_seed_default_39 = None
        gt_40: "b8[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_44, 0.1);  inductor_random_default_44 = None
        mul_166: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_40, mul_165);  mul_165 = None
        mul_167: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_166, 1.1111111111111112);  mul_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        convert_element_type_338: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_92, torch.bfloat16);  primals_92 = None
        convert_element_type_339: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_167, torch.bfloat16);  mul_167 = None
        permute_118: "bf16[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_338, [1, 0]);  convert_element_type_338 = None
        view_264: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_339, [4096, 1024]);  convert_element_type_339 = None
        mm_66: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_264, permute_118)
        view_265: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_66, [32, 128, 512]);  mm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_40: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 40)
        inductor_random_default_43: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_40, 'rand');  inductor_lookup_seed_default_40 = None
        convert_element_type_default_51: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_43, torch.bfloat16);  inductor_random_default_43 = None
        gt_41: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_51, 0.1);  convert_element_type_default_51 = None
        mul_168: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_41, view_265);  view_265 = None
        mul_169: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_168, 1.1111111111111112);  mul_168 = None
        add_82: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_78, mul_169);  mul_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_30: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_82, 2)
        mean_20: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_30, [-1], True);  pow_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_83: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_20, 1e-06);  mean_20 = None
        rsqrt_20: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
        mul_170: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_82, rsqrt_20)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_171: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_93, mul_170);  mul_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_342: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_94, torch.bfloat16);  primals_94 = None
        convert_element_type_343: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_171, torch.bfloat16);  mul_171 = None
        permute_119: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_342, [1, 0]);  convert_element_type_342 = None
        view_266: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_343, [4096, 512]);  convert_element_type_343 = None
        mm_67: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_266, permute_119)
        view_267: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_67, [32, 128, 384]);  mm_67 = None
        view_268: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_267, [32, 128, -1, 64]);  view_267 = None
        permute_120: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_268, [0, 2, 1, 3]);  view_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_346: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_95, torch.bfloat16);  primals_95 = None
        permute_121: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_346, [1, 0]);  convert_element_type_346 = None
        mm_68: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_266, permute_121)
        view_270: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_68, [32, 128, 384]);  mm_68 = None
        view_271: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_270, [32, 128, -1, 64]);  view_270 = None
        permute_122: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_271, [0, 2, 1, 3]);  view_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_350: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_96, torch.bfloat16);  primals_96 = None
        permute_123: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_350, [1, 0]);  convert_element_type_350 = None
        mm_69: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_266, permute_123)
        view_273: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_69, [32, 128, 384]);  mm_69 = None
        view_274: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_273, [32, 128, -1, 64]);  view_273 = None
        permute_124: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_274, [0, 2, 1, 3]);  view_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_125: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_122, [0, 1, 3, 2]);  permute_122 = None
        expand_43: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_120, [32, 6, 128, 64]);  permute_120 = None
        clone_40: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_275: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_40, [192, 128, 64]);  clone_40 = None
        expand_44: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_125, [32, 6, 64, 128]);  permute_125 = None
        clone_41: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_276: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_41, [192, 64, 128]);  clone_41 = None
        bmm_20: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_275, view_276)
        view_277: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_20, [32, 6, 128, 128]);  bmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_84: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_277, add_72);  view_277 = None
        convert_element_type_356: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_84, torch.bfloat16);  add_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_357: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_356, torch.float32)
        amax_10: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_357, [-1], True)
        sub_12: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_357, amax_10);  convert_element_type_357 = None
        exp_10: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_11: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_14: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = None
        convert_element_type_358: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_14, torch.bfloat16);  div_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_41: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 41)
        inductor_random_default_42: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 6, 128, 128], inductor_lookup_seed_default_41, 'rand');  inductor_lookup_seed_default_41 = None
        convert_element_type_default_50: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_42, torch.bfloat16);  inductor_random_default_42 = None
        gt_42: "b8[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_50, 0.1);  convert_element_type_default_50 = None
        mul_172: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_42, convert_element_type_358);  convert_element_type_358 = None
        mul_173: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_172, 1.1111111111111112);  mul_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_45: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(mul_173, [32, 6, 128, 128]);  mul_173 = None
        view_281: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_45, [192, 128, 128]);  expand_45 = None
        expand_46: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_124, [32, 6, 128, 64]);  permute_124 = None
        clone_42: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_282: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_42, [192, 128, 64]);  clone_42 = None
        bmm_21: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_281, view_282)
        view_283: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_21, [32, 6, 128, 64]);  bmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_126: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_283, [0, 2, 1, 3]);  view_283 = None
        clone_43: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_284: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_43, [32, 128, -1]);  clone_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_361: "bf16[512, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_97, torch.bfloat16);  primals_97 = None
        permute_127: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_361, [1, 0]);  convert_element_type_361 = None
        view_285: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_284, [4096, 384]);  view_284 = None
        mm_70: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_285, permute_127)
        view_286: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_70, [32, 128, 512]);  mm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_42: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 42)
        inductor_random_default_41: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_42, 'rand');  inductor_lookup_seed_default_42 = None
        convert_element_type_default_49: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_41, torch.bfloat16);  inductor_random_default_41 = None
        gt_43: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_49, 0.1);  convert_element_type_default_49 = None
        mul_174: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_43, view_286);  view_286 = None
        mul_175: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_174, 1.1111111111111112);  mul_174 = None
        add_85: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_82, mul_175);  mul_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_31: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_85, 2)
        mean_21: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_31, [-1], True);  pow_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_86: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_21, 1e-06);  mean_21 = None
        rsqrt_21: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        mul_176: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_85, rsqrt_21)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_177: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_98, mul_176);  mul_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_364: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_99, torch.bfloat16);  primals_99 = None
        convert_element_type_365: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_177, torch.bfloat16);  mul_177 = None
        permute_128: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_364, [1, 0]);  convert_element_type_364 = None
        view_287: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_365, [4096, 512]);  convert_element_type_365 = None
        mm_71: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_287, permute_128)
        view_288: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_71, [32, 128, 384]);  mm_71 = None
        view_289: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_288, [32, 128, -1, 64]);  view_288 = None
        permute_129: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_289, [0, 2, 1, 3]);  view_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_368: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_100, torch.bfloat16);  primals_100 = None
        permute_130: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_368, [1, 0]);  convert_element_type_368 = None
        mm_72: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_242, permute_130)
        view_291: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_72, [32, 128, 384]);  mm_72 = None
        view_292: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_291, [32, 128, -1, 64]);  view_291 = None
        permute_131: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_372: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_101, torch.bfloat16);  primals_101 = None
        permute_132: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_372, [1, 0]);  convert_element_type_372 = None
        mm_73: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_242, permute_132)
        view_294: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_73, [32, 128, 384]);  mm_73 = None
        view_295: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_294, [32, 128, -1, 64]);  view_294 = None
        permute_133: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_295, [0, 2, 1, 3]);  view_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_134: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_131, [0, 1, 3, 2]);  permute_131 = None
        expand_47: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_129, [32, 6, 128, 64]);  permute_129 = None
        clone_44: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_296: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_44, [192, 128, 64]);  clone_44 = None
        expand_48: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_134, [32, 6, 64, 128]);  permute_134 = None
        clone_45: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_297: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_45, [192, 64, 128]);  clone_45 = None
        bmm_22: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_296, view_297)
        view_298: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_22, [32, 6, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        convert_element_type_default_14: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_298, torch.bfloat16);  view_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_379: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_default_14, torch.float32);  convert_element_type_default_14 = None
        amax_11: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_379, [-1], True)
        sub_13: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_379, amax_11);  convert_element_type_379 = None
        exp_11: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_12: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_15: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = None
        convert_element_type_380: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_43: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 43)
        inductor_random_default_40: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 6, 128, 128], inductor_lookup_seed_default_43, 'rand');  inductor_lookup_seed_default_43 = None
        convert_element_type_default_48: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_40, torch.bfloat16);  inductor_random_default_40 = None
        gt_44: "b8[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_48, 0.1);  convert_element_type_default_48 = None
        mul_178: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_44, convert_element_type_380);  convert_element_type_380 = None
        mul_179: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_178, 1.1111111111111112);  mul_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_49: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(mul_179, [32, 6, 128, 128]);  mul_179 = None
        view_302: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_49, [192, 128, 128]);  expand_49 = None
        expand_50: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_133, [32, 6, 128, 64]);  permute_133 = None
        clone_46: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_303: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_46, [192, 128, 64]);  clone_46 = None
        bmm_23: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_302, view_303)
        view_304: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_23, [32, 6, 128, 64]);  bmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_135: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_304, [0, 2, 1, 3]);  view_304 = None
        clone_47: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_305: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_47, [32, 128, -1]);  clone_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_383: "bf16[512, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_102, torch.bfloat16);  primals_102 = None
        permute_136: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_383, [1, 0]);  convert_element_type_383 = None
        view_306: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_305, [4096, 384]);  view_305 = None
        mm_74: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_306, permute_136)
        view_307: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_74, [32, 128, 512]);  mm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:400 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_44: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 44)
        inductor_random_default_39: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_44, 'rand');  inductor_lookup_seed_default_44 = None
        convert_element_type_default_47: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_39, torch.bfloat16);  inductor_random_default_39 = None
        gt_45: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_47, 0.1);  convert_element_type_default_47 = None
        mul_180: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_45, view_307);  view_307 = None
        mul_181: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_180, 1.1111111111111112);  mul_180 = None
        add_88: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_85, mul_181);  mul_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_32: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_88, 2)
        mean_22: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_32, [-1], True);  pow_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_89: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_22, 1e-06);  mean_22 = None
        rsqrt_22: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
        mul_182: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_88, rsqrt_22)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_183: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_103, mul_182);  mul_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        convert_element_type_386: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_104, torch.bfloat16);  primals_104 = None
        convert_element_type_387: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_183, torch.bfloat16);  mul_183 = None
        permute_137: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_386, [1, 0]);  convert_element_type_386 = None
        view_308: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_387, [4096, 512]);  convert_element_type_387 = None
        mm_75: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_308, permute_137)
        view_309: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_75, [32, 128, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_184: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_309, 0.5)
        convert_element_type_390: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_309, torch.float32)
        pow_33: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_390, 3.0);  convert_element_type_390 = None
        mul_185: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_33, 0.044715);  pow_33 = None
        add_90: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_309, mul_185);  view_309 = mul_185 = None
        mul_186: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_90, 0.7978845608028654);  add_90 = None
        tanh_9: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.tanh.default(mul_186);  mul_186 = None
        add_91: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_187: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_184, add_91);  mul_184 = add_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        convert_element_type_391: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_105, torch.bfloat16);  primals_105 = None
        permute_138: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_391, [1, 0]);  convert_element_type_391 = None
        mm_76: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_308, permute_138)
        view_311: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_76, [32, 128, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_188: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_187, view_311);  mul_187 = view_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:109 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_45: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 45)
        inductor_random_default_38: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 1024], inductor_lookup_seed_default_45, 'rand');  inductor_lookup_seed_default_45 = None
        gt_46: "b8[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_38, 0.1);  inductor_random_default_38 = None
        mul_189: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_46, mul_188);  mul_188 = None
        mul_190: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_189, 1.1111111111111112);  mul_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        convert_element_type_395: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_106, torch.bfloat16);  primals_106 = None
        convert_element_type_396: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_190, torch.bfloat16);  mul_190 = None
        permute_139: "bf16[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_395, [1, 0]);  convert_element_type_395 = None
        view_312: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_396, [4096, 1024]);  convert_element_type_396 = None
        mm_77: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_312, permute_139)
        view_313: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_77, [32, 128, 512]);  mm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_46: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 46)
        inductor_random_default_37: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_46, 'rand');  inductor_lookup_seed_default_46 = None
        convert_element_type_default_46: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_37, torch.bfloat16);  inductor_random_default_37 = None
        gt_47: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_46, 0.1);  convert_element_type_default_46 = None
        mul_191: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_47, view_313);  view_313 = None
        mul_192: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_191, 1.1111111111111112);  mul_191 = None
        add_92: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_88, mul_192);  mul_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_34: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_92, 2)
        mean_23: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_34, [-1], True);  pow_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_93: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_23, 1e-06);  mean_23 = None
        rsqrt_23: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        mul_193: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_92, rsqrt_23)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_194: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_107, mul_193);  mul_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_399: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_108, torch.bfloat16);  primals_108 = None
        convert_element_type_400: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_194, torch.bfloat16);  mul_194 = None
        permute_140: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_399, [1, 0]);  convert_element_type_399 = None
        view_314: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_400, [4096, 512]);  convert_element_type_400 = None
        mm_78: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_314, permute_140)
        view_315: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_78, [32, 128, 384]);  mm_78 = None
        view_316: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_315, [32, 128, -1, 64]);  view_315 = None
        permute_141: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_403: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_109, torch.bfloat16);  primals_109 = None
        permute_142: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_403, [1, 0]);  convert_element_type_403 = None
        mm_79: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_314, permute_142)
        view_318: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_79, [32, 128, 384]);  mm_79 = None
        view_319: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_318, [32, 128, -1, 64]);  view_318 = None
        permute_143: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_319, [0, 2, 1, 3]);  view_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_407: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_110, torch.bfloat16);  primals_110 = None
        permute_144: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_407, [1, 0]);  convert_element_type_407 = None
        mm_80: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_314, permute_144)
        view_321: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_80, [32, 128, 384]);  mm_80 = None
        view_322: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_321, [32, 128, -1, 64]);  view_321 = None
        permute_145: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_322, [0, 2, 1, 3]);  view_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_146: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_143, [0, 1, 3, 2]);  permute_143 = None
        expand_51: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_141, [32, 6, 128, 64]);  permute_141 = None
        clone_48: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        view_323: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_48, [192, 128, 64]);  clone_48 = None
        expand_52: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_146, [32, 6, 64, 128]);  permute_146 = None
        clone_49: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_324: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_49, [192, 64, 128]);  clone_49 = None
        bmm_24: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_323, view_324)
        view_325: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_24, [32, 6, 128, 128]);  bmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_94: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_325, add_72);  view_325 = None
        convert_element_type_413: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_94, torch.bfloat16);  add_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_414: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_413, torch.float32)
        amax_12: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_414, [-1], True)
        sub_14: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_414, amax_12);  convert_element_type_414 = None
        exp_12: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
        sum_13: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_16: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = None
        convert_element_type_415: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_16, torch.bfloat16);  div_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_47: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 47)
        inductor_random_default_36: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 6, 128, 128], inductor_lookup_seed_default_47, 'rand');  inductor_lookup_seed_default_47 = None
        convert_element_type_default_45: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_36, torch.bfloat16);  inductor_random_default_36 = None
        gt_48: "b8[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_45, 0.1);  convert_element_type_default_45 = None
        mul_195: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_48, convert_element_type_415);  convert_element_type_415 = None
        mul_196: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_195, 1.1111111111111112);  mul_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_53: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(mul_196, [32, 6, 128, 128]);  mul_196 = None
        view_329: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_53, [192, 128, 128]);  expand_53 = None
        expand_54: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_145, [32, 6, 128, 64]);  permute_145 = None
        clone_50: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_54, memory_format = torch.contiguous_format);  expand_54 = None
        view_330: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_50, [192, 128, 64]);  clone_50 = None
        bmm_25: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_329, view_330)
        view_331: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_25, [32, 6, 128, 64]);  bmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_147: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_331, [0, 2, 1, 3]);  view_331 = None
        clone_51: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_332: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_51, [32, 128, -1]);  clone_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_418: "bf16[512, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_111, torch.bfloat16);  primals_111 = None
        permute_148: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_418, [1, 0]);  convert_element_type_418 = None
        view_333: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_332, [4096, 384]);  view_332 = None
        mm_81: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_333, permute_148)
        view_334: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_81, [32, 128, 512]);  mm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_48: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 48)
        inductor_random_default_35: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_48, 'rand');  inductor_lookup_seed_default_48 = None
        convert_element_type_default_44: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_35, torch.bfloat16);  inductor_random_default_35 = None
        gt_49: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_44, 0.1);  convert_element_type_default_44 = None
        mul_197: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_49, view_334);  view_334 = None
        mul_198: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_197, 1.1111111111111112);  mul_197 = None
        add_95: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_92, mul_198);  mul_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_35: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_95, 2)
        mean_24: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_35, [-1], True);  pow_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_96: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_24, 1e-06);  mean_24 = None
        rsqrt_24: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        mul_199: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_95, rsqrt_24)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_200: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_112, mul_199);  mul_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_421: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_113, torch.bfloat16);  primals_113 = None
        convert_element_type_422: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_200, torch.bfloat16);  mul_200 = None
        permute_149: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_421, [1, 0]);  convert_element_type_421 = None
        view_335: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_422, [4096, 512]);  convert_element_type_422 = None
        mm_82: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_335, permute_149)
        view_336: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_82, [32, 128, 384]);  mm_82 = None
        view_337: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_336, [32, 128, -1, 64]);  view_336 = None
        permute_150: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_337, [0, 2, 1, 3]);  view_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_425: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_114, torch.bfloat16);  primals_114 = None
        permute_151: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_425, [1, 0]);  convert_element_type_425 = None
        mm_83: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_242, permute_151)
        view_339: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_83, [32, 128, 384]);  mm_83 = None
        view_340: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_339, [32, 128, -1, 64]);  view_339 = None
        permute_152: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_340, [0, 2, 1, 3]);  view_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_429: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_115, torch.bfloat16);  primals_115 = None
        permute_153: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_429, [1, 0]);  convert_element_type_429 = None
        mm_84: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_242, permute_153)
        view_342: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_84, [32, 128, 384]);  mm_84 = None
        view_343: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_342, [32, 128, -1, 64]);  view_342 = None
        permute_154: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_343, [0, 2, 1, 3]);  view_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_155: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_152, [0, 1, 3, 2]);  permute_152 = None
        expand_55: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_150, [32, 6, 128, 64]);  permute_150 = None
        clone_52: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
        view_344: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_52, [192, 128, 64]);  clone_52 = None
        expand_56: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_155, [32, 6, 64, 128]);  permute_155 = None
        clone_53: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_345: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_53, [192, 64, 128]);  clone_53 = None
        bmm_26: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_344, view_345)
        view_346: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_26, [32, 6, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        convert_element_type_default_13: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_346, torch.bfloat16);  view_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_436: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_default_13, torch.float32);  convert_element_type_default_13 = None
        amax_13: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_436, [-1], True)
        sub_15: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_436, amax_13);  convert_element_type_436 = None
        exp_13: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
        sum_14: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_17: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = None
        convert_element_type_437: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_17, torch.bfloat16);  div_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_49: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 49)
        inductor_random_default_34: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 6, 128, 128], inductor_lookup_seed_default_49, 'rand');  inductor_lookup_seed_default_49 = None
        convert_element_type_default_43: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_34, torch.bfloat16);  inductor_random_default_34 = None
        gt_50: "b8[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_43, 0.1);  convert_element_type_default_43 = None
        mul_201: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_50, convert_element_type_437);  convert_element_type_437 = None
        mul_202: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_201, 1.1111111111111112);  mul_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_57: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(mul_202, [32, 6, 128, 128]);  mul_202 = None
        view_350: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_57, [192, 128, 128]);  expand_57 = None
        expand_58: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_154, [32, 6, 128, 64]);  permute_154 = None
        clone_54: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
        view_351: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_54, [192, 128, 64]);  clone_54 = None
        bmm_27: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_350, view_351)
        view_352: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_27, [32, 6, 128, 64]);  bmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_156: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_352, [0, 2, 1, 3]);  view_352 = None
        clone_55: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_156, memory_format = torch.contiguous_format);  permute_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_353: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_55, [32, 128, -1]);  clone_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_440: "bf16[512, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_116, torch.bfloat16);  primals_116 = None
        permute_157: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_440, [1, 0]);  convert_element_type_440 = None
        view_354: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_353, [4096, 384]);  view_353 = None
        mm_85: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_354, permute_157)
        view_355: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_85, [32, 128, 512]);  mm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:400 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_50: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 50)
        inductor_random_default_33: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_50, 'rand');  inductor_lookup_seed_default_50 = None
        convert_element_type_default_42: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_33, torch.bfloat16);  inductor_random_default_33 = None
        gt_51: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_42, 0.1);  convert_element_type_default_42 = None
        mul_203: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_51, view_355);  view_355 = None
        mul_204: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_203, 1.1111111111111112);  mul_203 = None
        add_98: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_95, mul_204);  mul_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_36: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_98, 2)
        mean_25: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_36, [-1], True);  pow_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_99: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_25, 1e-06);  mean_25 = None
        rsqrt_25: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
        mul_205: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_98, rsqrt_25)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_206: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_117, mul_205);  mul_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        convert_element_type_443: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_118, torch.bfloat16);  primals_118 = None
        convert_element_type_444: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_206, torch.bfloat16);  mul_206 = None
        permute_158: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_443, [1, 0]);  convert_element_type_443 = None
        view_356: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_444, [4096, 512]);  convert_element_type_444 = None
        mm_86: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_356, permute_158)
        view_357: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_86, [32, 128, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_207: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_357, 0.5)
        convert_element_type_447: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_357, torch.float32)
        pow_37: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_447, 3.0);  convert_element_type_447 = None
        mul_208: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_37, 0.044715);  pow_37 = None
        add_100: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_357, mul_208);  view_357 = mul_208 = None
        mul_209: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_100, 0.7978845608028654);  add_100 = None
        tanh_10: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.tanh.default(mul_209);  mul_209 = None
        add_101: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_210: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_207, add_101);  mul_207 = add_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        convert_element_type_448: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_119, torch.bfloat16);  primals_119 = None
        permute_159: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_448, [1, 0]);  convert_element_type_448 = None
        mm_87: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_356, permute_159)
        view_359: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_87, [32, 128, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_211: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_210, view_359);  mul_210 = view_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:109 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_51: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 51)
        inductor_random_default_32: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 1024], inductor_lookup_seed_default_51, 'rand');  inductor_lookup_seed_default_51 = None
        gt_52: "b8[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_32, 0.1);  inductor_random_default_32 = None
        mul_212: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_52, mul_211);  mul_211 = None
        mul_213: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_212, 1.1111111111111112);  mul_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        convert_element_type_452: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_120, torch.bfloat16);  primals_120 = None
        convert_element_type_453: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_213, torch.bfloat16);  mul_213 = None
        permute_160: "bf16[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_452, [1, 0]);  convert_element_type_452 = None
        view_360: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_453, [4096, 1024]);  convert_element_type_453 = None
        mm_88: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_360, permute_160)
        view_361: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_88, [32, 128, 512]);  mm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_52: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 52)
        inductor_random_default_31: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_52, 'rand');  inductor_lookup_seed_default_52 = None
        convert_element_type_default_41: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_31, torch.bfloat16);  inductor_random_default_31 = None
        gt_53: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_41, 0.1);  convert_element_type_default_41 = None
        mul_214: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_53, view_361);  view_361 = None
        mul_215: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_214, 1.1111111111111112);  mul_214 = None
        add_102: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_98, mul_215);  mul_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_38: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_102, 2)
        mean_26: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_38, [-1], True);  pow_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_103: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_26, 1e-06);  mean_26 = None
        rsqrt_26: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
        mul_216: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_102, rsqrt_26)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_217: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_121, mul_216);  mul_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_456: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_122, torch.bfloat16);  primals_122 = None
        convert_element_type_457: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_217, torch.bfloat16);  mul_217 = None
        permute_161: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_456, [1, 0]);  convert_element_type_456 = None
        view_362: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_457, [4096, 512]);  convert_element_type_457 = None
        mm_89: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_362, permute_161)
        view_363: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_89, [32, 128, 384]);  mm_89 = None
        view_364: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_363, [32, 128, -1, 64]);  view_363 = None
        permute_162: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_364, [0, 2, 1, 3]);  view_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_460: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_123, torch.bfloat16);  primals_123 = None
        permute_163: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_460, [1, 0]);  convert_element_type_460 = None
        mm_90: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_362, permute_163)
        view_366: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_90, [32, 128, 384]);  mm_90 = None
        view_367: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_366, [32, 128, -1, 64]);  view_366 = None
        permute_164: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_367, [0, 2, 1, 3]);  view_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_464: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_124, torch.bfloat16);  primals_124 = None
        permute_165: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_464, [1, 0]);  convert_element_type_464 = None
        mm_91: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_362, permute_165)
        view_369: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_91, [32, 128, 384]);  mm_91 = None
        view_370: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_369, [32, 128, -1, 64]);  view_369 = None
        permute_166: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_370, [0, 2, 1, 3]);  view_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_167: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_164, [0, 1, 3, 2]);  permute_164 = None
        expand_59: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_162, [32, 6, 128, 64]);  permute_162 = None
        clone_56: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
        view_371: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_56, [192, 128, 64]);  clone_56 = None
        expand_60: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_167, [32, 6, 64, 128]);  permute_167 = None
        clone_57: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_372: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_57, [192, 64, 128]);  clone_57 = None
        bmm_28: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_371, view_372)
        view_373: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_28, [32, 6, 128, 128]);  bmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_104: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_373, add_72);  view_373 = None
        convert_element_type_470: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_104, torch.bfloat16);  add_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_471: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_470, torch.float32)
        amax_14: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_471, [-1], True)
        sub_16: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_471, amax_14);  convert_element_type_471 = None
        exp_14: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_15: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_18: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = None
        convert_element_type_472: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_18, torch.bfloat16);  div_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_53: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 53)
        inductor_random_default_30: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 6, 128, 128], inductor_lookup_seed_default_53, 'rand');  inductor_lookup_seed_default_53 = None
        convert_element_type_default_40: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_30, torch.bfloat16);  inductor_random_default_30 = None
        gt_54: "b8[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_40, 0.1);  convert_element_type_default_40 = None
        mul_218: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_54, convert_element_type_472);  convert_element_type_472 = None
        mul_219: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_218, 1.1111111111111112);  mul_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_61: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(mul_219, [32, 6, 128, 128]);  mul_219 = None
        view_377: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_61, [192, 128, 128]);  expand_61 = None
        expand_62: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_166, [32, 6, 128, 64]);  permute_166 = None
        clone_58: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_62, memory_format = torch.contiguous_format);  expand_62 = None
        view_378: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_58, [192, 128, 64]);  clone_58 = None
        bmm_29: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_377, view_378)
        view_379: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_29, [32, 6, 128, 64]);  bmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_168: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_379, [0, 2, 1, 3]);  view_379 = None
        clone_59: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_380: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_59, [32, 128, -1]);  clone_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_475: "bf16[512, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_125, torch.bfloat16);  primals_125 = None
        permute_169: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_475, [1, 0]);  convert_element_type_475 = None
        view_381: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_380, [4096, 384]);  view_380 = None
        mm_92: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_381, permute_169)
        view_382: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_92, [32, 128, 512]);  mm_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_54: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 54)
        inductor_random_default_29: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_54, 'rand');  inductor_lookup_seed_default_54 = None
        convert_element_type_default_39: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_29, torch.bfloat16);  inductor_random_default_29 = None
        gt_55: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_39, 0.1);  convert_element_type_default_39 = None
        mul_220: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_55, view_382);  view_382 = None
        mul_221: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_220, 1.1111111111111112);  mul_220 = None
        add_105: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_102, mul_221);  mul_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_39: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_105, 2)
        mean_27: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_39, [-1], True);  pow_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_106: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_27, 1e-06);  mean_27 = None
        rsqrt_27: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        mul_222: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_105, rsqrt_27)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_223: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_126, mul_222);  mul_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_478: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_127, torch.bfloat16);  primals_127 = None
        convert_element_type_479: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_223, torch.bfloat16);  mul_223 = None
        permute_170: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_478, [1, 0]);  convert_element_type_478 = None
        view_383: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_479, [4096, 512]);  convert_element_type_479 = None
        mm_93: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_383, permute_170)
        view_384: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_93, [32, 128, 384]);  mm_93 = None
        view_385: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_384, [32, 128, -1, 64]);  view_384 = None
        permute_171: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_385, [0, 2, 1, 3]);  view_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_482: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_128, torch.bfloat16);  primals_128 = None
        permute_172: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_482, [1, 0]);  convert_element_type_482 = None
        mm_94: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_242, permute_172)
        view_387: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_94, [32, 128, 384]);  mm_94 = None
        view_388: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_387, [32, 128, -1, 64]);  view_387 = None
        permute_173: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_388, [0, 2, 1, 3]);  view_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_486: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_129, torch.bfloat16);  primals_129 = None
        permute_174: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_486, [1, 0]);  convert_element_type_486 = None
        mm_95: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_242, permute_174)
        view_390: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_95, [32, 128, 384]);  mm_95 = None
        view_391: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_390, [32, 128, -1, 64]);  view_390 = None
        permute_175: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_391, [0, 2, 1, 3]);  view_391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_176: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_173, [0, 1, 3, 2]);  permute_173 = None
        expand_63: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_171, [32, 6, 128, 64]);  permute_171 = None
        clone_60: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
        view_392: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_60, [192, 128, 64]);  clone_60 = None
        expand_64: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_176, [32, 6, 64, 128]);  permute_176 = None
        clone_61: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
        view_393: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_61, [192, 64, 128]);  clone_61 = None
        bmm_30: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_392, view_393)
        view_394: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_30, [32, 6, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        convert_element_type_default_12: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_394, torch.bfloat16);  view_394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_493: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_default_12, torch.float32);  convert_element_type_default_12 = None
        amax_15: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_493, [-1], True)
        sub_17: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_493, amax_15);  convert_element_type_493 = None
        exp_15: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
        sum_16: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_19: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = None
        convert_element_type_494: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16);  div_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_55: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 55)
        inductor_random_default_28: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 6, 128, 128], inductor_lookup_seed_default_55, 'rand');  inductor_lookup_seed_default_55 = None
        convert_element_type_default_38: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_28, torch.bfloat16);  inductor_random_default_28 = None
        gt_56: "b8[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_38, 0.1);  convert_element_type_default_38 = None
        mul_224: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_56, convert_element_type_494);  convert_element_type_494 = None
        mul_225: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_224, 1.1111111111111112);  mul_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_65: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(mul_225, [32, 6, 128, 128]);  mul_225 = None
        view_398: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_65, [192, 128, 128]);  expand_65 = None
        expand_66: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_175, [32, 6, 128, 64]);  permute_175 = None
        clone_62: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
        view_399: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_62, [192, 128, 64]);  clone_62 = None
        bmm_31: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_398, view_399)
        view_400: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_31, [32, 6, 128, 64]);  bmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_177: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_400, [0, 2, 1, 3]);  view_400 = None
        clone_63: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_401: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_63, [32, 128, -1]);  clone_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_497: "bf16[512, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_130, torch.bfloat16);  primals_130 = None
        permute_178: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_497, [1, 0]);  convert_element_type_497 = None
        view_402: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_401, [4096, 384]);  view_401 = None
        mm_96: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_402, permute_178)
        view_403: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_96, [32, 128, 512]);  mm_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:400 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_56: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 56)
        inductor_random_default_27: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_56, 'rand');  inductor_lookup_seed_default_56 = None
        convert_element_type_default_37: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_27, torch.bfloat16);  inductor_random_default_27 = None
        gt_57: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_37, 0.1);  convert_element_type_default_37 = None
        mul_226: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_57, view_403);  view_403 = None
        mul_227: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_226, 1.1111111111111112);  mul_226 = None
        add_108: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_105, mul_227);  mul_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_40: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_108, 2)
        mean_28: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_40, [-1], True);  pow_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_109: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_28, 1e-06);  mean_28 = None
        rsqrt_28: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
        mul_228: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_108, rsqrt_28)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_229: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_131, mul_228);  mul_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        convert_element_type_500: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_132, torch.bfloat16);  primals_132 = None
        convert_element_type_501: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_229, torch.bfloat16);  mul_229 = None
        permute_179: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_500, [1, 0]);  convert_element_type_500 = None
        view_404: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_501, [4096, 512]);  convert_element_type_501 = None
        mm_97: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_404, permute_179)
        view_405: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_97, [32, 128, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_230: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_405, 0.5)
        convert_element_type_504: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_405, torch.float32)
        pow_41: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_504, 3.0);  convert_element_type_504 = None
        mul_231: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_41, 0.044715);  pow_41 = None
        add_110: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_405, mul_231);  view_405 = mul_231 = None
        mul_232: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_110, 0.7978845608028654);  add_110 = None
        tanh_11: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.tanh.default(mul_232);  mul_232 = None
        add_111: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_233: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_230, add_111);  mul_230 = add_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        convert_element_type_505: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_133, torch.bfloat16);  primals_133 = None
        permute_180: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_505, [1, 0]);  convert_element_type_505 = None
        mm_98: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_404, permute_180)
        view_407: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_98, [32, 128, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_234: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_233, view_407);  mul_233 = view_407 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:109 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_57: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 57)
        inductor_random_default_26: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 1024], inductor_lookup_seed_default_57, 'rand');  inductor_lookup_seed_default_57 = None
        gt_58: "b8[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_26, 0.1);  inductor_random_default_26 = None
        mul_235: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_58, mul_234);  mul_234 = None
        mul_236: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_235, 1.1111111111111112);  mul_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        convert_element_type_509: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_134, torch.bfloat16);  primals_134 = None
        convert_element_type_510: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_236, torch.bfloat16);  mul_236 = None
        permute_181: "bf16[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_509, [1, 0]);  convert_element_type_509 = None
        view_408: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_510, [4096, 1024]);  convert_element_type_510 = None
        mm_99: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_408, permute_181)
        view_409: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_99, [32, 128, 512]);  mm_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_58: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 58)
        inductor_random_default_25: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_58, 'rand');  inductor_lookup_seed_default_58 = None
        convert_element_type_default_36: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_25, torch.bfloat16);  inductor_random_default_25 = None
        gt_59: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_36, 0.1);  convert_element_type_default_36 = None
        mul_237: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_59, view_409);  view_409 = None
        mul_238: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_237, 1.1111111111111112);  mul_237 = None
        add_112: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_108, mul_238);  mul_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_42: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_112, 2)
        mean_29: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_42, [-1], True);  pow_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_113: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_29, 1e-06);  mean_29 = None
        rsqrt_29: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
        mul_239: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_112, rsqrt_29)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_240: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_135, mul_239);  mul_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_513: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_136, torch.bfloat16);  primals_136 = None
        convert_element_type_514: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_240, torch.bfloat16);  mul_240 = None
        permute_182: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_513, [1, 0]);  convert_element_type_513 = None
        view_410: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_514, [4096, 512]);  convert_element_type_514 = None
        mm_100: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_410, permute_182)
        view_411: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_100, [32, 128, 384]);  mm_100 = None
        view_412: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_411, [32, 128, -1, 64]);  view_411 = None
        permute_183: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_412, [0, 2, 1, 3]);  view_412 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_517: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_137, torch.bfloat16);  primals_137 = None
        permute_184: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_517, [1, 0]);  convert_element_type_517 = None
        mm_101: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_410, permute_184)
        view_414: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_101, [32, 128, 384]);  mm_101 = None
        view_415: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_414, [32, 128, -1, 64]);  view_414 = None
        permute_185: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_415, [0, 2, 1, 3]);  view_415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_521: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_138, torch.bfloat16);  primals_138 = None
        permute_186: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_521, [1, 0]);  convert_element_type_521 = None
        mm_102: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_410, permute_186)
        view_417: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_102, [32, 128, 384]);  mm_102 = None
        view_418: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_417, [32, 128, -1, 64]);  view_417 = None
        permute_187: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_418, [0, 2, 1, 3]);  view_418 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_188: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_185, [0, 1, 3, 2]);  permute_185 = None
        expand_67: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_183, [32, 6, 128, 64]);  permute_183 = None
        clone_64: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
        view_419: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_64, [192, 128, 64]);  clone_64 = None
        expand_68: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_188, [32, 6, 64, 128]);  permute_188 = None
        clone_65: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
        view_420: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_65, [192, 64, 128]);  clone_65 = None
        bmm_32: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_419, view_420)
        view_421: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_32, [32, 6, 128, 128]);  bmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_114: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_421, add_72);  view_421 = None
        convert_element_type_527: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_114, torch.bfloat16);  add_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_528: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_527, torch.float32)
        amax_16: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_528, [-1], True)
        sub_18: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_528, amax_16);  convert_element_type_528 = None
        exp_16: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
        sum_17: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_20: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = None
        convert_element_type_529: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_20, torch.bfloat16);  div_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_59: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 59)
        inductor_random_default_24: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 6, 128, 128], inductor_lookup_seed_default_59, 'rand');  inductor_lookup_seed_default_59 = None
        convert_element_type_default_35: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_24, torch.bfloat16);  inductor_random_default_24 = None
        gt_60: "b8[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_35, 0.1);  convert_element_type_default_35 = None
        mul_241: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_60, convert_element_type_529);  convert_element_type_529 = None
        mul_242: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_241, 1.1111111111111112);  mul_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_69: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(mul_242, [32, 6, 128, 128]);  mul_242 = None
        view_425: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_69, [192, 128, 128]);  expand_69 = None
        expand_70: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_187, [32, 6, 128, 64]);  permute_187 = None
        clone_66: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_70, memory_format = torch.contiguous_format);  expand_70 = None
        view_426: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_66, [192, 128, 64]);  clone_66 = None
        bmm_33: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_425, view_426)
        view_427: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_33, [32, 6, 128, 64]);  bmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_189: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_427, [0, 2, 1, 3]);  view_427 = None
        clone_67: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_428: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_67, [32, 128, -1]);  clone_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_532: "bf16[512, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_139, torch.bfloat16);  primals_139 = None
        permute_190: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_532, [1, 0]);  convert_element_type_532 = None
        view_429: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_428, [4096, 384]);  view_428 = None
        mm_103: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_429, permute_190)
        view_430: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_103, [32, 128, 512]);  mm_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_60: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 60)
        inductor_random_default_23: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_60, 'rand');  inductor_lookup_seed_default_60 = None
        convert_element_type_default_34: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_23, torch.bfloat16);  inductor_random_default_23 = None
        gt_61: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_34, 0.1);  convert_element_type_default_34 = None
        mul_243: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_61, view_430);  view_430 = None
        mul_244: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_243, 1.1111111111111112);  mul_243 = None
        add_115: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_112, mul_244);  mul_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_43: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_115, 2)
        mean_30: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_43, [-1], True);  pow_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_116: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_30, 1e-06);  mean_30 = None
        rsqrt_30: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        mul_245: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_115, rsqrt_30)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_246: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_140, mul_245);  mul_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_535: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_141, torch.bfloat16);  primals_141 = None
        convert_element_type_536: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_246, torch.bfloat16);  mul_246 = None
        permute_191: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_535, [1, 0]);  convert_element_type_535 = None
        view_431: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_536, [4096, 512]);  convert_element_type_536 = None
        mm_104: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_431, permute_191)
        view_432: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_104, [32, 128, 384]);  mm_104 = None
        view_433: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_432, [32, 128, -1, 64]);  view_432 = None
        permute_192: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_433, [0, 2, 1, 3]);  view_433 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_539: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_142, torch.bfloat16);  primals_142 = None
        permute_193: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_539, [1, 0]);  convert_element_type_539 = None
        mm_105: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_242, permute_193)
        view_435: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_105, [32, 128, 384]);  mm_105 = None
        view_436: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_435, [32, 128, -1, 64]);  view_435 = None
        permute_194: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_436, [0, 2, 1, 3]);  view_436 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_543: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_143, torch.bfloat16);  primals_143 = None
        permute_195: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_543, [1, 0]);  convert_element_type_543 = None
        mm_106: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_242, permute_195)
        view_438: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_106, [32, 128, 384]);  mm_106 = None
        view_439: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_438, [32, 128, -1, 64]);  view_438 = None
        permute_196: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_439, [0, 2, 1, 3]);  view_439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_197: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_194, [0, 1, 3, 2]);  permute_194 = None
        expand_71: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_192, [32, 6, 128, 64]);  permute_192 = None
        clone_68: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
        view_440: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_68, [192, 128, 64]);  clone_68 = None
        expand_72: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_197, [32, 6, 64, 128]);  permute_197 = None
        clone_69: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
        view_441: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_69, [192, 64, 128]);  clone_69 = None
        bmm_34: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_440, view_441)
        view_442: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_34, [32, 6, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        convert_element_type_default_11: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_442, torch.bfloat16);  view_442 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_550: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_default_11, torch.float32);  convert_element_type_default_11 = None
        amax_17: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_550, [-1], True)
        sub_19: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_550, amax_17);  convert_element_type_550 = None
        exp_17: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_18: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_21: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = None
        convert_element_type_551: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16);  div_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_61: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 61)
        inductor_random_default_22: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 6, 128, 128], inductor_lookup_seed_default_61, 'rand');  inductor_lookup_seed_default_61 = None
        convert_element_type_default_33: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_22, torch.bfloat16);  inductor_random_default_22 = None
        gt_62: "b8[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_33, 0.1);  convert_element_type_default_33 = None
        mul_247: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_62, convert_element_type_551);  convert_element_type_551 = None
        mul_248: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_247, 1.1111111111111112);  mul_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_73: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(mul_248, [32, 6, 128, 128]);  mul_248 = None
        view_446: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_73, [192, 128, 128]);  expand_73 = None
        expand_74: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_196, [32, 6, 128, 64]);  permute_196 = None
        clone_70: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
        view_447: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_70, [192, 128, 64]);  clone_70 = None
        bmm_35: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_446, view_447)
        view_448: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_35, [32, 6, 128, 64]);  bmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_198: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None
        clone_71: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_198, memory_format = torch.contiguous_format);  permute_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_449: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_71, [32, 128, -1]);  clone_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_554: "bf16[512, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_144, torch.bfloat16);  primals_144 = None
        permute_199: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_554, [1, 0]);  convert_element_type_554 = None
        view_450: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_449, [4096, 384]);  view_449 = None
        mm_107: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_450, permute_199)
        view_451: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_107, [32, 128, 512]);  mm_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:400 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_62: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 62)
        inductor_random_default_21: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_62, 'rand');  inductor_lookup_seed_default_62 = None
        convert_element_type_default_32: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_21, torch.bfloat16);  inductor_random_default_21 = None
        gt_63: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_32, 0.1);  convert_element_type_default_32 = None
        mul_249: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_63, view_451);  view_451 = None
        mul_250: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_249, 1.1111111111111112);  mul_249 = None
        add_118: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_115, mul_250);  mul_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_44: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_118, 2)
        mean_31: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_44, [-1], True);  pow_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_119: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_31, 1e-06);  mean_31 = None
        rsqrt_31: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
        mul_251: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_118, rsqrt_31)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_252: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_145, mul_251);  mul_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        convert_element_type_557: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_146, torch.bfloat16);  primals_146 = None
        convert_element_type_558: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_252, torch.bfloat16);  mul_252 = None
        permute_200: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_557, [1, 0]);  convert_element_type_557 = None
        view_452: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_558, [4096, 512]);  convert_element_type_558 = None
        mm_108: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_452, permute_200)
        view_453: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_108, [32, 128, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_253: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_453, 0.5)
        convert_element_type_561: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_453, torch.float32)
        pow_45: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_561, 3.0);  convert_element_type_561 = None
        mul_254: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_45, 0.044715);  pow_45 = None
        add_120: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_453, mul_254);  view_453 = mul_254 = None
        mul_255: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_120, 0.7978845608028654);  add_120 = None
        tanh_12: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.tanh.default(mul_255);  mul_255 = None
        add_121: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_12, 1.0);  tanh_12 = None
        mul_256: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_253, add_121);  mul_253 = add_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        convert_element_type_562: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_147, torch.bfloat16);  primals_147 = None
        permute_201: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_562, [1, 0]);  convert_element_type_562 = None
        mm_109: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_452, permute_201)
        view_455: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_109, [32, 128, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_257: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_256, view_455);  mul_256 = view_455 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:109 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_63: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 63)
        inductor_random_default_20: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 1024], inductor_lookup_seed_default_63, 'rand');  inductor_lookup_seed_default_63 = None
        gt_64: "b8[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_20, 0.1);  inductor_random_default_20 = None
        mul_258: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_64, mul_257);  mul_257 = None
        mul_259: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_258, 1.1111111111111112);  mul_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        convert_element_type_566: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_148, torch.bfloat16);  primals_148 = None
        convert_element_type_567: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_259, torch.bfloat16);  mul_259 = None
        permute_202: "bf16[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_566, [1, 0]);  convert_element_type_566 = None
        view_456: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_567, [4096, 1024]);  convert_element_type_567 = None
        mm_110: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_456, permute_202)
        view_457: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_110, [32, 128, 512]);  mm_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_64: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 64)
        inductor_random_default_19: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_64, 'rand');  inductor_lookup_seed_default_64 = None
        convert_element_type_default_31: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_19, torch.bfloat16);  inductor_random_default_19 = None
        gt_65: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_31, 0.1);  convert_element_type_default_31 = None
        mul_260: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_65, view_457);  view_457 = None
        mul_261: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_260, 1.1111111111111112);  mul_260 = None
        add_122: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_118, mul_261);  mul_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_46: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_122, 2)
        mean_32: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_46, [-1], True);  pow_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_123: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_32, 1e-06);  mean_32 = None
        rsqrt_32: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
        mul_262: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_122, rsqrt_32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_263: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_149, mul_262);  mul_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_570: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_150, torch.bfloat16);  primals_150 = None
        convert_element_type_571: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_263, torch.bfloat16);  mul_263 = None
        permute_203: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_570, [1, 0]);  convert_element_type_570 = None
        view_458: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_571, [4096, 512]);  convert_element_type_571 = None
        mm_111: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_458, permute_203)
        view_459: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_111, [32, 128, 384]);  mm_111 = None
        view_460: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_459, [32, 128, -1, 64]);  view_459 = None
        permute_204: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_460, [0, 2, 1, 3]);  view_460 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_574: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_151, torch.bfloat16);  primals_151 = None
        permute_205: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_574, [1, 0]);  convert_element_type_574 = None
        mm_112: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_458, permute_205)
        view_462: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_112, [32, 128, 384]);  mm_112 = None
        view_463: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_462, [32, 128, -1, 64]);  view_462 = None
        permute_206: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_463, [0, 2, 1, 3]);  view_463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_578: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_152, torch.bfloat16);  primals_152 = None
        permute_207: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_578, [1, 0]);  convert_element_type_578 = None
        mm_113: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_458, permute_207)
        view_465: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_113, [32, 128, 384]);  mm_113 = None
        view_466: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_465, [32, 128, -1, 64]);  view_465 = None
        permute_208: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_466, [0, 2, 1, 3]);  view_466 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_209: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_206, [0, 1, 3, 2]);  permute_206 = None
        expand_75: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_204, [32, 6, 128, 64]);  permute_204 = None
        clone_72: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
        view_467: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_72, [192, 128, 64]);  clone_72 = None
        expand_76: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_209, [32, 6, 64, 128]);  permute_209 = None
        clone_73: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
        view_468: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_73, [192, 64, 128]);  clone_73 = None
        bmm_36: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_467, view_468)
        view_469: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_36, [32, 6, 128, 128]);  bmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_124: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_469, add_72);  view_469 = None
        convert_element_type_584: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_124, torch.bfloat16);  add_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_585: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_584, torch.float32)
        amax_18: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_585, [-1], True)
        sub_20: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_585, amax_18);  convert_element_type_585 = None
        exp_18: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
        sum_19: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_22: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = None
        convert_element_type_586: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_22, torch.bfloat16);  div_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_65: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 65)
        inductor_random_default_18: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 6, 128, 128], inductor_lookup_seed_default_65, 'rand');  inductor_lookup_seed_default_65 = None
        convert_element_type_default_30: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_18, torch.bfloat16);  inductor_random_default_18 = None
        gt_66: "b8[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_30, 0.1);  convert_element_type_default_30 = None
        mul_264: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_66, convert_element_type_586);  convert_element_type_586 = None
        mul_265: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_264, 1.1111111111111112);  mul_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_77: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(mul_265, [32, 6, 128, 128]);  mul_265 = None
        view_473: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_77, [192, 128, 128]);  expand_77 = None
        expand_78: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_208, [32, 6, 128, 64]);  permute_208 = None
        clone_74: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_78, memory_format = torch.contiguous_format);  expand_78 = None
        view_474: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_74, [192, 128, 64]);  clone_74 = None
        bmm_37: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_473, view_474)
        view_475: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_37, [32, 6, 128, 64]);  bmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_210: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_475, [0, 2, 1, 3]);  view_475 = None
        clone_75: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_476: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_75, [32, 128, -1]);  clone_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_589: "bf16[512, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_153, torch.bfloat16);  primals_153 = None
        permute_211: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_589, [1, 0]);  convert_element_type_589 = None
        view_477: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_476, [4096, 384]);  view_476 = None
        mm_114: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_477, permute_211)
        view_478: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_114, [32, 128, 512]);  mm_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_66: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 66)
        inductor_random_default_17: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_66, 'rand');  inductor_lookup_seed_default_66 = None
        convert_element_type_default_29: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_17, torch.bfloat16);  inductor_random_default_17 = None
        gt_67: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_29, 0.1);  convert_element_type_default_29 = None
        mul_266: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_67, view_478);  view_478 = None
        mul_267: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_266, 1.1111111111111112);  mul_266 = None
        add_125: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_122, mul_267);  mul_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_47: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_125, 2)
        mean_33: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_47, [-1], True);  pow_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_126: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_33, 1e-06);  mean_33 = None
        rsqrt_33: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        mul_268: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_125, rsqrt_33)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_269: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_154, mul_268);  mul_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_592: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_155, torch.bfloat16);  primals_155 = None
        convert_element_type_593: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_269, torch.bfloat16);  mul_269 = None
        permute_212: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_592, [1, 0]);  convert_element_type_592 = None
        view_479: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_593, [4096, 512]);  convert_element_type_593 = None
        mm_115: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_479, permute_212)
        view_480: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_115, [32, 128, 384]);  mm_115 = None
        view_481: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_480, [32, 128, -1, 64]);  view_480 = None
        permute_213: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_481, [0, 2, 1, 3]);  view_481 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_596: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_156, torch.bfloat16);  primals_156 = None
        permute_214: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_596, [1, 0]);  convert_element_type_596 = None
        mm_116: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_242, permute_214)
        view_483: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_116, [32, 128, 384]);  mm_116 = None
        view_484: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_483, [32, 128, -1, 64]);  view_483 = None
        permute_215: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_484, [0, 2, 1, 3]);  view_484 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_600: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_157, torch.bfloat16);  primals_157 = None
        permute_216: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_600, [1, 0]);  convert_element_type_600 = None
        mm_117: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_242, permute_216)
        view_486: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_117, [32, 128, 384]);  mm_117 = None
        view_487: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_486, [32, 128, -1, 64]);  view_486 = None
        permute_217: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_487, [0, 2, 1, 3]);  view_487 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_218: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_215, [0, 1, 3, 2]);  permute_215 = None
        expand_79: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_213, [32, 6, 128, 64]);  permute_213 = None
        clone_76: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_79, memory_format = torch.contiguous_format);  expand_79 = None
        view_488: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_76, [192, 128, 64]);  clone_76 = None
        expand_80: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_218, [32, 6, 64, 128]);  permute_218 = None
        clone_77: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
        view_489: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_77, [192, 64, 128]);  clone_77 = None
        bmm_38: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_488, view_489)
        view_490: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_38, [32, 6, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        convert_element_type_default_10: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_490, torch.bfloat16);  view_490 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_607: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_default_10, torch.float32);  convert_element_type_default_10 = None
        amax_19: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_607, [-1], True)
        sub_21: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_607, amax_19);  convert_element_type_607 = None
        exp_19: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
        sum_20: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_23: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = None
        convert_element_type_608: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_23, torch.bfloat16);  div_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_67: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 67)
        inductor_random_default_16: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 6, 128, 128], inductor_lookup_seed_default_67, 'rand');  inductor_lookup_seed_default_67 = None
        convert_element_type_default_28: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_16, torch.bfloat16);  inductor_random_default_16 = None
        gt_68: "b8[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_28, 0.1);  convert_element_type_default_28 = None
        mul_270: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_68, convert_element_type_608);  convert_element_type_608 = None
        mul_271: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_270, 1.1111111111111112);  mul_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_81: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(mul_271, [32, 6, 128, 128]);  mul_271 = None
        view_494: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_81, [192, 128, 128]);  expand_81 = None
        expand_82: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_217, [32, 6, 128, 64]);  permute_217 = None
        clone_78: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_82, memory_format = torch.contiguous_format);  expand_82 = None
        view_495: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_78, [192, 128, 64]);  clone_78 = None
        bmm_39: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_494, view_495)
        view_496: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_39, [32, 6, 128, 64]);  bmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_219: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_496, [0, 2, 1, 3]);  view_496 = None
        clone_79: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_219, memory_format = torch.contiguous_format);  permute_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_497: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_79, [32, 128, -1]);  clone_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_611: "bf16[512, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_158, torch.bfloat16);  primals_158 = None
        permute_220: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_611, [1, 0]);  convert_element_type_611 = None
        view_498: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_497, [4096, 384]);  view_497 = None
        mm_118: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_498, permute_220)
        view_499: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_118, [32, 128, 512]);  mm_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:400 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_68: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 68)
        inductor_random_default_15: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_68, 'rand');  inductor_lookup_seed_default_68 = None
        convert_element_type_default_27: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_15, torch.bfloat16);  inductor_random_default_15 = None
        gt_69: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_27, 0.1);  convert_element_type_default_27 = None
        mul_272: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_69, view_499);  view_499 = None
        mul_273: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_272, 1.1111111111111112);  mul_272 = None
        add_128: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_125, mul_273);  mul_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_48: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_128, 2)
        mean_34: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_48, [-1], True);  pow_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_129: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_34, 1e-06);  mean_34 = None
        rsqrt_34: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
        mul_274: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_128, rsqrt_34)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_275: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_159, mul_274);  mul_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        convert_element_type_614: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_160, torch.bfloat16);  primals_160 = None
        convert_element_type_615: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_275, torch.bfloat16);  mul_275 = None
        permute_221: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_614, [1, 0]);  convert_element_type_614 = None
        view_500: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_615, [4096, 512]);  convert_element_type_615 = None
        mm_119: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_500, permute_221)
        view_501: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_119, [32, 128, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_276: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_501, 0.5)
        convert_element_type_618: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_501, torch.float32)
        pow_49: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_618, 3.0);  convert_element_type_618 = None
        mul_277: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_49, 0.044715);  pow_49 = None
        add_130: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_501, mul_277);  view_501 = mul_277 = None
        mul_278: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_130, 0.7978845608028654);  add_130 = None
        tanh_13: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.tanh.default(mul_278);  mul_278 = None
        add_131: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_13, 1.0);  tanh_13 = None
        mul_279: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_276, add_131);  mul_276 = add_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        convert_element_type_619: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_161, torch.bfloat16);  primals_161 = None
        permute_222: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_619, [1, 0]);  convert_element_type_619 = None
        mm_120: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_500, permute_222)
        view_503: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_120, [32, 128, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_280: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_279, view_503);  mul_279 = view_503 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:109 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_69: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 69)
        inductor_random_default_14: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 1024], inductor_lookup_seed_default_69, 'rand');  inductor_lookup_seed_default_69 = None
        gt_70: "b8[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_14, 0.1);  inductor_random_default_14 = None
        mul_281: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_70, mul_280);  mul_280 = None
        mul_282: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_281, 1.1111111111111112);  mul_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        convert_element_type_623: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_162, torch.bfloat16);  primals_162 = None
        convert_element_type_624: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_282, torch.bfloat16);  mul_282 = None
        permute_223: "bf16[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_623, [1, 0]);  convert_element_type_623 = None
        view_504: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_624, [4096, 1024]);  convert_element_type_624 = None
        mm_121: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_504, permute_223)
        view_505: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_121, [32, 128, 512]);  mm_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_70: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 70)
        inductor_random_default_13: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_70, 'rand');  inductor_lookup_seed_default_70 = None
        convert_element_type_default_26: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_13, torch.bfloat16);  inductor_random_default_13 = None
        gt_71: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_26, 0.1);  convert_element_type_default_26 = None
        mul_283: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_71, view_505);  view_505 = None
        mul_284: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_283, 1.1111111111111112);  mul_283 = None
        add_132: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_128, mul_284);  mul_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_50: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_132, 2)
        mean_35: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_50, [-1], True);  pow_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_133: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_35, 1e-06);  mean_35 = None
        rsqrt_35: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
        mul_285: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_132, rsqrt_35)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_286: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_163, mul_285);  mul_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_627: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_164, torch.bfloat16);  primals_164 = None
        convert_element_type_628: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_286, torch.bfloat16);  mul_286 = None
        permute_224: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_627, [1, 0]);  convert_element_type_627 = None
        view_506: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_628, [4096, 512]);  convert_element_type_628 = None
        mm_122: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_506, permute_224)
        view_507: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_122, [32, 128, 384]);  mm_122 = None
        view_508: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_507, [32, 128, -1, 64]);  view_507 = None
        permute_225: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_508, [0, 2, 1, 3]);  view_508 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_631: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_165, torch.bfloat16);  primals_165 = None
        permute_226: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_631, [1, 0]);  convert_element_type_631 = None
        mm_123: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_506, permute_226)
        view_510: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_123, [32, 128, 384]);  mm_123 = None
        view_511: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_510, [32, 128, -1, 64]);  view_510 = None
        permute_227: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_511, [0, 2, 1, 3]);  view_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_635: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_166, torch.bfloat16);  primals_166 = None
        permute_228: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_635, [1, 0]);  convert_element_type_635 = None
        mm_124: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_506, permute_228)
        view_513: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_124, [32, 128, 384]);  mm_124 = None
        view_514: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_513, [32, 128, -1, 64]);  view_513 = None
        permute_229: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_514, [0, 2, 1, 3]);  view_514 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_230: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_227, [0, 1, 3, 2]);  permute_227 = None
        expand_83: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_225, [32, 6, 128, 64]);  permute_225 = None
        clone_80: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_83, memory_format = torch.contiguous_format);  expand_83 = None
        view_515: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_80, [192, 128, 64]);  clone_80 = None
        expand_84: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_230, [32, 6, 64, 128]);  permute_230 = None
        clone_81: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
        view_516: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_81, [192, 64, 128]);  clone_81 = None
        bmm_40: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_515, view_516)
        view_517: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_40, [32, 6, 128, 128]);  bmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_134: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_517, add_72);  view_517 = None
        convert_element_type_641: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_134, torch.bfloat16);  add_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_642: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_641, torch.float32)
        amax_20: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_642, [-1], True)
        sub_22: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_642, amax_20);  convert_element_type_642 = None
        exp_20: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_21: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_24: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = None
        convert_element_type_643: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_24, torch.bfloat16);  div_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_71: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 71)
        inductor_random_default_12: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 6, 128, 128], inductor_lookup_seed_default_71, 'rand');  inductor_lookup_seed_default_71 = None
        convert_element_type_default_25: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_12, torch.bfloat16);  inductor_random_default_12 = None
        gt_72: "b8[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_25, 0.1);  convert_element_type_default_25 = None
        mul_287: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_72, convert_element_type_643);  convert_element_type_643 = None
        mul_288: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_287, 1.1111111111111112);  mul_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_85: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(mul_288, [32, 6, 128, 128]);  mul_288 = None
        view_521: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_85, [192, 128, 128]);  expand_85 = None
        expand_86: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_229, [32, 6, 128, 64]);  permute_229 = None
        clone_82: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_86, memory_format = torch.contiguous_format);  expand_86 = None
        view_522: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_82, [192, 128, 64]);  clone_82 = None
        bmm_41: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_521, view_522)
        view_523: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_41, [32, 6, 128, 64]);  bmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_231: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_523, [0, 2, 1, 3]);  view_523 = None
        clone_83: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_231, memory_format = torch.contiguous_format);  permute_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_524: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_83, [32, 128, -1]);  clone_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_646: "bf16[512, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_167, torch.bfloat16);  primals_167 = None
        permute_232: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_646, [1, 0]);  convert_element_type_646 = None
        view_525: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_524, [4096, 384]);  view_524 = None
        mm_125: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_525, permute_232)
        view_526: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_125, [32, 128, 512]);  mm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_72: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 72)
        inductor_random_default_11: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_72, 'rand');  inductor_lookup_seed_default_72 = None
        convert_element_type_default_24: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_11, torch.bfloat16);  inductor_random_default_11 = None
        gt_73: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_24, 0.1);  convert_element_type_default_24 = None
        mul_289: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_73, view_526);  view_526 = None
        mul_290: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_289, 1.1111111111111112);  mul_289 = None
        add_135: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_132, mul_290);  mul_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_51: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_135, 2)
        mean_36: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_51, [-1], True);  pow_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_136: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_36, 1e-06);  mean_36 = None
        rsqrt_36: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
        mul_291: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_135, rsqrt_36)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_292: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_168, mul_291);  mul_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_649: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_169, torch.bfloat16);  primals_169 = None
        convert_element_type_650: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_292, torch.bfloat16);  mul_292 = None
        permute_233: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_649, [1, 0]);  convert_element_type_649 = None
        view_527: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_650, [4096, 512]);  convert_element_type_650 = None
        mm_126: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_527, permute_233)
        view_528: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_126, [32, 128, 384]);  mm_126 = None
        view_529: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_528, [32, 128, -1, 64]);  view_528 = None
        permute_234: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_529, [0, 2, 1, 3]);  view_529 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_653: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_170, torch.bfloat16);  primals_170 = None
        permute_235: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_653, [1, 0]);  convert_element_type_653 = None
        mm_127: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_242, permute_235)
        view_531: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_127, [32, 128, 384]);  mm_127 = None
        view_532: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_531, [32, 128, -1, 64]);  view_531 = None
        permute_236: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_532, [0, 2, 1, 3]);  view_532 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_657: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_171, torch.bfloat16);  primals_171 = None
        permute_237: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_657, [1, 0]);  convert_element_type_657 = None
        mm_128: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_242, permute_237)
        view_534: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_128, [32, 128, 384]);  mm_128 = None
        view_535: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_534, [32, 128, -1, 64]);  view_534 = None
        permute_238: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_535, [0, 2, 1, 3]);  view_535 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_239: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_236, [0, 1, 3, 2]);  permute_236 = None
        expand_87: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_234, [32, 6, 128, 64]);  permute_234 = None
        clone_84: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_87, memory_format = torch.contiguous_format);  expand_87 = None
        view_536: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_84, [192, 128, 64]);  clone_84 = None
        expand_88: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_239, [32, 6, 64, 128]);  permute_239 = None
        clone_85: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
        view_537: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_85, [192, 64, 128]);  clone_85 = None
        bmm_42: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_536, view_537)
        view_538: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_42, [32, 6, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        convert_element_type_default_9: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_538, torch.bfloat16);  view_538 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_664: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_default_9, torch.float32);  convert_element_type_default_9 = None
        amax_21: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_664, [-1], True)
        sub_23: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_664, amax_21);  convert_element_type_664 = None
        exp_21: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_23);  sub_23 = None
        sum_22: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_25: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = None
        convert_element_type_665: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_25, torch.bfloat16);  div_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_73: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 73)
        inductor_random_default_10: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 6, 128, 128], inductor_lookup_seed_default_73, 'rand');  inductor_lookup_seed_default_73 = None
        convert_element_type_default_23: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_10, torch.bfloat16);  inductor_random_default_10 = None
        gt_74: "b8[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_23, 0.1);  convert_element_type_default_23 = None
        mul_293: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_74, convert_element_type_665);  convert_element_type_665 = None
        mul_294: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_293, 1.1111111111111112);  mul_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_89: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(mul_294, [32, 6, 128, 128]);  mul_294 = None
        view_542: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_89, [192, 128, 128]);  expand_89 = None
        expand_90: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_238, [32, 6, 128, 64]);  permute_238 = None
        clone_86: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_90, memory_format = torch.contiguous_format);  expand_90 = None
        view_543: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_86, [192, 128, 64]);  clone_86 = None
        bmm_43: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_542, view_543)
        view_544: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_43, [32, 6, 128, 64]);  bmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_240: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_544, [0, 2, 1, 3]);  view_544 = None
        clone_87: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_240, memory_format = torch.contiguous_format);  permute_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_545: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_87, [32, 128, -1]);  clone_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_668: "bf16[512, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_172, torch.bfloat16);  primals_172 = None
        permute_241: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_668, [1, 0]);  convert_element_type_668 = None
        view_546: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_545, [4096, 384]);  view_545 = None
        mm_129: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_546, permute_241)
        view_547: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_129, [32, 128, 512]);  mm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:400 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_74: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 74)
        inductor_random_default_9: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_74, 'rand');  inductor_lookup_seed_default_74 = None
        convert_element_type_default_22: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_9, torch.bfloat16);  inductor_random_default_9 = None
        gt_75: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_22, 0.1);  convert_element_type_default_22 = None
        mul_295: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_75, view_547);  view_547 = None
        mul_296: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_295, 1.1111111111111112);  mul_295 = None
        add_138: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_135, mul_296);  mul_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_52: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_138, 2)
        mean_37: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_52, [-1], True);  pow_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_139: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_37, 1e-06);  mean_37 = None
        rsqrt_37: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
        mul_297: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_138, rsqrt_37)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_298: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_173, mul_297);  mul_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        convert_element_type_671: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_174, torch.bfloat16);  primals_174 = None
        convert_element_type_672: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_298, torch.bfloat16);  mul_298 = None
        permute_242: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_671, [1, 0]);  convert_element_type_671 = None
        view_548: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_672, [4096, 512]);  convert_element_type_672 = None
        mm_130: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_548, permute_242)
        view_549: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_130, [32, 128, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_299: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_549, 0.5)
        convert_element_type_675: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_549, torch.float32)
        pow_53: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_675, 3.0);  convert_element_type_675 = None
        mul_300: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_53, 0.044715);  pow_53 = None
        add_140: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_549, mul_300);  view_549 = mul_300 = None
        mul_301: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_140, 0.7978845608028654);  add_140 = None
        tanh_14: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.tanh.default(mul_301);  mul_301 = None
        add_141: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_14, 1.0);  tanh_14 = None
        mul_302: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_299, add_141);  mul_299 = add_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        convert_element_type_676: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_175, torch.bfloat16);  primals_175 = None
        permute_243: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_676, [1, 0]);  convert_element_type_676 = None
        mm_131: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_548, permute_243)
        view_551: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_131, [32, 128, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_303: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_302, view_551);  mul_302 = view_551 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:109 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_75: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 75)
        inductor_random_default_8: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 1024], inductor_lookup_seed_default_75, 'rand');  inductor_lookup_seed_default_75 = None
        gt_76: "b8[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_8, 0.1);  inductor_random_default_8 = None
        mul_304: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_76, mul_303);  mul_303 = None
        mul_305: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_304, 1.1111111111111112);  mul_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        convert_element_type_680: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_176, torch.bfloat16);  primals_176 = None
        convert_element_type_681: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_305, torch.bfloat16);  mul_305 = None
        permute_244: "bf16[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_680, [1, 0]);  convert_element_type_680 = None
        view_552: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_681, [4096, 1024]);  convert_element_type_681 = None
        mm_132: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_552, permute_244)
        view_553: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_132, [32, 128, 512]);  mm_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_76: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 76)
        inductor_random_default_7: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_76, 'rand');  inductor_lookup_seed_default_76 = None
        convert_element_type_default_21: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_7, torch.bfloat16);  inductor_random_default_7 = None
        gt_77: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_21, 0.1);  convert_element_type_default_21 = None
        mul_306: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_77, view_553);  view_553 = None
        mul_307: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_306, 1.1111111111111112);  mul_306 = None
        add_142: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_138, mul_307);  mul_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_54: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_142, 2)
        mean_38: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_54, [-1], True);  pow_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_143: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_38, 1e-06);  mean_38 = None
        rsqrt_38: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
        mul_308: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_142, rsqrt_38)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_309: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_177, mul_308);  mul_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_684: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_178, torch.bfloat16);  primals_178 = None
        convert_element_type_685: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_309, torch.bfloat16);  mul_309 = None
        permute_245: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_684, [1, 0]);  convert_element_type_684 = None
        view_554: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_685, [4096, 512]);  convert_element_type_685 = None
        mm_133: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_554, permute_245)
        view_555: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_133, [32, 128, 384]);  mm_133 = None
        view_556: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_555, [32, 128, -1, 64]);  view_555 = None
        permute_246: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_556, [0, 2, 1, 3]);  view_556 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_688: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_179, torch.bfloat16);  primals_179 = None
        permute_247: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_688, [1, 0]);  convert_element_type_688 = None
        mm_134: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_554, permute_247)
        view_558: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_134, [32, 128, 384]);  mm_134 = None
        view_559: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_558, [32, 128, -1, 64]);  view_558 = None
        permute_248: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_559, [0, 2, 1, 3]);  view_559 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_692: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_180, torch.bfloat16);  primals_180 = None
        permute_249: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_692, [1, 0]);  convert_element_type_692 = None
        mm_135: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_554, permute_249)
        view_561: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_135, [32, 128, 384]);  mm_135 = None
        view_562: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_561, [32, 128, -1, 64]);  view_561 = None
        permute_250: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_562, [0, 2, 1, 3]);  view_562 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_251: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_248, [0, 1, 3, 2]);  permute_248 = None
        expand_91: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_246, [32, 6, 128, 64]);  permute_246 = None
        clone_88: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_91, memory_format = torch.contiguous_format);  expand_91 = None
        view_563: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_88, [192, 128, 64]);  clone_88 = None
        expand_92: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_251, [32, 6, 64, 128]);  permute_251 = None
        clone_89: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
        view_564: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_89, [192, 64, 128]);  clone_89 = None
        bmm_44: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_563, view_564)
        view_565: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_44, [32, 6, 128, 128]);  bmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_144: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_565, add_72);  view_565 = add_72 = None
        convert_element_type_698: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_144, torch.bfloat16);  add_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_699: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_698, torch.float32)
        amax_22: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_699, [-1], True)
        sub_24: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_699, amax_22);  convert_element_type_699 = None
        exp_22: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_24);  sub_24 = None
        sum_23: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_26: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = None
        convert_element_type_700: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_26, torch.bfloat16);  div_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_77: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 77)
        inductor_random_default_6: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 6, 128, 128], inductor_lookup_seed_default_77, 'rand');  inductor_lookup_seed_default_77 = None
        convert_element_type_default_20: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_6, torch.bfloat16);  inductor_random_default_6 = None
        gt_78: "b8[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_20, 0.1);  convert_element_type_default_20 = None
        mul_310: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_78, convert_element_type_700);  convert_element_type_700 = None
        mul_311: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_310, 1.1111111111111112);  mul_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_93: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(mul_311, [32, 6, 128, 128]);  mul_311 = None
        view_569: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_93, [192, 128, 128]);  expand_93 = None
        expand_94: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_250, [32, 6, 128, 64]);  permute_250 = None
        clone_90: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_94, memory_format = torch.contiguous_format);  expand_94 = None
        view_570: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_90, [192, 128, 64]);  clone_90 = None
        bmm_45: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_569, view_570)
        view_571: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_45, [32, 6, 128, 64]);  bmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_252: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_571, [0, 2, 1, 3]);  view_571 = None
        clone_91: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_252, memory_format = torch.contiguous_format);  permute_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_572: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_91, [32, 128, -1]);  clone_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_703: "bf16[512, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_181, torch.bfloat16);  primals_181 = None
        permute_253: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_703, [1, 0]);  convert_element_type_703 = None
        view_573: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_572, [4096, 384]);  view_572 = None
        mm_136: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_573, permute_253)
        view_574: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_136, [32, 128, 512]);  mm_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_78: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 78)
        inductor_random_default_5: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_78, 'rand');  inductor_lookup_seed_default_78 = None
        convert_element_type_default_19: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_5, torch.bfloat16);  inductor_random_default_5 = None
        gt_79: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_19, 0.1);  convert_element_type_default_19 = None
        mul_312: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_79, view_574);  view_574 = None
        mul_313: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_312, 1.1111111111111112);  mul_312 = None
        add_145: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_142, mul_313);  mul_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_55: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_145, 2)
        mean_39: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_55, [-1], True);  pow_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_146: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_39, 1e-06);  mean_39 = None
        rsqrt_39: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
        mul_314: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_145, rsqrt_39)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_315: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_182, mul_314);  mul_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_706: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_183, torch.bfloat16);  primals_183 = None
        convert_element_type_707: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_315, torch.bfloat16);  mul_315 = None
        permute_254: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_706, [1, 0]);  convert_element_type_706 = None
        view_575: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_707, [4096, 512]);  convert_element_type_707 = None
        mm_137: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_575, permute_254)
        view_576: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_137, [32, 128, 384]);  mm_137 = None
        view_577: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_576, [32, 128, -1, 64]);  view_576 = None
        permute_255: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_577, [0, 2, 1, 3]);  view_577 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_710: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_184, torch.bfloat16);  primals_184 = None
        permute_256: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_710, [1, 0]);  convert_element_type_710 = None
        mm_138: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_242, permute_256)
        view_579: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_138, [32, 128, 384]);  mm_138 = None
        view_580: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_579, [32, 128, -1, 64]);  view_579 = None
        permute_257: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_580, [0, 2, 1, 3]);  view_580 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_714: "bf16[384, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_185, torch.bfloat16);  primals_185 = None
        permute_258: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_714, [1, 0]);  convert_element_type_714 = None
        mm_139: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_242, permute_258)
        view_582: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_139, [32, 128, 384]);  mm_139 = None
        view_583: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_582, [32, 128, -1, 64]);  view_582 = None
        permute_259: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_583, [0, 2, 1, 3]);  view_583 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_260: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_257, [0, 1, 3, 2]);  permute_257 = None
        expand_95: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_255, [32, 6, 128, 64]);  permute_255 = None
        clone_92: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_95, memory_format = torch.contiguous_format);  expand_95 = None
        view_584: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_92, [192, 128, 64]);  clone_92 = None
        expand_96: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_260, [32, 6, 64, 128]);  permute_260 = None
        clone_93: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_96, memory_format = torch.contiguous_format);  expand_96 = None
        view_585: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_93, [192, 64, 128]);  clone_93 = None
        bmm_46: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_584, view_585)
        view_586: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_46, [32, 6, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        convert_element_type_default_8: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_586, torch.bfloat16);  view_586 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_721: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_default_8, torch.float32);  convert_element_type_default_8 = None
        amax_23: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_721, [-1], True)
        sub_25: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_721, amax_23);  convert_element_type_721 = None
        exp_23: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_24: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_27: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = None
        convert_element_type_722: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_27, torch.bfloat16);  div_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_79: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 79)
        inductor_random_default_4: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 6, 128, 128], inductor_lookup_seed_default_79, 'rand');  inductor_lookup_seed_default_79 = None
        convert_element_type_default_18: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_4, torch.bfloat16);  inductor_random_default_4 = None
        gt_80: "b8[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_18, 0.1);  convert_element_type_default_18 = None
        mul_316: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_80, convert_element_type_722);  convert_element_type_722 = None
        mul_317: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_316, 1.1111111111111112);  mul_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_97: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(mul_317, [32, 6, 128, 128]);  mul_317 = None
        view_590: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_97, [192, 128, 128]);  expand_97 = None
        expand_98: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_259, [32, 6, 128, 64]);  permute_259 = None
        clone_94: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_98, memory_format = torch.contiguous_format);  expand_98 = None
        view_591: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_94, [192, 128, 64]);  clone_94 = None
        bmm_47: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_590, view_591)
        view_592: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_47, [32, 6, 128, 64]);  bmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_261: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_592, [0, 2, 1, 3]);  view_592 = None
        clone_95: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_261, memory_format = torch.contiguous_format);  permute_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_593: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_95, [32, 128, -1]);  clone_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_725: "bf16[512, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_186, torch.bfloat16);  primals_186 = None
        permute_262: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(convert_element_type_725, [1, 0]);  convert_element_type_725 = None
        view_594: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_593, [4096, 384]);  view_593 = None
        mm_140: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_594, permute_262)
        view_595: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_140, [32, 128, 512]);  mm_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:400 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_80: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 80)
        inductor_random_default_3: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_80, 'rand');  inductor_lookup_seed_default_80 = None
        convert_element_type_default_17: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_3, torch.bfloat16);  inductor_random_default_3 = None
        gt_81: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_17, 0.1);  convert_element_type_default_17 = None
        mul_318: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_81, view_595);  view_595 = None
        mul_319: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_318, 1.1111111111111112);  mul_318 = None
        add_148: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_145, mul_319);  mul_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_56: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_148, 2)
        mean_40: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_56, [-1], True);  pow_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_149: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_40, 1e-06);  mean_40 = None
        rsqrt_40: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
        mul_320: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_148, rsqrt_40)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_321: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_187, mul_320);  mul_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        convert_element_type_728: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_188, torch.bfloat16);  primals_188 = None
        convert_element_type_729: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_321, torch.bfloat16);  mul_321 = None
        permute_263: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_728, [1, 0]);  convert_element_type_728 = None
        view_596: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_729, [4096, 512]);  convert_element_type_729 = None
        mm_141: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_596, permute_263)
        view_597: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_141, [32, 128, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_322: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_597, 0.5)
        convert_element_type_732: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_597, torch.float32)
        pow_57: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_732, 3.0);  convert_element_type_732 = None
        mul_323: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_57, 0.044715);  pow_57 = None
        add_150: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_597, mul_323);  view_597 = mul_323 = None
        mul_324: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_150, 0.7978845608028654);  add_150 = None
        tanh_15: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.tanh.default(mul_324);  mul_324 = None
        add_151: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_15, 1.0);  tanh_15 = None
        mul_325: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_322, add_151);  mul_322 = add_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        convert_element_type_733: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_189, torch.bfloat16);  primals_189 = None
        permute_264: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_733, [1, 0]);  convert_element_type_733 = None
        mm_142: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_596, permute_264)
        view_599: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_142, [32, 128, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_326: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_325, view_599);  mul_325 = view_599 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:109 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_81: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 81)
        inductor_random_default_2: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 1024], inductor_lookup_seed_default_81, 'rand');  inductor_lookup_seed_default_81 = None
        gt_82: "b8[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_2, 0.1);  inductor_random_default_2 = None
        mul_327: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_82, mul_326);  mul_326 = None
        mul_328: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_327, 1.1111111111111112);  mul_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        convert_element_type_737: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_190, torch.bfloat16);  primals_190 = None
        convert_element_type_738: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_328, torch.bfloat16);  mul_328 = None
        permute_265: "bf16[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_737, [1, 0]);  convert_element_type_737 = None
        view_600: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_738, [4096, 1024]);  convert_element_type_738 = None
        mm_143: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_600, permute_265)
        view_601: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_143, [32, 128, 512]);  mm_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_82: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 82)
        inductor_random_default_1: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_82, 'rand');  inductor_lookup_seed_default_82 = None
        convert_element_type_default_16: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_1, torch.bfloat16);  inductor_random_default_1 = None
        gt_83: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_16, 0.1);  convert_element_type_default_16 = None
        mul_329: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_83, view_601);  view_601 = None
        mul_330: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_329, 1.1111111111111112);  mul_329 = None
        add_152: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_148, mul_330);  mul_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_58: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_152, 2)
        mean_41: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_58, [-1], True);  pow_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_153: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_41, 1e-06);  mean_41 = None
        rsqrt_41: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
        mul_331: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_152, rsqrt_41)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_332: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_191, mul_331);  mul_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:766 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_83: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 83);  inductor_seeds_default = None
        inductor_random_default: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default_83, 'rand');  inductor_lookup_seed_default_83 = None
        gt_84: "b8[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_333: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_84, mul_332);  mul_332 = None
        mul_334: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_333, 1.1111111111111112);  mul_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:1143 in forward, code: lm_logits = self.lm_head(sequence_output)
        convert_element_type_741: "bf16[250112, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16);  primals_2 = None
        convert_element_type_742: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_334, torch.bfloat16);  mul_334 = None
        permute_266: "bf16[512, 250112][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_741, [1, 0]);  convert_element_type_741 = None
        view_602: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_742, [4096, 512]);  convert_element_type_742 = None
        mm_144: "bf16[4096, 250112][250112, 1]cuda:0" = torch.ops.aten.mm.default(view_602, permute_266)
        view_603: "bf16[32, 128, 250112][32014336, 250112, 1]cuda:0" = torch.ops.aten.reshape.default(mm_144, [32, 128, 250112]);  mm_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:1150 in forward, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        view_604: "bf16[4096, 250112][250112, 1]cuda:0" = torch.ops.aten.reshape.default(view_603, [-1, 250112])
        view_605: "i64[4096][1]cuda:0" = torch.ops.aten.reshape.default(primals_77, [-1])
        convert_element_type_745: "f32[4096, 250112][250112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_604, torch.float32);  view_604 = None
        amax_24: "f32[4096, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_745, [1], True)
        sub_26: "f32[4096, 250112][250112, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_745, amax_24);  convert_element_type_745 = None
        exp_24: "f32[4096, 250112][250112, 1]cuda:0" = torch.ops.aten.exp.default(sub_26)
        sum_25: "f32[4096, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log_2: "f32[4096, 1][1, 1]cuda:0" = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_27: "f32[4096, 250112][250112, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_26, log_2);  sub_26 = None
        convert_element_type_746: "bf16[4096, 250112][250112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_27, torch.bfloat16);  sub_27 = None
        convert_element_type_747: "f32[4096, 250112][250112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_746, torch.float32);  convert_element_type_746 = None
        ne: "b8[4096][1]cuda:0" = torch.ops.aten.ne.Scalar(view_605, -100)
        full_default_4: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5: "i64[4096][1]cuda:0" = torch.ops.aten.where.self(ne, view_605, full_default_4);  view_605 = full_default_4 = None
        unsqueeze_18: "i64[4096, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(where_5, 1);  where_5 = None
        gather: "f32[4096, 1][1, 1]cuda:0" = torch.ops.aten.gather.default(convert_element_type_747, 1, unsqueeze_18);  convert_element_type_747 = unsqueeze_18 = None
        squeeze: "f32[4096][1]cuda:0" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg_1: "f32[4096][1]cuda:0" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        where_6: "f32[4096][1]cuda:0" = torch.ops.aten.where.self(ne, neg_1, full_default_1);  neg_1 = full_default_1 = None
        sum_26: "i64[][]cuda:0" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type_748: "f32[][]cuda:0" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
        sum_27: "f32[][]cuda:0" = torch.ops.aten.sum.default(where_6);  where_6 = None
        div_28: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(sum_27, convert_element_type_748);  sum_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:1143 in forward, code: lm_logits = self.lm_head(sequence_output)
        permute_269: "bf16[250112, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_266, [1, 0]);  permute_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        permute_273: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        permute_277: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_281: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_285: "bf16[512, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_288: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_590, [0, 2, 1]);  view_590 = None
        permute_289: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_591, [0, 2, 1]);  view_591 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_290: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_584, [0, 2, 1]);  view_584 = None
        permute_291: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_585, [0, 2, 1]);  view_585 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_296: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_258, [1, 0]);  permute_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_301: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_256, [1, 0]);  permute_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_306: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_254, [1, 0]);  permute_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_310: "bf16[512, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_253, [1, 0]);  permute_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_313: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_569, [0, 2, 1]);  view_569 = None
        permute_314: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_570, [0, 2, 1]);  view_570 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_315: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_563, [0, 2, 1]);  view_563 = None
        permute_316: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_564, [0, 2, 1]);  view_564 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_321: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_249, [1, 0]);  permute_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_326: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_247, [1, 0]);  permute_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_331: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_245, [1, 0]);  permute_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        permute_335: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        permute_339: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_343: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_347: "bf16[512, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_241, [1, 0]);  permute_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_350: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_542, [0, 2, 1]);  view_542 = None
        permute_351: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_543, [0, 2, 1]);  view_543 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_352: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_536, [0, 2, 1]);  view_536 = None
        permute_353: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_537, [0, 2, 1]);  view_537 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_358: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_237, [1, 0]);  permute_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_363: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_235, [1, 0]);  permute_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_368: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_372: "bf16[512, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_375: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_521, [0, 2, 1]);  view_521 = None
        permute_376: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_522, [0, 2, 1]);  view_522 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_377: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_515, [0, 2, 1]);  view_515 = None
        permute_378: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_516, [0, 2, 1]);  view_516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_383: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_388: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_226, [1, 0]);  permute_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_393: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_224, [1, 0]);  permute_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        permute_397: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_223, [1, 0]);  permute_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        permute_401: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_405: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_221, [1, 0]);  permute_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_409: "bf16[512, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_412: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_494, [0, 2, 1]);  view_494 = None
        permute_413: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_495, [0, 2, 1]);  view_495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_414: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_488, [0, 2, 1]);  view_488 = None
        permute_415: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_489, [0, 2, 1]);  view_489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_420: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_216, [1, 0]);  permute_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_425: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_214, [1, 0]);  permute_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_430: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_212, [1, 0]);  permute_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_434: "bf16[512, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_437: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_473, [0, 2, 1]);  view_473 = None
        permute_438: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_474, [0, 2, 1]);  view_474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_439: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_467, [0, 2, 1]);  view_467 = None
        permute_440: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_468, [0, 2, 1]);  view_468 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_445: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_450: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_205, [1, 0]);  permute_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_455: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_203, [1, 0]);  permute_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        permute_459: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        permute_463: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_467: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_471: "bf16[512, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_199, [1, 0]);  permute_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_474: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_446, [0, 2, 1]);  view_446 = None
        permute_475: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_447, [0, 2, 1]);  view_447 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_476: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_440, [0, 2, 1]);  view_440 = None
        permute_477: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_441, [0, 2, 1]);  view_441 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_482: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_195, [1, 0]);  permute_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_487: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_193, [1, 0]);  permute_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_492: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_191, [1, 0]);  permute_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_496: "bf16[512, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_190, [1, 0]);  permute_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_499: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_425, [0, 2, 1]);  view_425 = None
        permute_500: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_426, [0, 2, 1]);  view_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_501: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_419, [0, 2, 1]);  view_419 = None
        permute_502: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_420, [0, 2, 1]);  view_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_507: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_512: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_517: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_182, [1, 0]);  permute_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        permute_521: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        permute_525: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_180, [1, 0]);  permute_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_529: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_179, [1, 0]);  permute_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_533: "bf16[512, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_536: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_398, [0, 2, 1]);  view_398 = None
        permute_537: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_399, [0, 2, 1]);  view_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_538: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_392, [0, 2, 1]);  view_392 = None
        permute_539: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_393, [0, 2, 1]);  view_393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_544: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_549: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_554: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_558: "bf16[512, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_169, [1, 0]);  permute_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_561: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_377, [0, 2, 1]);  view_377 = None
        permute_562: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_378, [0, 2, 1]);  view_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_563: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_371, [0, 2, 1]);  view_371 = None
        permute_564: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_372, [0, 2, 1]);  view_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_569: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_574: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_579: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        permute_583: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_160, [1, 0]);  permute_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        permute_587: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_159, [1, 0]);  permute_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_591: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_158, [1, 0]);  permute_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_595: "bf16[512, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_157, [1, 0]);  permute_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_598: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_350, [0, 2, 1]);  view_350 = None
        permute_599: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_351, [0, 2, 1]);  view_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_600: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_344, [0, 2, 1]);  view_344 = None
        permute_601: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_345, [0, 2, 1]);  view_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_606: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_611: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_616: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_149, [1, 0]);  permute_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_620: "bf16[512, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_623: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_329, [0, 2, 1]);  view_329 = None
        permute_624: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_330, [0, 2, 1]);  view_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_625: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_323, [0, 2, 1]);  view_323 = None
        permute_626: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_324, [0, 2, 1]);  view_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_631: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_636: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_641: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        permute_645: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_139, [1, 0]);  permute_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        permute_649: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_138, [1, 0]);  permute_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_653: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_137, [1, 0]);  permute_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_657: "bf16[512, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_660: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_302, [0, 2, 1]);  view_302 = None
        permute_661: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_303, [0, 2, 1]);  view_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_662: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_296, [0, 2, 1]);  view_296 = None
        permute_663: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_297, [0, 2, 1]);  view_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_668: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_673: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_678: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_682: "bf16[512, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_127, [1, 0]);  permute_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_685: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_281, [0, 2, 1]);  view_281 = None
        permute_686: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_282, [0, 2, 1]);  view_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_687: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_275, [0, 2, 1]);  view_275 = None
        permute_688: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_276, [0, 2, 1]);  view_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_693: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_698: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_703: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        permute_707: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        permute_711: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_117, [1, 0]);  permute_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_715: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_719: "bf16[512, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_722: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
        permute_723: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_255, [0, 2, 1]);  view_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_724: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_248, [0, 2, 1]);  view_248 = None
        permute_725: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_249, [0, 2, 1]);  view_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_730: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_735: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_740: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_744: "bf16[512, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_747: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_233, [0, 2, 1]);  view_233 = None
        permute_748: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_234, [0, 2, 1]);  view_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_750: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_227, [0, 2, 1]);  view_227 = None
        permute_751: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_228, [0, 2, 1]);  view_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_756: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_761: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_766: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        permute_770: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        permute_774: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_95, [1, 0]);  permute_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_778: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_782: "bf16[512, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_785: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_205, [0, 2, 1]);  view_205 = None
        permute_786: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_206, [0, 2, 1]);  view_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_787: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_199, [0, 2, 1]);  view_199 = None
        permute_788: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_200, [0, 2, 1]);  view_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_793: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_798: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_803: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        permute_807: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_84, [1, 0]);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        permute_811: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_815: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_82, [1, 0]);  permute_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_819: "bf16[512, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_822: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_178, [0, 2, 1]);  view_178 = None
        permute_823: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_179, [0, 2, 1]);  view_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_824: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_172, [0, 2, 1]);  view_172 = None
        permute_825: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_173, [0, 2, 1]);  view_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_830: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_835: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_840: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        permute_844: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        permute_848: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_852: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_856: "bf16[512, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_859: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_151, [0, 2, 1]);  view_151 = None
        permute_860: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_152, [0, 2, 1]);  view_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_861: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None
        permute_862: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_146, [0, 2, 1]);  view_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_867: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_872: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_877: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        permute_881: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        permute_885: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_889: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_893: "bf16[512, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_896: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_124, [0, 2, 1]);  view_124 = None
        permute_897: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_125, [0, 2, 1]);  view_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_898: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_118, [0, 2, 1]);  view_118 = None
        permute_899: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_119, [0, 2, 1]);  view_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_904: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_909: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_914: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        permute_918: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        permute_922: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_926: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_930: "bf16[512, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_933: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
        permute_934: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_935: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_91, [0, 2, 1]);  view_91 = None
        permute_936: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_92, [0, 2, 1]);  view_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_941: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_946: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_951: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        permute_955: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        permute_959: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_963: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_967: "bf16[512, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_970: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_70, [0, 2, 1]);  view_70 = None
        permute_971: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_71, [0, 2, 1]);  view_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_972: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_64, [0, 2, 1]);  view_64 = None
        permute_973: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_65, [0, 2, 1]);  view_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_978: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_983: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_988: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        permute_992: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        permute_996: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_1000: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_1004: "bf16[512, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_1007: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_43, [0, 2, 1]);  view_43 = None
        permute_1008: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_44, [0, 2, 1]);  view_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_1009: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_37, [0, 2, 1]);  view_37 = None
        permute_1010: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_38, [0, 2, 1]);  view_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_1015: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_1020: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_1025: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        permute_1029: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        permute_1033: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_1037: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_1041: "bf16[512, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_1044: "bf16[192, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_16, [0, 2, 1]);  view_16 = None
        permute_1045: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_17, [0, 2, 1]);  view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_1047: "bf16[192, 64, 128][8192, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
        permute_1048: "bf16[192, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_1053: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_1058: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_1063: "bf16[384, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (div_28, view_603, mul_143, primals_1, primals_3, primals_9, primals_13, primals_18, primals_22, primals_27, primals_31, primals_36, primals_40, primals_45, primals_49, primals_54, primals_58, primals_63, primals_67, primals_72, primals_76, primals_77, primals_78, primals_84, primals_89, primals_93, primals_98, primals_103, primals_107, primals_112, primals_117, primals_121, primals_126, primals_131, primals_135, primals_140, primals_145, primals_149, primals_154, primals_159, primals_163, primals_168, primals_173, primals_177, primals_182, primals_187, primals_191, embedding, gt, rsqrt, view_1, bmm, add_6, embedding_1, amax, sum_1, gt_2, view_20, gt_3, add_9, rsqrt_1, view_22, mm_4, mm_5, gt_4, view_26, gt_5, add_13, rsqrt_2, view_28, convert_element_type_52, amax_1, sum_2, gt_6, view_47, gt_7, add_16, rsqrt_3, view_49, mm_11, mm_12, gt_8, view_53, gt_9, add_20, rsqrt_4, view_55, convert_element_type_87, amax_2, sum_3, gt_10, view_74, gt_11, add_23, rsqrt_5, view_76, mm_18, mm_19, gt_12, view_80, gt_13, add_27, rsqrt_6, view_82, convert_element_type_122, amax_3, sum_4, gt_14, view_101, gt_15, add_30, rsqrt_7, view_103, mm_25, mm_26, gt_16, view_107, gt_17, add_34, rsqrt_8, view_109, convert_element_type_157, amax_4, sum_5, gt_18, view_128, gt_19, add_37, rsqrt_9, view_130, mm_32, mm_33, gt_20, view_134, gt_21, add_41, rsqrt_10, view_136, convert_element_type_192, amax_5, sum_6, gt_22, view_155, gt_23, add_44, rsqrt_11, view_157, mm_39, mm_40, gt_24, view_161, gt_25, add_48, rsqrt_12, view_163, convert_element_type_227, amax_6, sum_7, gt_26, view_182, gt_27, add_51, rsqrt_13, view_184, mm_46, mm_47, gt_28, view_188, gt_29, add_55, rsqrt_14, view_190, convert_element_type_262, amax_7, sum_8, gt_30, view_209, gt_31, add_58, rsqrt_15, view_211, mm_53, mm_54, gt_32, view_215, gt_33, add_62, rsqrt_16, gt_34, unsqueeze_7, gt_35, rsqrt_17, view_218, bmm_16, add_71, embedding_3, amax_8, sum_9, gt_36, view_237, gt_37, add_74, rsqrt_18, view_239, view_242, bmm_18, amax_9, sum_10, gt_38, view_258, gt_39, add_78, rsqrt_19, view_260, mm_64, mm_65, gt_40, view_264, gt_41, add_82, rsqrt_20, view_266, convert_element_type_356, amax_10, sum_11, gt_42, view_285, gt_43, add_85, rsqrt_21, view_287, bmm_22, amax_11, sum_12, gt_44, view_306, gt_45, add_88, rsqrt_22, view_308, mm_75, mm_76, gt_46, view_312, gt_47, add_92, rsqrt_23, view_314, convert_element_type_413, amax_12, sum_13, gt_48, view_333, gt_49, add_95, rsqrt_24, view_335, bmm_26, amax_13, sum_14, gt_50, view_354, gt_51, add_98, rsqrt_25, view_356, mm_86, mm_87, gt_52, view_360, gt_53, add_102, rsqrt_26, view_362, convert_element_type_470, amax_14, sum_15, gt_54, view_381, gt_55, add_105, rsqrt_27, view_383, bmm_30, amax_15, sum_16, gt_56, view_402, gt_57, add_108, rsqrt_28, view_404, mm_97, mm_98, gt_58, view_408, gt_59, add_112, rsqrt_29, view_410, convert_element_type_527, amax_16, sum_17, gt_60, view_429, gt_61, add_115, rsqrt_30, view_431, bmm_34, amax_17, sum_18, gt_62, view_450, gt_63, add_118, rsqrt_31, view_452, mm_108, mm_109, gt_64, view_456, gt_65, add_122, rsqrt_32, view_458, convert_element_type_584, amax_18, sum_19, gt_66, view_477, gt_67, add_125, rsqrt_33, view_479, bmm_38, amax_19, sum_20, gt_68, view_498, gt_69, add_128, rsqrt_34, view_500, mm_119, mm_120, gt_70, view_504, gt_71, add_132, rsqrt_35, view_506, convert_element_type_641, amax_20, sum_21, gt_72, view_525, gt_73, add_135, rsqrt_36, view_527, bmm_42, amax_21, sum_22, gt_74, view_546, gt_75, add_138, rsqrt_37, view_548, mm_130, mm_131, gt_76, view_552, gt_77, add_142, rsqrt_38, view_554, convert_element_type_698, amax_22, sum_23, gt_78, view_573, gt_79, add_145, rsqrt_39, view_575, bmm_46, amax_23, sum_24, gt_80, view_594, gt_81, add_148, rsqrt_40, view_596, mm_141, mm_142, gt_82, view_600, gt_83, add_152, rsqrt_41, gt_84, view_602, view_603, amax_24, log_2, convert_element_type_748, permute_269, permute_273, permute_277, permute_281, permute_285, permute_288, permute_289, permute_290, permute_291, permute_296, permute_301, permute_306, permute_310, permute_313, permute_314, permute_315, permute_316, permute_321, permute_326, permute_331, permute_335, permute_339, permute_343, permute_347, permute_350, permute_351, permute_352, permute_353, permute_358, permute_363, permute_368, permute_372, permute_375, permute_376, permute_377, permute_378, permute_383, permute_388, permute_393, permute_397, permute_401, permute_405, permute_409, permute_412, permute_413, permute_414, permute_415, permute_420, permute_425, permute_430, permute_434, permute_437, permute_438, permute_439, permute_440, permute_445, permute_450, permute_455, permute_459, permute_463, permute_467, permute_471, permute_474, permute_475, permute_476, permute_477, permute_482, permute_487, permute_492, permute_496, permute_499, permute_500, permute_501, permute_502, permute_507, permute_512, permute_517, permute_521, permute_525, permute_529, permute_533, permute_536, permute_537, permute_538, permute_539, permute_544, permute_549, permute_554, permute_558, permute_561, permute_562, permute_563, permute_564, permute_569, permute_574, permute_579, permute_583, permute_587, permute_591, permute_595, permute_598, permute_599, permute_600, permute_601, permute_606, permute_611, permute_616, permute_620, permute_623, permute_624, permute_625, permute_626, permute_631, permute_636, permute_641, permute_645, permute_649, permute_653, permute_657, permute_660, permute_661, permute_662, permute_663, permute_668, permute_673, permute_678, permute_682, permute_685, permute_686, permute_687, permute_688, permute_693, permute_698, permute_703, permute_707, permute_711, permute_715, permute_719, permute_722, permute_723, permute_724, permute_725, permute_730, permute_735, permute_740, permute_744, permute_747, permute_748, permute_750, permute_751, permute_756, permute_761, permute_766, permute_770, permute_774, permute_778, permute_782, permute_785, permute_786, permute_787, permute_788, permute_793, permute_798, permute_803, permute_807, permute_811, permute_815, permute_819, permute_822, permute_823, permute_824, permute_825, permute_830, permute_835, permute_840, permute_844, permute_848, permute_852, permute_856, permute_859, permute_860, permute_861, permute_862, permute_867, permute_872, permute_877, permute_881, permute_885, permute_889, permute_893, permute_896, permute_897, permute_898, permute_899, permute_904, permute_909, permute_914, permute_918, permute_922, permute_926, permute_930, permute_933, permute_934, permute_935, permute_936, permute_941, permute_946, permute_951, permute_955, permute_959, permute_963, permute_967, permute_970, permute_971, permute_972, permute_973, permute_978, permute_983, permute_988, permute_992, permute_996, permute_1000, permute_1004, permute_1007, permute_1008, permute_1009, permute_1010, permute_1015, permute_1020, permute_1025, permute_1029, permute_1033, permute_1037, permute_1041, permute_1044, permute_1045, permute_1047, permute_1048, permute_1053, permute_1058, permute_1063)
