class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[8, 1024][1024, 1]cuda:0", primals_2: "f32[32128, 512][512, 1]cuda:0", primals_3: "f32[512][1]cuda:0", primals_4: "f32[512, 512][512, 1]cuda:0", primals_5: "f32[512, 512][512, 1]cuda:0", primals_6: "f32[512, 512][512, 1]cuda:0", primals_7: "f32[32, 8][8, 1]cuda:0", primals_8: "f32[512, 512][512, 1]cuda:0", primals_9: "f32[512][1]cuda:0", primals_10: "f32[2048, 512][512, 1]cuda:0", primals_11: "f32[512, 2048][2048, 1]cuda:0", primals_12: "f32[512][1]cuda:0", primals_13: "f32[512, 512][512, 1]cuda:0", primals_14: "f32[512, 512][512, 1]cuda:0", primals_15: "f32[512, 512][512, 1]cuda:0", primals_16: "f32[512, 512][512, 1]cuda:0", primals_17: "f32[512][1]cuda:0", primals_18: "f32[2048, 512][512, 1]cuda:0", primals_19: "f32[512, 2048][2048, 1]cuda:0", primals_20: "f32[512][1]cuda:0", primals_21: "f32[512, 512][512, 1]cuda:0", primals_22: "f32[512, 512][512, 1]cuda:0", primals_23: "f32[512, 512][512, 1]cuda:0", primals_24: "f32[512, 512][512, 1]cuda:0", primals_25: "f32[512][1]cuda:0", primals_26: "f32[2048, 512][512, 1]cuda:0", primals_27: "f32[512, 2048][2048, 1]cuda:0", primals_28: "f32[512][1]cuda:0", primals_29: "f32[512, 512][512, 1]cuda:0", primals_30: "f32[512, 512][512, 1]cuda:0", primals_31: "f32[512, 512][512, 1]cuda:0", primals_32: "f32[512, 512][512, 1]cuda:0", primals_33: "f32[512][1]cuda:0", primals_34: "f32[2048, 512][512, 1]cuda:0", primals_35: "f32[512, 2048][2048, 1]cuda:0", primals_36: "f32[512][1]cuda:0", primals_37: "f32[512, 512][512, 1]cuda:0", primals_38: "f32[512, 512][512, 1]cuda:0", primals_39: "f32[512, 512][512, 1]cuda:0", primals_40: "f32[512, 512][512, 1]cuda:0", primals_41: "f32[512][1]cuda:0", primals_42: "f32[2048, 512][512, 1]cuda:0", primals_43: "f32[512, 2048][2048, 1]cuda:0", primals_44: "f32[512][1]cuda:0", primals_45: "f32[512, 512][512, 1]cuda:0", primals_46: "f32[512, 512][512, 1]cuda:0", primals_47: "f32[512, 512][512, 1]cuda:0", primals_48: "f32[512, 512][512, 1]cuda:0", primals_49: "f32[512][1]cuda:0", primals_50: "f32[2048, 512][512, 1]cuda:0", primals_51: "f32[512, 2048][2048, 1]cuda:0", primals_52: "f32[512][1]cuda:0", primals_53: "i64[8, 1024][1024, 1]cuda:0", primals_54: "f32[512][1]cuda:0", primals_55: "f32[512, 512][512, 1]cuda:0", primals_56: "f32[512, 512][512, 1]cuda:0", primals_57: "f32[512, 512][512, 1]cuda:0", primals_58: "f32[32, 8][8, 1]cuda:0", primals_59: "f32[512, 512][512, 1]cuda:0", primals_60: "f32[512][1]cuda:0", primals_61: "f32[512, 512][512, 1]cuda:0", primals_62: "f32[512, 512][512, 1]cuda:0", primals_63: "f32[512, 512][512, 1]cuda:0", primals_64: "f32[512, 512][512, 1]cuda:0", primals_65: "f32[512][1]cuda:0", primals_66: "f32[2048, 512][512, 1]cuda:0", primals_67: "f32[512, 2048][2048, 1]cuda:0", primals_68: "f32[512][1]cuda:0", primals_69: "f32[512, 512][512, 1]cuda:0", primals_70: "f32[512, 512][512, 1]cuda:0", primals_71: "f32[512, 512][512, 1]cuda:0", primals_72: "f32[512, 512][512, 1]cuda:0", primals_73: "f32[512][1]cuda:0", primals_74: "f32[512, 512][512, 1]cuda:0", primals_75: "f32[512, 512][512, 1]cuda:0", primals_76: "f32[512, 512][512, 1]cuda:0", primals_77: "f32[512, 512][512, 1]cuda:0", primals_78: "f32[512][1]cuda:0", primals_79: "f32[2048, 512][512, 1]cuda:0", primals_80: "f32[512, 2048][2048, 1]cuda:0", primals_81: "f32[512][1]cuda:0", primals_82: "f32[512, 512][512, 1]cuda:0", primals_83: "f32[512, 512][512, 1]cuda:0", primals_84: "f32[512, 512][512, 1]cuda:0", primals_85: "f32[512, 512][512, 1]cuda:0", primals_86: "f32[512][1]cuda:0", primals_87: "f32[512, 512][512, 1]cuda:0", primals_88: "f32[512, 512][512, 1]cuda:0", primals_89: "f32[512, 512][512, 1]cuda:0", primals_90: "f32[512, 512][512, 1]cuda:0", primals_91: "f32[512][1]cuda:0", primals_92: "f32[2048, 512][512, 1]cuda:0", primals_93: "f32[512, 2048][2048, 1]cuda:0", primals_94: "f32[512][1]cuda:0", primals_95: "f32[512, 512][512, 1]cuda:0", primals_96: "f32[512, 512][512, 1]cuda:0", primals_97: "f32[512, 512][512, 1]cuda:0", primals_98: "f32[512, 512][512, 1]cuda:0", primals_99: "f32[512][1]cuda:0", primals_100: "f32[512, 512][512, 1]cuda:0", primals_101: "f32[512, 512][512, 1]cuda:0", primals_102: "f32[512, 512][512, 1]cuda:0", primals_103: "f32[512, 512][512, 1]cuda:0", primals_104: "f32[512][1]cuda:0", primals_105: "f32[2048, 512][512, 1]cuda:0", primals_106: "f32[512, 2048][2048, 1]cuda:0", primals_107: "f32[512][1]cuda:0", primals_108: "f32[512, 512][512, 1]cuda:0", primals_109: "f32[512, 512][512, 1]cuda:0", primals_110: "f32[512, 512][512, 1]cuda:0", primals_111: "f32[512, 512][512, 1]cuda:0", primals_112: "f32[512][1]cuda:0", primals_113: "f32[512, 512][512, 1]cuda:0", primals_114: "f32[512, 512][512, 1]cuda:0", primals_115: "f32[512, 512][512, 1]cuda:0", primals_116: "f32[512, 512][512, 1]cuda:0", primals_117: "f32[512][1]cuda:0", primals_118: "f32[2048, 512][512, 1]cuda:0", primals_119: "f32[512, 2048][2048, 1]cuda:0", primals_120: "f32[512][1]cuda:0", primals_121: "f32[512, 512][512, 1]cuda:0", primals_122: "f32[512, 512][512, 1]cuda:0", primals_123: "f32[512, 512][512, 1]cuda:0", primals_124: "f32[512, 512][512, 1]cuda:0", primals_125: "f32[512][1]cuda:0", primals_126: "f32[512, 512][512, 1]cuda:0", primals_127: "f32[512, 512][512, 1]cuda:0", primals_128: "f32[512, 512][512, 1]cuda:0", primals_129: "f32[512, 512][512, 1]cuda:0", primals_130: "f32[512][1]cuda:0", primals_131: "f32[2048, 512][512, 1]cuda:0", primals_132: "f32[512, 2048][2048, 1]cuda:0", primals_133: "f32[512][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:669 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.embedding.default(primals_2, primals_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default: "f32[8, 1, 1024, 1024][1048576, 1048576, 1024, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 1024, 1024], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[64][1]cuda:0" = torch.ops.prims.inductor_seeds.default(64, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:721 in forward, code: hidden_states = self.dropout(inputs_embeds)
        inductor_lookup_seed_default: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_63: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_63, 0.1);  inductor_random_default_63 = None
        mul: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt, embedding)
        mul_1: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_1: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(mul_1, 2)
        mean: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_2: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_2: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1, rsqrt)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_3: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_3, mul_2);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_4, torch.bfloat16);  primals_4 = None
        convert_element_type_1: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        permute: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type, [1, 0]);  convert_element_type = None
        view_1: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1, [8192, 512]);  convert_element_type_1 = None
        mm: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_1, permute)
        view_2: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [8, 1024, 512]);  mm = None
        view_3: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_2, [8, 1024, -1, 64]);  view_2 = None
        permute_1: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_4: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_5, torch.bfloat16);  primals_5 = None
        permute_2: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_4, [1, 0]);  convert_element_type_4 = None
        mm_1: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_1, permute_2)
        view_5: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_1, [8, 1024, 512]);  mm_1 = None
        view_6: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_5, [8, 1024, -1, 64]);  view_5 = None
        permute_3: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_8: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_6, torch.bfloat16);  primals_6 = None
        permute_4: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_8, [1, 0]);  convert_element_type_8 = None
        mm_2: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_1, permute_4)
        view_8: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [8, 1024, 512]);  mm_2 = None
        view_9: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_8, [8, 1024, -1, 64]);  view_8 = None
        permute_5: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_6: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        expand_1: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_1, [8, 8, 1024, 64]);  permute_1 = None
        clone: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_10: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone, [64, 1024, 64]);  clone = None
        expand_2: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_6, [8, 8, 64, 1024]);  permute_6 = None
        clone_1: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_11: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_1, [64, 64, 1024]);  clone_1 = None
        bmm: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_10, view_11)
        view_12: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm, [8, 8, 1024, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:227 in compute_bias, code: context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None] + past_seen_tokens
        iota_4: "i64[1024][1]cuda:0" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_3: "i64[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_4, 1)
        add_3: "i64[1024, 1][1, 1]cuda:0" = torch.ops.aten.add.Tensor(unsqueeze_3, 0);  unsqueeze_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:228 in compute_bias, code: memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        unsqueeze_4: "i64[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_4, 0)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:229 in compute_bias, code: relative_position = memory_position - context_position  # shape (query_length, key_length)
        sub: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(unsqueeze_4, add_3);  unsqueeze_4 = add_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:200 in _relative_position_bucket, code: relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        gt_1: "b8[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(sub, 0)
        convert_element_type_14: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_1, torch.int64);  gt_1 = None
        mul_4: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_14, 16);  convert_element_type_14 = None
        add_4: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_4, 0);  mul_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:201 in _relative_position_bucket, code: relative_position = torch.abs(relative_position)
        abs_1: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.abs.default(sub)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:208 in _relative_position_bucket, code: is_small = relative_position < max_exact
        lt: "b8[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.lt.Scalar(abs_1, 8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:212 in _relative_position_bucket, code: torch.log(relative_position.float() / max_exact)
        convert_element_type_15: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(abs_1, torch.float32)
        div: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_15, 8);  convert_element_type_15 = None
        log: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.log.default(div);  div = None
        div_1: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.div.Tensor(log, 2.772588722239781);  log = None
        mul_5: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_1, 8);  div_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:215 in _relative_position_bucket, code: ).to(torch.long)
        convert_element_type_16: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_5, torch.int64);  mul_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:211 in _relative_position_bucket, code: relative_position_if_large = max_exact + (
        add_5: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_16, 8);  convert_element_type_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:217 in _relative_position_bucket, code: relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        full: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.full.default([1024, 1024], 15, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:216 in _relative_position_bucket, code: relative_position_if_large = torch.min(
        minimum: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.minimum.default(add_5, full);  add_5 = full = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:220 in _relative_position_bucket, code: relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        where_1: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.where.self(lt, abs_1, minimum);  lt = abs_1 = minimum = None
        add_6: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4, where_1);  add_4 = where_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:236 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        embedding_1: "f32[1024, 1024, 8][8192, 8, 1]cuda:0" = torch.ops.aten.embedding.default(primals_7, add_6);  primals_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:237 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        permute_7: "f32[8, 1024, 1024][1, 8192, 8]cuda:0" = torch.ops.aten.permute.default(embedding_1, [2, 0, 1])
        unsqueeze_5: "f32[1, 8, 1024, 1024][8, 1, 8192, 8]cuda:0" = torch.ops.aten.unsqueeze.default(permute_7, 0);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        add_7: "f32[8, 8, 1024, 1024][8388608, 1, 8192, 8]cuda:0" = torch.ops.aten.add.Tensor(unsqueeze_5, full_default);  unsqueeze_5 = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_8: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_12, add_7);  view_12 = None
        convert_element_type_17: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_8, torch.bfloat16);  add_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_18: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_17, torch.float32);  convert_element_type_17 = None
        amax: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_18, [-1], True)
        sub_1: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_18, amax);  convert_element_type_18 = None
        exp: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_2: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        convert_element_type_19: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_1: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_62: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        convert_element_type_default_95: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_62, torch.bfloat16);  inductor_random_default_62 = None
        gt_2: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_95, 0.1);  convert_element_type_default_95 = None
        mul_6: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_2, convert_element_type_19);  convert_element_type_19 = None
        mul_7: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_6, 1.1111111111111112);  mul_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_3: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_7, [8, 8, 1024, 1024]);  mul_7 = None
        view_16: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_3, [64, 1024, 1024]);  expand_3 = None
        expand_4: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_5, [8, 8, 1024, 64]);  permute_5 = None
        clone_2: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_17: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [64, 1024, 64]);  clone_2 = None
        bmm_1: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_16, view_17)
        view_18: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_1, [8, 8, 1024, 64]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_8: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        clone_3: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_19: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_3, [8, 1024, -1]);  clone_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_22: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_8, torch.bfloat16);  primals_8 = None
        permute_9: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_22, [1, 0]);  convert_element_type_22 = None
        view_20: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_19, [8192, 512]);  view_19 = None
        mm_3: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_20, permute_9)
        view_21: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_3, [8, 1024, 512]);  mm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_2: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_61: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        convert_element_type_default_94: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_61, torch.bfloat16);  inductor_random_default_61 = None
        gt_3: "b8[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_94, 0.1);  convert_element_type_default_94 = None
        mul_8: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_3, view_21);  view_21 = None
        mul_9: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_8, 1.1111111111111112);  mul_8 = None
        add_9: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, mul_9);  mul_1 = mul_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_2: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_9, 2)
        mean_1: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_10: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        rsqrt_1: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_10: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_9, rsqrt_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_11: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_9, mul_10);  mul_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        convert_element_type_25: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_10, torch.bfloat16);  primals_10 = None
        convert_element_type_26: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_11, torch.bfloat16);  mul_11 = None
        permute_10: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_25, [1, 0]);  convert_element_type_25 = None
        view_22: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_26, [8192, 512]);  convert_element_type_26 = None
        mm_4: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_22, permute_10)
        view_23: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [8, 1024, 2048]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.relu.default(view_23);  view_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_3: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_60: "f32[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 2048], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        convert_element_type_default_93: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_60, torch.bfloat16);  inductor_random_default_60 = None
        gt_4: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_93, 0.1);  convert_element_type_default_93 = None
        mul_12: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_4, relu)
        mul_13: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_12, 1.1111111111111112);  mul_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        convert_element_type_30: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_11, torch.bfloat16);  primals_11 = None
        convert_element_type_default_35: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_13, torch.bfloat16);  mul_13 = None
        permute_11: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_30, [1, 0]);  convert_element_type_30 = None
        view_24: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_default_35, [8192, 2048]);  convert_element_type_default_35 = None
        mm_5: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_24, permute_11)
        view_25: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_5, [8, 1024, 512]);  mm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_4: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_random_default_59: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        convert_element_type_default_92: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_59, torch.bfloat16);  inductor_random_default_59 = None
        gt_5: "b8[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_92, 0.1);  convert_element_type_default_92 = None
        mul_14: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_5, view_25);  view_25 = None
        mul_15: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, 1.1111111111111112);  mul_14 = None
        add_11: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_9, mul_15);  mul_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_3: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_11, 2)
        mean_2: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_12: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
        rsqrt_2: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_16: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_11, rsqrt_2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_17: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_12, mul_16);  mul_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_34: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_13, torch.bfloat16);  primals_13 = None
        convert_element_type_35: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16);  mul_17 = None
        permute_12: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_34, [1, 0]);  convert_element_type_34 = None
        view_26: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_35, [8192, 512]);  convert_element_type_35 = None
        mm_6: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_26, permute_12)
        view_27: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [8, 1024, 512]);  mm_6 = None
        view_28: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_27, [8, 1024, -1, 64]);  view_27 = None
        permute_13: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_38: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
        permute_14: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_38, [1, 0]);  convert_element_type_38 = None
        mm_7: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_26, permute_14)
        view_30: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_7, [8, 1024, 512]);  mm_7 = None
        view_31: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_30, [8, 1024, -1, 64]);  view_30 = None
        permute_15: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_42: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16);  primals_15 = None
        permute_16: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_42, [1, 0]);  convert_element_type_42 = None
        mm_8: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_26, permute_16)
        view_33: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [8, 1024, 512]);  mm_8 = None
        view_34: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_33, [8, 1024, -1, 64]);  view_33 = None
        permute_17: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_18: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_15, [0, 1, 3, 2]);  permute_15 = None
        expand_5: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_13, [8, 8, 1024, 64]);  permute_13 = None
        clone_4: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_35: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_4, [64, 1024, 64]);  clone_4 = None
        expand_6: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_18, [8, 8, 64, 1024]);  permute_18 = None
        clone_5: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_36: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_5, [64, 64, 1024]);  clone_5 = None
        bmm_2: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_35, view_36)
        view_37: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_2, [8, 8, 1024, 1024]);  bmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_13: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_37, add_7);  view_37 = None
        convert_element_type_48: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_13, torch.bfloat16);  add_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_49: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_48, torch.float32)
        amax_1: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_49, [-1], True)
        sub_2: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_49, amax_1);  convert_element_type_49 = None
        exp_1: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_2: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_3: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = None
        convert_element_type_50: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_5: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_random_default_58: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_5, 'rand');  inductor_lookup_seed_default_5 = None
        convert_element_type_default_91: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_58, torch.bfloat16);  inductor_random_default_58 = None
        gt_6: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_91, 0.1);  convert_element_type_default_91 = None
        mul_18: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_6, convert_element_type_50);  convert_element_type_50 = None
        mul_19: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_18, 1.1111111111111112);  mul_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_7: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_19, [8, 8, 1024, 1024]);  mul_19 = None
        view_41: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_7, [64, 1024, 1024]);  expand_7 = None
        expand_8: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_17, [8, 8, 1024, 64]);  permute_17 = None
        clone_6: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_42: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_6, [64, 1024, 64]);  clone_6 = None
        bmm_3: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_41, view_42)
        view_43: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_3, [8, 8, 1024, 64]);  bmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_19: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_43, [0, 2, 1, 3]);  view_43 = None
        clone_7: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_44: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_7, [8, 1024, -1]);  clone_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_53: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_16, torch.bfloat16);  primals_16 = None
        permute_20: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_53, [1, 0]);  convert_element_type_53 = None
        view_45: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_44, [8192, 512]);  view_44 = None
        mm_9: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_45, permute_20)
        view_46: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_9, [8, 1024, 512]);  mm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_6: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6)
        inductor_random_default_57: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_6, 'rand');  inductor_lookup_seed_default_6 = None
        convert_element_type_default_90: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_57, torch.bfloat16);  inductor_random_default_57 = None
        gt_7: "b8[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_90, 0.1);  convert_element_type_default_90 = None
        mul_20: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_7, view_46);  view_46 = None
        mul_21: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_20, 1.1111111111111112);  mul_20 = None
        add_14: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_11, mul_21);  mul_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_4: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_14, 2)
        mean_3: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_15: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None
        rsqrt_3: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        mul_22: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_14, rsqrt_3)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_23: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_17, mul_22);  mul_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        convert_element_type_56: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_18, torch.bfloat16);  primals_18 = None
        convert_element_type_57: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_23, torch.bfloat16);  mul_23 = None
        permute_21: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_56, [1, 0]);  convert_element_type_56 = None
        view_47: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_57, [8192, 512]);  convert_element_type_57 = None
        mm_10: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_47, permute_21)
        view_48: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [8, 1024, 2048]);  mm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_1: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.relu.default(view_48);  view_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_7: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 7)
        inductor_random_default_56: "f32[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 2048], inductor_lookup_seed_default_7, 'rand');  inductor_lookup_seed_default_7 = None
        convert_element_type_default_89: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_56, torch.bfloat16);  inductor_random_default_56 = None
        gt_8: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_89, 0.1);  convert_element_type_default_89 = None
        mul_24: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_8, relu_1)
        mul_25: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, 1.1111111111111112);  mul_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        convert_element_type_61: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_19, torch.bfloat16);  primals_19 = None
        convert_element_type_default_34: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_25, torch.bfloat16);  mul_25 = None
        permute_22: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_61, [1, 0]);  convert_element_type_61 = None
        view_49: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_default_34, [8192, 2048]);  convert_element_type_default_34 = None
        mm_11: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_49, permute_22)
        view_50: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_11, [8, 1024, 512]);  mm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_8: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 8)
        inductor_random_default_55: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_8, 'rand');  inductor_lookup_seed_default_8 = None
        convert_element_type_default_88: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_55, torch.bfloat16);  inductor_random_default_55 = None
        gt_9: "b8[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_88, 0.1);  convert_element_type_default_88 = None
        mul_26: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_9, view_50);  view_50 = None
        mul_27: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_26, 1.1111111111111112);  mul_26 = None
        add_16: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_14, mul_27);  mul_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_5: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_16, 2)
        mean_4: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_17: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_4, 1e-06);  mean_4 = None
        rsqrt_4: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        mul_28: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_16, rsqrt_4)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_29: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_20, mul_28);  mul_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_65: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_21, torch.bfloat16);  primals_21 = None
        convert_element_type_66: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_29, torch.bfloat16);  mul_29 = None
        permute_23: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_65, [1, 0]);  convert_element_type_65 = None
        view_51: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_66, [8192, 512]);  convert_element_type_66 = None
        mm_12: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_51, permute_23)
        view_52: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_12, [8, 1024, 512]);  mm_12 = None
        view_53: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_52, [8, 1024, -1, 64]);  view_52 = None
        permute_24: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_69: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_22, torch.bfloat16);  primals_22 = None
        permute_25: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_69, [1, 0]);  convert_element_type_69 = None
        mm_13: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_51, permute_25)
        view_55: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_13, [8, 1024, 512]);  mm_13 = None
        view_56: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_55, [8, 1024, -1, 64]);  view_55 = None
        permute_26: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_73: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_23, torch.bfloat16);  primals_23 = None
        permute_27: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_73, [1, 0]);  convert_element_type_73 = None
        mm_14: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_51, permute_27)
        view_58: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_14, [8, 1024, 512]);  mm_14 = None
        view_59: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_58, [8, 1024, -1, 64]);  view_58 = None
        permute_28: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_59, [0, 2, 1, 3]);  view_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_29: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_26, [0, 1, 3, 2]);  permute_26 = None
        expand_9: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_24, [8, 8, 1024, 64]);  permute_24 = None
        clone_8: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_60: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_8, [64, 1024, 64]);  clone_8 = None
        expand_10: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_29, [8, 8, 64, 1024]);  permute_29 = None
        clone_9: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_61: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_9, [64, 64, 1024]);  clone_9 = None
        bmm_4: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_60, view_61)
        view_62: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_4, [8, 8, 1024, 1024]);  bmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_18: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_62, add_7);  view_62 = None
        convert_element_type_79: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_18, torch.bfloat16);  add_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_80: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_79, torch.float32)
        amax_2: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_80, [-1], True)
        sub_3: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_80, amax_2);  convert_element_type_80 = None
        exp_2: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        sum_3: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_4: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = None
        convert_element_type_81: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_9: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 9)
        inductor_random_default_54: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_9, 'rand');  inductor_lookup_seed_default_9 = None
        convert_element_type_default_87: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_54, torch.bfloat16);  inductor_random_default_54 = None
        gt_10: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_87, 0.1);  convert_element_type_default_87 = None
        mul_30: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_10, convert_element_type_81);  convert_element_type_81 = None
        mul_31: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_30, 1.1111111111111112);  mul_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_11: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_31, [8, 8, 1024, 1024]);  mul_31 = None
        view_66: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_11, [64, 1024, 1024]);  expand_11 = None
        expand_12: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_28, [8, 8, 1024, 64]);  permute_28 = None
        clone_10: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_67: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_10, [64, 1024, 64]);  clone_10 = None
        bmm_5: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_66, view_67)
        view_68: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_5, [8, 8, 1024, 64]);  bmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_30: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
        clone_11: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_69: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_11, [8, 1024, -1]);  clone_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_84: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_24, torch.bfloat16);  primals_24 = None
        permute_31: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_84, [1, 0]);  convert_element_type_84 = None
        view_70: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_69, [8192, 512]);  view_69 = None
        mm_15: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_70, permute_31)
        view_71: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_15, [8, 1024, 512]);  mm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_10: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 10)
        inductor_random_default_53: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_10, 'rand');  inductor_lookup_seed_default_10 = None
        convert_element_type_default_86: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_53, torch.bfloat16);  inductor_random_default_53 = None
        gt_11: "b8[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_86, 0.1);  convert_element_type_default_86 = None
        mul_32: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_11, view_71);  view_71 = None
        mul_33: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_32, 1.1111111111111112);  mul_32 = None
        add_19: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_16, mul_33);  mul_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_6: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_19, 2)
        mean_5: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_6, [-1], True);  pow_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_20: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_5, 1e-06);  mean_5 = None
        rsqrt_5: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_34: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_19, rsqrt_5)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_35: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_25, mul_34);  mul_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        convert_element_type_87: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_26, torch.bfloat16);  primals_26 = None
        convert_element_type_88: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_35, torch.bfloat16);  mul_35 = None
        permute_32: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_87, [1, 0]);  convert_element_type_87 = None
        view_72: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_88, [8192, 512]);  convert_element_type_88 = None
        mm_16: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_72, permute_32)
        view_73: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_16, [8, 1024, 2048]);  mm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_2: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.relu.default(view_73);  view_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_11: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 11)
        inductor_random_default_52: "f32[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 2048], inductor_lookup_seed_default_11, 'rand');  inductor_lookup_seed_default_11 = None
        convert_element_type_default_85: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_52, torch.bfloat16);  inductor_random_default_52 = None
        gt_12: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_85, 0.1);  convert_element_type_default_85 = None
        mul_36: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_12, relu_2)
        mul_37: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_36, 1.1111111111111112);  mul_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        convert_element_type_92: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_27, torch.bfloat16);  primals_27 = None
        convert_element_type_default_33: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_37, torch.bfloat16);  mul_37 = None
        permute_33: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_92, [1, 0]);  convert_element_type_92 = None
        view_74: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_default_33, [8192, 2048]);  convert_element_type_default_33 = None
        mm_17: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_74, permute_33)
        view_75: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_17, [8, 1024, 512]);  mm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_12: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 12)
        inductor_random_default_51: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_12, 'rand');  inductor_lookup_seed_default_12 = None
        convert_element_type_default_84: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_51, torch.bfloat16);  inductor_random_default_51 = None
        gt_13: "b8[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_84, 0.1);  convert_element_type_default_84 = None
        mul_38: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_13, view_75);  view_75 = None
        mul_39: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, 1.1111111111111112);  mul_38 = None
        add_21: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_19, mul_39);  mul_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_7: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_21, 2)
        mean_6: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_22: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_6, 1e-06);  mean_6 = None
        rsqrt_6: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        mul_40: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_21, rsqrt_6)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_41: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_28, mul_40);  mul_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_96: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_29, torch.bfloat16);  primals_29 = None
        convert_element_type_97: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_41, torch.bfloat16);  mul_41 = None
        permute_34: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_96, [1, 0]);  convert_element_type_96 = None
        view_76: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_97, [8192, 512]);  convert_element_type_97 = None
        mm_18: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_76, permute_34)
        view_77: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_18, [8, 1024, 512]);  mm_18 = None
        view_78: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_77, [8, 1024, -1, 64]);  view_77 = None
        permute_35: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_100: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_30, torch.bfloat16);  primals_30 = None
        permute_36: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_100, [1, 0]);  convert_element_type_100 = None
        mm_19: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_76, permute_36)
        view_80: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_19, [8, 1024, 512]);  mm_19 = None
        view_81: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_80, [8, 1024, -1, 64]);  view_80 = None
        permute_37: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_81, [0, 2, 1, 3]);  view_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_104: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_31, torch.bfloat16);  primals_31 = None
        permute_38: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_104, [1, 0]);  convert_element_type_104 = None
        mm_20: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_76, permute_38)
        view_83: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_20, [8, 1024, 512]);  mm_20 = None
        view_84: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_83, [8, 1024, -1, 64]);  view_83 = None
        permute_39: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_40: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_37, [0, 1, 3, 2]);  permute_37 = None
        expand_13: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_35, [8, 8, 1024, 64]);  permute_35 = None
        clone_12: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_85: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_12, [64, 1024, 64]);  clone_12 = None
        expand_14: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_40, [8, 8, 64, 1024]);  permute_40 = None
        clone_13: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_86: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_13, [64, 64, 1024]);  clone_13 = None
        bmm_6: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_85, view_86)
        view_87: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_6, [8, 8, 1024, 1024]);  bmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_23: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_87, add_7);  view_87 = None
        convert_element_type_110: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_23, torch.bfloat16);  add_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_111: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_110, torch.float32)
        amax_3: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_111, [-1], True)
        sub_4: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_111, amax_3);  convert_element_type_111 = None
        exp_3: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_4: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_5: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = None
        convert_element_type_112: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_13: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 13)
        inductor_random_default_50: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_13, 'rand');  inductor_lookup_seed_default_13 = None
        convert_element_type_default_83: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_50, torch.bfloat16);  inductor_random_default_50 = None
        gt_14: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_83, 0.1);  convert_element_type_default_83 = None
        mul_42: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_14, convert_element_type_112);  convert_element_type_112 = None
        mul_43: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, 1.1111111111111112);  mul_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_15: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_43, [8, 8, 1024, 1024]);  mul_43 = None
        view_91: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_15, [64, 1024, 1024]);  expand_15 = None
        expand_16: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_39, [8, 8, 1024, 64]);  permute_39 = None
        clone_14: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_92: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_14, [64, 1024, 64]);  clone_14 = None
        bmm_7: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_91, view_92)
        view_93: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_7, [8, 8, 1024, 64]);  bmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_41: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None
        clone_15: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_94: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_15, [8, 1024, -1]);  clone_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_115: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_32, torch.bfloat16);  primals_32 = None
        permute_42: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_115, [1, 0]);  convert_element_type_115 = None
        view_95: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_94, [8192, 512]);  view_94 = None
        mm_21: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_95, permute_42)
        view_96: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_21, [8, 1024, 512]);  mm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_14: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 14)
        inductor_random_default_49: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_14, 'rand');  inductor_lookup_seed_default_14 = None
        convert_element_type_default_82: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_49, torch.bfloat16);  inductor_random_default_49 = None
        gt_15: "b8[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_82, 0.1);  convert_element_type_default_82 = None
        mul_44: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_15, view_96);  view_96 = None
        mul_45: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_44, 1.1111111111111112);  mul_44 = None
        add_24: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_21, mul_45);  mul_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_8: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_24, 2)
        mean_7: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_8, [-1], True);  pow_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_25: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_7, 1e-06);  mean_7 = None
        rsqrt_7: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        mul_46: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_24, rsqrt_7)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_47: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_33, mul_46);  mul_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        convert_element_type_118: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_34, torch.bfloat16);  primals_34 = None
        convert_element_type_119: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_47, torch.bfloat16);  mul_47 = None
        permute_43: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_118, [1, 0]);  convert_element_type_118 = None
        view_97: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_119, [8192, 512]);  convert_element_type_119 = None
        mm_22: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_97, permute_43)
        view_98: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_22, [8, 1024, 2048]);  mm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_3: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.relu.default(view_98);  view_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_15: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 15)
        inductor_random_default_48: "f32[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 2048], inductor_lookup_seed_default_15, 'rand');  inductor_lookup_seed_default_15 = None
        convert_element_type_default_81: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_48, torch.bfloat16);  inductor_random_default_48 = None
        gt_16: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_81, 0.1);  convert_element_type_default_81 = None
        mul_48: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_16, relu_3)
        mul_49: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_48, 1.1111111111111112);  mul_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        convert_element_type_123: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_35, torch.bfloat16);  primals_35 = None
        convert_element_type_default_32: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_49, torch.bfloat16);  mul_49 = None
        permute_44: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_123, [1, 0]);  convert_element_type_123 = None
        view_99: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_default_32, [8192, 2048]);  convert_element_type_default_32 = None
        mm_23: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_99, permute_44)
        view_100: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_23, [8, 1024, 512]);  mm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_16: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 16)
        inductor_random_default_47: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_16, 'rand');  inductor_lookup_seed_default_16 = None
        convert_element_type_default_80: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_47, torch.bfloat16);  inductor_random_default_47 = None
        gt_17: "b8[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_80, 0.1);  convert_element_type_default_80 = None
        mul_50: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_17, view_100);  view_100 = None
        mul_51: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_50, 1.1111111111111112);  mul_50 = None
        add_26: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_24, mul_51);  mul_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_9: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_26, 2)
        mean_8: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_9, [-1], True);  pow_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_27: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_8, 1e-06);  mean_8 = None
        rsqrt_8: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        mul_52: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_26, rsqrt_8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_53: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_36, mul_52);  mul_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_127: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_37, torch.bfloat16);  primals_37 = None
        convert_element_type_128: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_53, torch.bfloat16);  mul_53 = None
        permute_45: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_127, [1, 0]);  convert_element_type_127 = None
        view_101: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_128, [8192, 512]);  convert_element_type_128 = None
        mm_24: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_101, permute_45)
        view_102: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_24, [8, 1024, 512]);  mm_24 = None
        view_103: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_102, [8, 1024, -1, 64]);  view_102 = None
        permute_46: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_103, [0, 2, 1, 3]);  view_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_131: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_38, torch.bfloat16);  primals_38 = None
        permute_47: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_131, [1, 0]);  convert_element_type_131 = None
        mm_25: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_101, permute_47)
        view_105: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_25, [8, 1024, 512]);  mm_25 = None
        view_106: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_105, [8, 1024, -1, 64]);  view_105 = None
        permute_48: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_135: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_39, torch.bfloat16);  primals_39 = None
        permute_49: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_135, [1, 0]);  convert_element_type_135 = None
        mm_26: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_101, permute_49)
        view_108: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_26, [8, 1024, 512]);  mm_26 = None
        view_109: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_108, [8, 1024, -1, 64]);  view_108 = None
        permute_50: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_109, [0, 2, 1, 3]);  view_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_51: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_48, [0, 1, 3, 2]);  permute_48 = None
        expand_17: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_46, [8, 8, 1024, 64]);  permute_46 = None
        clone_16: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_110: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_16, [64, 1024, 64]);  clone_16 = None
        expand_18: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_51, [8, 8, 64, 1024]);  permute_51 = None
        clone_17: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_111: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_17, [64, 64, 1024]);  clone_17 = None
        bmm_8: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_110, view_111)
        view_112: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_8, [8, 8, 1024, 1024]);  bmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_28: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_112, add_7);  view_112 = None
        convert_element_type_141: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_28, torch.bfloat16);  add_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_142: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_141, torch.float32)
        amax_4: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_142, [-1], True)
        sub_5: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_142, amax_4);  convert_element_type_142 = None
        exp_4: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        sum_5: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_6: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = None
        convert_element_type_143: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_17: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 17)
        inductor_random_default_46: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_17, 'rand');  inductor_lookup_seed_default_17 = None
        convert_element_type_default_79: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_46, torch.bfloat16);  inductor_random_default_46 = None
        gt_18: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_79, 0.1);  convert_element_type_default_79 = None
        mul_54: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_18, convert_element_type_143);  convert_element_type_143 = None
        mul_55: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_54, 1.1111111111111112);  mul_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_19: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_55, [8, 8, 1024, 1024]);  mul_55 = None
        view_116: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_19, [64, 1024, 1024]);  expand_19 = None
        expand_20: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_50, [8, 8, 1024, 64]);  permute_50 = None
        clone_18: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_117: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_18, [64, 1024, 64]);  clone_18 = None
        bmm_9: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_116, view_117)
        view_118: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_9, [8, 8, 1024, 64]);  bmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_52: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
        clone_19: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_119: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_19, [8, 1024, -1]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_146: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_40, torch.bfloat16);  primals_40 = None
        permute_53: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_146, [1, 0]);  convert_element_type_146 = None
        view_120: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_119, [8192, 512]);  view_119 = None
        mm_27: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_120, permute_53)
        view_121: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_27, [8, 1024, 512]);  mm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_18: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 18)
        inductor_random_default_45: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_18, 'rand');  inductor_lookup_seed_default_18 = None
        convert_element_type_default_78: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_45, torch.bfloat16);  inductor_random_default_45 = None
        gt_19: "b8[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_78, 0.1);  convert_element_type_default_78 = None
        mul_56: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_19, view_121);  view_121 = None
        mul_57: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_56, 1.1111111111111112);  mul_56 = None
        add_29: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_26, mul_57);  mul_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_10: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_29, 2)
        mean_9: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_30: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_9, 1e-06);  mean_9 = None
        rsqrt_9: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_58: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_29, rsqrt_9)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_59: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_41, mul_58);  mul_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        convert_element_type_149: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_42, torch.bfloat16);  primals_42 = None
        convert_element_type_150: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_59, torch.bfloat16);  mul_59 = None
        permute_54: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_149, [1, 0]);  convert_element_type_149 = None
        view_122: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_150, [8192, 512]);  convert_element_type_150 = None
        mm_28: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_122, permute_54)
        view_123: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_28, [8, 1024, 2048]);  mm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_4: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.relu.default(view_123);  view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_19: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 19)
        inductor_random_default_44: "f32[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 2048], inductor_lookup_seed_default_19, 'rand');  inductor_lookup_seed_default_19 = None
        convert_element_type_default_77: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_44, torch.bfloat16);  inductor_random_default_44 = None
        gt_20: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_77, 0.1);  convert_element_type_default_77 = None
        mul_60: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_20, relu_4)
        mul_61: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_60, 1.1111111111111112);  mul_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        convert_element_type_154: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_43, torch.bfloat16);  primals_43 = None
        convert_element_type_default_31: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_61, torch.bfloat16);  mul_61 = None
        permute_55: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_154, [1, 0]);  convert_element_type_154 = None
        view_124: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_default_31, [8192, 2048]);  convert_element_type_default_31 = None
        mm_29: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_124, permute_55)
        view_125: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_29, [8, 1024, 512]);  mm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_20: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 20)
        inductor_random_default_43: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_20, 'rand');  inductor_lookup_seed_default_20 = None
        convert_element_type_default_76: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_43, torch.bfloat16);  inductor_random_default_43 = None
        gt_21: "b8[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_76, 0.1);  convert_element_type_default_76 = None
        mul_62: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_21, view_125);  view_125 = None
        mul_63: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_62, 1.1111111111111112);  mul_62 = None
        add_31: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_29, mul_63);  mul_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_11: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_31, 2)
        mean_10: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_32: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_10, 1e-06);  mean_10 = None
        rsqrt_10: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        mul_64: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_31, rsqrt_10)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_65: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_44, mul_64);  mul_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_158: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_45, torch.bfloat16);  primals_45 = None
        convert_element_type_159: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_65, torch.bfloat16);  mul_65 = None
        permute_56: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_158, [1, 0]);  convert_element_type_158 = None
        view_126: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_159, [8192, 512]);  convert_element_type_159 = None
        mm_30: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_126, permute_56)
        view_127: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_30, [8, 1024, 512]);  mm_30 = None
        view_128: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_127, [8, 1024, -1, 64]);  view_127 = None
        permute_57: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_128, [0, 2, 1, 3]);  view_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_162: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_46, torch.bfloat16);  primals_46 = None
        permute_58: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_162, [1, 0]);  convert_element_type_162 = None
        mm_31: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_126, permute_58)
        view_130: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_31, [8, 1024, 512]);  mm_31 = None
        view_131: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_130, [8, 1024, -1, 64]);  view_130 = None
        permute_59: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_131, [0, 2, 1, 3]);  view_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_166: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_47, torch.bfloat16);  primals_47 = None
        permute_60: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_166, [1, 0]);  convert_element_type_166 = None
        mm_32: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_126, permute_60)
        view_133: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_32, [8, 1024, 512]);  mm_32 = None
        view_134: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_133, [8, 1024, -1, 64]);  view_133 = None
        permute_61: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_62: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_59, [0, 1, 3, 2]);  permute_59 = None
        expand_21: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_57, [8, 8, 1024, 64]);  permute_57 = None
        clone_20: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_135: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_20, [64, 1024, 64]);  clone_20 = None
        expand_22: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_62, [8, 8, 64, 1024]);  permute_62 = None
        clone_21: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_136: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_21, [64, 64, 1024]);  clone_21 = None
        bmm_10: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_135, view_136)
        view_137: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_10, [8, 8, 1024, 1024]);  bmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_33: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_137, add_7);  view_137 = add_7 = None
        convert_element_type_172: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_33, torch.bfloat16);  add_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_173: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_172, torch.float32)
        amax_5: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_173, [-1], True)
        sub_6: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_173, amax_5);  convert_element_type_173 = None
        exp_5: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_6: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_7: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = None
        convert_element_type_174: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_21: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 21)
        inductor_random_default_42: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_21, 'rand');  inductor_lookup_seed_default_21 = None
        convert_element_type_default_75: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_42, torch.bfloat16);  inductor_random_default_42 = None
        gt_22: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_75, 0.1);  convert_element_type_default_75 = None
        mul_66: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_22, convert_element_type_174);  convert_element_type_174 = None
        mul_67: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, 1.1111111111111112);  mul_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_23: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_67, [8, 8, 1024, 1024]);  mul_67 = None
        view_141: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_23, [64, 1024, 1024]);  expand_23 = None
        expand_24: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_61, [8, 8, 1024, 64]);  permute_61 = None
        clone_22: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_142: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_22, [64, 1024, 64]);  clone_22 = None
        bmm_11: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_141, view_142)
        view_143: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_11, [8, 8, 1024, 64]);  bmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_63: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_143, [0, 2, 1, 3]);  view_143 = None
        clone_23: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_144: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_23, [8, 1024, -1]);  clone_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_177: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_48, torch.bfloat16);  primals_48 = None
        permute_64: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_177, [1, 0]);  convert_element_type_177 = None
        view_145: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_144, [8192, 512]);  view_144 = None
        mm_33: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_145, permute_64)
        view_146: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_33, [8, 1024, 512]);  mm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_22: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 22)
        inductor_random_default_41: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_22, 'rand');  inductor_lookup_seed_default_22 = None
        convert_element_type_default_74: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_41, torch.bfloat16);  inductor_random_default_41 = None
        gt_23: "b8[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_74, 0.1);  convert_element_type_default_74 = None
        mul_68: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_23, view_146);  view_146 = None
        mul_69: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, 1.1111111111111112);  mul_68 = None
        add_34: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_31, mul_69);  mul_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_12: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_34, 2)
        mean_11: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_12, [-1], True);  pow_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_35: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_11, 1e-06);  mean_11 = None
        rsqrt_11: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        mul_70: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_34, rsqrt_11)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_71: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_49, mul_70);  mul_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        convert_element_type_180: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_50, torch.bfloat16);  primals_50 = None
        convert_element_type_181: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_71, torch.bfloat16);  mul_71 = None
        permute_65: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_180, [1, 0]);  convert_element_type_180 = None
        view_147: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_181, [8192, 512]);  convert_element_type_181 = None
        mm_34: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_147, permute_65)
        view_148: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_34, [8, 1024, 2048]);  mm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_5: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.relu.default(view_148);  view_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_23: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 23)
        inductor_random_default_40: "f32[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 2048], inductor_lookup_seed_default_23, 'rand');  inductor_lookup_seed_default_23 = None
        convert_element_type_default_73: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_40, torch.bfloat16);  inductor_random_default_40 = None
        gt_24: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_73, 0.1);  convert_element_type_default_73 = None
        mul_72: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_24, relu_5)
        mul_73: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_72, 1.1111111111111112);  mul_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        convert_element_type_185: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_51, torch.bfloat16);  primals_51 = None
        convert_element_type_default_30: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_73, torch.bfloat16);  mul_73 = None
        permute_66: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_185, [1, 0]);  convert_element_type_185 = None
        view_149: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_default_30, [8192, 2048]);  convert_element_type_default_30 = None
        mm_35: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_149, permute_66)
        view_150: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_35, [8, 1024, 512]);  mm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_24: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 24)
        inductor_random_default_39: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_24, 'rand');  inductor_lookup_seed_default_24 = None
        convert_element_type_default_72: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_39, torch.bfloat16);  inductor_random_default_39 = None
        gt_25: "b8[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_72, 0.1);  convert_element_type_default_72 = None
        mul_74: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_25, view_150);  view_150 = None
        mul_75: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_74, 1.1111111111111112);  mul_74 = None
        add_36: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_34, mul_75);  mul_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_13: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_36, 2)
        mean_12: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_37: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_12, 1e-06);  mean_12 = None
        rsqrt_12: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        mul_76: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_36, rsqrt_12)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_77: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_52, mul_76);  mul_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:755 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_25: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 25)
        inductor_random_default_38: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_25, 'rand');  inductor_lookup_seed_default_25 = None
        gt_26: "b8[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_38, 0.1);  inductor_random_default_38 = None
        mul_78: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_26, mul_77);  mul_77 = None
        mul_79: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_78, 1.1111111111111112);  mul_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        add_38: "i64[1024][1]cuda:0" = torch.ops.aten.add.Tensor(iota_4, 0);  iota_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_6: "i64[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_38, 0);  add_38 = None
        unsqueeze_7: "i64[1, 1, 1024][1024, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 1);  unsqueeze_6 = None
        unsqueeze_8: "i64[1, 1, 1024, 1][1024, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_11: "i64[1, 1, 1, 1024][1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 1024, 1024][1048576, 1048576, 1024, 1]cuda:0" = torch.ops.aten.le.Tensor(unsqueeze_11, unsqueeze_8);  unsqueeze_11 = unsqueeze_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_25: "b8[8, 1, 1024, 1024][0, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(le, [8, -1, 1024, 1024]);  le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default_1: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[8, 1, 1024, 1024][1048576, 1048576, 1024, 1]cuda:0" = torch.ops.aten.where.self(expand_25, full_default_1, full_default_2);  expand_25 = full_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:721 in forward, code: hidden_states = self.dropout(inputs_embeds)
        inductor_lookup_seed_default_26: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 26)
        inductor_random_default_37: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_26, 'rand');  inductor_lookup_seed_default_26 = None
        gt_27: "b8[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_37, 0.1);  inductor_random_default_37 = None
        mul_80: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_27, embedding)
        mul_81: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_80, 1.1111111111111112);  mul_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_14: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(mul_81, 2)
        mean_13: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_14, [-1], True);  pow_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_42: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_13, 1e-06);  mean_13 = None
        rsqrt_13: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_82: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_81, rsqrt_13)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_83: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_54, mul_82);  mul_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_189: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_55, torch.bfloat16);  primals_55 = None
        convert_element_type_190: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_83, torch.bfloat16);  mul_83 = None
        permute_67: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_189, [1, 0]);  convert_element_type_189 = None
        view_152: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_190, [8192, 512]);  convert_element_type_190 = None
        mm_36: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_152, permute_67)
        view_153: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_36, [8, 1024, 512]);  mm_36 = None
        view_154: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_153, [8, 1024, -1, 64]);  view_153 = None
        permute_68: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_154, [0, 2, 1, 3]);  view_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_193: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_56, torch.bfloat16);  primals_56 = None
        permute_69: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_193, [1, 0]);  convert_element_type_193 = None
        mm_37: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_152, permute_69)
        view_156: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_37, [8, 1024, 512]);  mm_37 = None
        view_157: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_156, [8, 1024, -1, 64]);  view_156 = None
        permute_70: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_157, [0, 2, 1, 3]);  view_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_197: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_57, torch.bfloat16);  primals_57 = None
        permute_71: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_197, [1, 0]);  convert_element_type_197 = None
        mm_38: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_152, permute_71)
        view_159: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_38, [8, 1024, 512]);  mm_38 = None
        view_160: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_159, [8, 1024, -1, 64]);  view_159 = None
        permute_72: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_73: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_70, [0, 1, 3, 2]);  permute_70 = None
        expand_27: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_68, [8, 8, 1024, 64]);  permute_68 = None
        clone_24: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_161: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_24, [64, 1024, 64]);  clone_24 = None
        expand_28: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_73, [8, 8, 64, 1024]);  permute_73 = None
        clone_25: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_162: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_25, [64, 64, 1024]);  clone_25 = None
        bmm_12: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_161, view_162)
        view_163: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_12, [8, 8, 1024, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:203 in _relative_position_bucket, code: relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        full_1: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.full.default([1024, 1024], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_1: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.minimum.default(sub, full_1);  sub = full_1 = None
        neg: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.neg.default(minimum_1);  minimum_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:208 in _relative_position_bucket, code: is_small = relative_position < max_exact
        lt_1: "b8[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.lt.Scalar(neg, 16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:212 in _relative_position_bucket, code: torch.log(relative_position.float() / max_exact)
        convert_element_type_203: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(neg, torch.float32)
        div_8: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_203, 16);  convert_element_type_203 = None
        log_1: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.log.default(div_8);  div_8 = None
        div_9: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.div.Tensor(log_1, 2.0794415416798357);  log_1 = None
        mul_84: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_9, 16);  div_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:215 in _relative_position_bucket, code: ).to(torch.long)
        convert_element_type_204: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_84, torch.int64);  mul_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:211 in _relative_position_bucket, code: relative_position_if_large = max_exact + (
        add_44: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_204, 16);  convert_element_type_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:217 in _relative_position_bucket, code: relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        full_2: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.full.default([1024, 1024], 31, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:216 in _relative_position_bucket, code: relative_position_if_large = torch.min(
        minimum_2: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.minimum.default(add_44, full_2);  add_44 = full_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:220 in _relative_position_bucket, code: relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        where_4: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.where.self(lt_1, neg, minimum_2);  lt_1 = neg = minimum_2 = None
        add_45: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.add.Tensor(where_4, 0);  where_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:236 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        embedding_3: "f32[1024, 1024, 8][8192, 8, 1]cuda:0" = torch.ops.aten.embedding.default(primals_58, add_45);  primals_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:237 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        permute_74: "f32[8, 1024, 1024][1, 8192, 8]cuda:0" = torch.ops.aten.permute.default(embedding_3, [2, 0, 1])
        unsqueeze_17: "f32[1, 8, 1024, 1024][8, 1, 8192, 8]cuda:0" = torch.ops.aten.unsqueeze.default(permute_74, 0);  permute_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        add_46: "f32[8, 8, 1024, 1024][8388608, 1, 8192, 8]cuda:0" = torch.ops.aten.add.Tensor(unsqueeze_17, where_2);  unsqueeze_17 = where_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_47: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_163, add_46);  view_163 = None
        convert_element_type_205: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_47, torch.bfloat16);  add_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_206: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_205, torch.float32);  convert_element_type_205 = None
        amax_6: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_206, [-1], True)
        sub_8: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_206, amax_6);  convert_element_type_206 = None
        exp_6: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
        sum_7: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_10: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = None
        convert_element_type_207: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_27: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 27)
        inductor_random_default_36: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_27, 'rand');  inductor_lookup_seed_default_27 = None
        convert_element_type_default_71: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_36, torch.bfloat16);  inductor_random_default_36 = None
        gt_28: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_71, 0.1);  convert_element_type_default_71 = None
        mul_85: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_28, convert_element_type_207);  convert_element_type_207 = None
        mul_86: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_85, 1.1111111111111112);  mul_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_29: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_86, [8, 8, 1024, 1024]);  mul_86 = None
        view_167: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_29, [64, 1024, 1024]);  expand_29 = None
        expand_30: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_72, [8, 8, 1024, 64]);  permute_72 = None
        clone_26: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_168: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_26, [64, 1024, 64]);  clone_26 = None
        bmm_13: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_167, view_168)
        view_169: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_13, [8, 8, 1024, 64]);  bmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_75: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_169, [0, 2, 1, 3]);  view_169 = None
        clone_27: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_75, memory_format = torch.contiguous_format);  permute_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_170: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_27, [8, 1024, -1]);  clone_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_210: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_59, torch.bfloat16);  primals_59 = None
        permute_76: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_210, [1, 0]);  convert_element_type_210 = None
        view_171: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_170, [8192, 512]);  view_170 = None
        mm_39: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_171, permute_76)
        view_172: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_39, [8, 1024, 512]);  mm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_28: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 28)
        inductor_random_default_35: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_28, 'rand');  inductor_lookup_seed_default_28 = None
        convert_element_type_default_70: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_35, torch.bfloat16);  inductor_random_default_35 = None
        gt_29: "b8[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_70, 0.1);  convert_element_type_default_70 = None
        mul_87: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_29, view_172);  view_172 = None
        mul_88: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_87, 1.1111111111111112);  mul_87 = None
        add_48: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_81, mul_88);  mul_81 = mul_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_15: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_48, 2)
        mean_14: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_15, [-1], True);  pow_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_49: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_14, 1e-06);  mean_14 = None
        rsqrt_14: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        mul_89: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_48, rsqrt_14)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_90: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_60, mul_89);  mul_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_213: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_61, torch.bfloat16);  primals_61 = None
        convert_element_type_214: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_90, torch.bfloat16);  mul_90 = None
        permute_77: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_213, [1, 0]);  convert_element_type_213 = None
        view_173: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_214, [8192, 512]);  convert_element_type_214 = None
        mm_40: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_173, permute_77)
        view_174: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_40, [8, 1024, 512]);  mm_40 = None
        view_175: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_174, [8, 1024, -1, 64]);  view_174 = None
        permute_78: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_175, [0, 2, 1, 3]);  view_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_217: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_62, torch.bfloat16);  primals_62 = None
        convert_element_type_218: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_79, torch.bfloat16)
        permute_79: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_217, [1, 0]);  convert_element_type_217 = None
        view_176: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_218, [8192, 512]);  convert_element_type_218 = None
        mm_41: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_176, permute_79)
        view_177: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_41, [8, 1024, 512]);  mm_41 = None
        view_178: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_177, [8, 1024, -1, 64]);  view_177 = None
        permute_80: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_221: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_63, torch.bfloat16);  primals_63 = None
        permute_81: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_221, [1, 0]);  convert_element_type_221 = None
        mm_42: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_176, permute_81)
        view_180: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_42, [8, 1024, 512]);  mm_42 = None
        view_181: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_180, [8, 1024, -1, 64]);  view_180 = None
        permute_82: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_83: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_80, [0, 1, 3, 2]);  permute_80 = None
        expand_31: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_78, [8, 8, 1024, 64]);  permute_78 = None
        clone_28: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_182: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_28, [64, 1024, 64]);  clone_28 = None
        expand_32: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_83, [8, 8, 64, 1024]);  permute_83 = None
        clone_29: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_183: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_29, [64, 64, 1024]);  clone_29 = None
        bmm_14: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_182, view_183)
        view_184: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_14, [8, 8, 1024, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        convert_element_type_default_29: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_184, torch.bfloat16);  view_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_228: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_default_29, torch.float32);  convert_element_type_default_29 = None
        amax_7: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_228, [-1], True)
        sub_9: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_228, amax_7);  convert_element_type_228 = None
        exp_7: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        sum_8: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_11: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = None
        convert_element_type_229: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_29: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 29)
        inductor_random_default_34: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_29, 'rand');  inductor_lookup_seed_default_29 = None
        convert_element_type_default_69: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_34, torch.bfloat16);  inductor_random_default_34 = None
        gt_30: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_69, 0.1);  convert_element_type_default_69 = None
        mul_91: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_30, convert_element_type_229);  convert_element_type_229 = None
        mul_92: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_91, 1.1111111111111112);  mul_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_33: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_92, [8, 8, 1024, 1024]);  mul_92 = None
        view_188: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_33, [64, 1024, 1024]);  expand_33 = None
        expand_34: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_82, [8, 8, 1024, 64]);  permute_82 = None
        clone_30: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_189: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_30, [64, 1024, 64]);  clone_30 = None
        bmm_15: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_188, view_189)
        view_190: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_15, [8, 8, 1024, 64]);  bmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_84: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
        clone_31: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_191: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_31, [8, 1024, -1]);  clone_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_232: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_64, torch.bfloat16);  primals_64 = None
        permute_85: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_232, [1, 0]);  convert_element_type_232 = None
        view_192: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_191, [8192, 512]);  view_191 = None
        mm_43: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_192, permute_85)
        view_193: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_43, [8, 1024, 512]);  mm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_30: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 30)
        inductor_random_default_33: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_30, 'rand');  inductor_lookup_seed_default_30 = None
        convert_element_type_default_68: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_33, torch.bfloat16);  inductor_random_default_33 = None
        gt_31: "b8[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_68, 0.1);  convert_element_type_default_68 = None
        mul_93: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_31, view_193);  view_193 = None
        mul_94: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_93, 1.1111111111111112);  mul_93 = None
        add_52: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_48, mul_94);  mul_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_16: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_52, 2)
        mean_15: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_53: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_15, 1e-06);  mean_15 = None
        rsqrt_15: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        mul_95: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_52, rsqrt_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_96: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_65, mul_95);  mul_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        convert_element_type_235: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_66, torch.bfloat16);  primals_66 = None
        convert_element_type_236: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_96, torch.bfloat16);  mul_96 = None
        permute_86: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_235, [1, 0]);  convert_element_type_235 = None
        view_194: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_236, [8192, 512]);  convert_element_type_236 = None
        mm_44: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_194, permute_86)
        view_195: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_44, [8, 1024, 2048]);  mm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_6: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.relu.default(view_195);  view_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_31: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 31)
        inductor_random_default_32: "f32[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 2048], inductor_lookup_seed_default_31, 'rand');  inductor_lookup_seed_default_31 = None
        convert_element_type_default_67: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_32, torch.bfloat16);  inductor_random_default_32 = None
        gt_32: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_67, 0.1);  convert_element_type_default_67 = None
        mul_97: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_32, relu_6)
        mul_98: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_97, 1.1111111111111112);  mul_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        convert_element_type_240: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_67, torch.bfloat16);  primals_67 = None
        convert_element_type_default_28: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_98, torch.bfloat16);  mul_98 = None
        permute_87: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_240, [1, 0]);  convert_element_type_240 = None
        view_196: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_default_28, [8192, 2048]);  convert_element_type_default_28 = None
        mm_45: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_196, permute_87)
        view_197: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_45, [8, 1024, 512]);  mm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_32: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 32)
        inductor_random_default_31: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_32, 'rand');  inductor_lookup_seed_default_32 = None
        convert_element_type_default_66: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_31, torch.bfloat16);  inductor_random_default_31 = None
        gt_33: "b8[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_66, 0.1);  convert_element_type_default_66 = None
        mul_99: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_33, view_197);  view_197 = None
        mul_100: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_99, 1.1111111111111112);  mul_99 = None
        add_54: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_52, mul_100);  mul_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_17: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_54, 2)
        mean_16: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_55: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_16, 1e-06);  mean_16 = None
        rsqrt_16: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
        mul_101: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_54, rsqrt_16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_102: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_68, mul_101);  mul_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_244: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_69, torch.bfloat16);  primals_69 = None
        convert_element_type_245: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_102, torch.bfloat16);  mul_102 = None
        permute_88: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_244, [1, 0]);  convert_element_type_244 = None
        view_198: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_245, [8192, 512]);  convert_element_type_245 = None
        mm_46: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_198, permute_88)
        view_199: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_46, [8, 1024, 512]);  mm_46 = None
        view_200: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_199, [8, 1024, -1, 64]);  view_199 = None
        permute_89: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_248: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_70, torch.bfloat16);  primals_70 = None
        permute_90: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_248, [1, 0]);  convert_element_type_248 = None
        mm_47: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_198, permute_90)
        view_202: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_47, [8, 1024, 512]);  mm_47 = None
        view_203: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_202, [8, 1024, -1, 64]);  view_202 = None
        permute_91: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_252: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_71, torch.bfloat16);  primals_71 = None
        permute_92: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_252, [1, 0]);  convert_element_type_252 = None
        mm_48: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_198, permute_92)
        view_205: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_48, [8, 1024, 512]);  mm_48 = None
        view_206: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_205, [8, 1024, -1, 64]);  view_205 = None
        permute_93: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_94: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_91, [0, 1, 3, 2]);  permute_91 = None
        expand_35: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_89, [8, 8, 1024, 64]);  permute_89 = None
        clone_32: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_207: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_32, [64, 1024, 64]);  clone_32 = None
        expand_36: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_94, [8, 8, 64, 1024]);  permute_94 = None
        clone_33: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_208: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_33, [64, 64, 1024]);  clone_33 = None
        bmm_16: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_207, view_208)
        view_209: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_16, [8, 8, 1024, 1024]);  bmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_56: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_209, add_46);  view_209 = None
        convert_element_type_258: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_56, torch.bfloat16);  add_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_259: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_258, torch.float32)
        amax_8: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_259, [-1], True)
        sub_10: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_259, amax_8);  convert_element_type_259 = None
        exp_8: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_9: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_12: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = None
        convert_element_type_260: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_12, torch.bfloat16);  div_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_33: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 33)
        inductor_random_default_30: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_33, 'rand');  inductor_lookup_seed_default_33 = None
        convert_element_type_default_65: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_30, torch.bfloat16);  inductor_random_default_30 = None
        gt_34: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_65, 0.1);  convert_element_type_default_65 = None
        mul_103: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_34, convert_element_type_260);  convert_element_type_260 = None
        mul_104: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_103, 1.1111111111111112);  mul_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_37: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_104, [8, 8, 1024, 1024]);  mul_104 = None
        view_213: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_37, [64, 1024, 1024]);  expand_37 = None
        expand_38: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_93, [8, 8, 1024, 64]);  permute_93 = None
        clone_34: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_214: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_34, [64, 1024, 64]);  clone_34 = None
        bmm_17: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_213, view_214)
        view_215: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_17, [8, 8, 1024, 64]);  bmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_95: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_215, [0, 2, 1, 3]);  view_215 = None
        clone_35: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_216: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_35, [8, 1024, -1]);  clone_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_263: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_72, torch.bfloat16);  primals_72 = None
        permute_96: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_263, [1, 0]);  convert_element_type_263 = None
        view_217: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_216, [8192, 512]);  view_216 = None
        mm_49: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_217, permute_96)
        view_218: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_49, [8, 1024, 512]);  mm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_34: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 34)
        inductor_random_default_29: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_34, 'rand');  inductor_lookup_seed_default_34 = None
        convert_element_type_default_64: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_29, torch.bfloat16);  inductor_random_default_29 = None
        gt_35: "b8[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_64, 0.1);  convert_element_type_default_64 = None
        mul_105: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_35, view_218);  view_218 = None
        mul_106: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_105, 1.1111111111111112);  mul_105 = None
        add_57: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_54, mul_106);  mul_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_18: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_57, 2)
        mean_17: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_18, [-1], True);  pow_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_58: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_17, 1e-06);  mean_17 = None
        rsqrt_17: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_107: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_57, rsqrt_17)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_108: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_73, mul_107);  mul_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_266: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_74, torch.bfloat16);  primals_74 = None
        convert_element_type_267: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_108, torch.bfloat16);  mul_108 = None
        permute_97: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_266, [1, 0]);  convert_element_type_266 = None
        view_219: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_267, [8192, 512]);  convert_element_type_267 = None
        mm_50: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_219, permute_97)
        view_220: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_50, [8, 1024, 512]);  mm_50 = None
        view_221: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_220, [8, 1024, -1, 64]);  view_220 = None
        permute_98: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_221, [0, 2, 1, 3]);  view_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_270: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_75, torch.bfloat16);  primals_75 = None
        permute_99: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_270, [1, 0]);  convert_element_type_270 = None
        mm_51: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_176, permute_99)
        view_223: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_51, [8, 1024, 512]);  mm_51 = None
        view_224: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_223, [8, 1024, -1, 64]);  view_223 = None
        permute_100: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_274: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_76, torch.bfloat16);  primals_76 = None
        permute_101: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_274, [1, 0]);  convert_element_type_274 = None
        mm_52: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_176, permute_101)
        view_226: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_52, [8, 1024, 512]);  mm_52 = None
        view_227: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_226, [8, 1024, -1, 64]);  view_226 = None
        permute_102: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_103: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_100, [0, 1, 3, 2]);  permute_100 = None
        expand_39: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_98, [8, 8, 1024, 64]);  permute_98 = None
        clone_36: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_228: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_36, [64, 1024, 64]);  clone_36 = None
        expand_40: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_103, [8, 8, 64, 1024]);  permute_103 = None
        clone_37: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_229: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_37, [64, 64, 1024]);  clone_37 = None
        bmm_18: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_228, view_229)
        view_230: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_18, [8, 8, 1024, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        convert_element_type_default_27: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_230, torch.bfloat16);  view_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_281: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_default_27, torch.float32);  convert_element_type_default_27 = None
        amax_9: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_281, [-1], True)
        sub_11: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_281, amax_9);  convert_element_type_281 = None
        exp_9: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        sum_10: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_13: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = None
        convert_element_type_282: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_35: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 35)
        inductor_random_default_28: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_35, 'rand');  inductor_lookup_seed_default_35 = None
        convert_element_type_default_63: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_28, torch.bfloat16);  inductor_random_default_28 = None
        gt_36: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_63, 0.1);  convert_element_type_default_63 = None
        mul_109: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_36, convert_element_type_282);  convert_element_type_282 = None
        mul_110: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_109, 1.1111111111111112);  mul_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_41: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_110, [8, 8, 1024, 1024]);  mul_110 = None
        view_234: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_41, [64, 1024, 1024]);  expand_41 = None
        expand_42: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_102, [8, 8, 1024, 64]);  permute_102 = None
        clone_38: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_235: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_38, [64, 1024, 64]);  clone_38 = None
        bmm_19: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_234, view_235)
        view_236: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_19, [8, 8, 1024, 64]);  bmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_104: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
        clone_39: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_237: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_39, [8, 1024, -1]);  clone_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_285: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_77, torch.bfloat16);  primals_77 = None
        permute_105: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_285, [1, 0]);  convert_element_type_285 = None
        view_238: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_237, [8192, 512]);  view_237 = None
        mm_53: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_238, permute_105)
        view_239: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_53, [8, 1024, 512]);  mm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_36: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 36)
        inductor_random_default_27: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_36, 'rand');  inductor_lookup_seed_default_36 = None
        convert_element_type_default_62: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_27, torch.bfloat16);  inductor_random_default_27 = None
        gt_37: "b8[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_62, 0.1);  convert_element_type_default_62 = None
        mul_111: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_37, view_239);  view_239 = None
        mul_112: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_111, 1.1111111111111112);  mul_111 = None
        add_60: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_57, mul_112);  mul_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_19: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_60, 2)
        mean_18: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_61: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_18, 1e-06);  mean_18 = None
        rsqrt_18: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        mul_113: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_60, rsqrt_18)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_114: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_78, mul_113);  mul_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        convert_element_type_288: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_79, torch.bfloat16);  primals_79 = None
        convert_element_type_289: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_114, torch.bfloat16);  mul_114 = None
        permute_106: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_288, [1, 0]);  convert_element_type_288 = None
        view_240: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_289, [8192, 512]);  convert_element_type_289 = None
        mm_54: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_240, permute_106)
        view_241: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_54, [8, 1024, 2048]);  mm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_7: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.relu.default(view_241);  view_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_37: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 37)
        inductor_random_default_26: "f32[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 2048], inductor_lookup_seed_default_37, 'rand');  inductor_lookup_seed_default_37 = None
        convert_element_type_default_61: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_26, torch.bfloat16);  inductor_random_default_26 = None
        gt_38: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_61, 0.1);  convert_element_type_default_61 = None
        mul_115: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_38, relu_7)
        mul_116: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_115, 1.1111111111111112);  mul_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        convert_element_type_293: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_80, torch.bfloat16);  primals_80 = None
        convert_element_type_default_26: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_116, torch.bfloat16);  mul_116 = None
        permute_107: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_293, [1, 0]);  convert_element_type_293 = None
        view_242: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_default_26, [8192, 2048]);  convert_element_type_default_26 = None
        mm_55: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_242, permute_107)
        view_243: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_55, [8, 1024, 512]);  mm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_38: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 38)
        inductor_random_default_25: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_38, 'rand');  inductor_lookup_seed_default_38 = None
        convert_element_type_default_60: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_25, torch.bfloat16);  inductor_random_default_25 = None
        gt_39: "b8[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_60, 0.1);  convert_element_type_default_60 = None
        mul_117: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_39, view_243);  view_243 = None
        mul_118: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_117, 1.1111111111111112);  mul_117 = None
        add_62: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_60, mul_118);  mul_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_20: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_62, 2)
        mean_19: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_20, [-1], True);  pow_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_63: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_19, 1e-06);  mean_19 = None
        rsqrt_19: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
        mul_119: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_62, rsqrt_19)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_120: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_81, mul_119);  mul_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_297: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_82, torch.bfloat16);  primals_82 = None
        convert_element_type_298: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_120, torch.bfloat16);  mul_120 = None
        permute_108: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_297, [1, 0]);  convert_element_type_297 = None
        view_244: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_298, [8192, 512]);  convert_element_type_298 = None
        mm_56: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_244, permute_108)
        view_245: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_56, [8, 1024, 512]);  mm_56 = None
        view_246: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_245, [8, 1024, -1, 64]);  view_245 = None
        permute_109: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_301: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_83, torch.bfloat16);  primals_83 = None
        permute_110: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_301, [1, 0]);  convert_element_type_301 = None
        mm_57: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_244, permute_110)
        view_248: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_57, [8, 1024, 512]);  mm_57 = None
        view_249: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_248, [8, 1024, -1, 64]);  view_248 = None
        permute_111: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_305: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_84, torch.bfloat16);  primals_84 = None
        permute_112: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_305, [1, 0]);  convert_element_type_305 = None
        mm_58: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_244, permute_112)
        view_251: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_58, [8, 1024, 512]);  mm_58 = None
        view_252: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_251, [8, 1024, -1, 64]);  view_251 = None
        permute_113: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_114: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_111, [0, 1, 3, 2]);  permute_111 = None
        expand_43: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_109, [8, 8, 1024, 64]);  permute_109 = None
        clone_40: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_253: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_40, [64, 1024, 64]);  clone_40 = None
        expand_44: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_114, [8, 8, 64, 1024]);  permute_114 = None
        clone_41: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_254: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_41, [64, 64, 1024]);  clone_41 = None
        bmm_20: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_253, view_254)
        view_255: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_20, [8, 8, 1024, 1024]);  bmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_64: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_255, add_46);  view_255 = None
        convert_element_type_311: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_64, torch.bfloat16);  add_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_312: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_311, torch.float32)
        amax_10: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_312, [-1], True)
        sub_12: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_312, amax_10);  convert_element_type_312 = None
        exp_10: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_11: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_14: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = None
        convert_element_type_313: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_14, torch.bfloat16);  div_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_39: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 39)
        inductor_random_default_24: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_39, 'rand');  inductor_lookup_seed_default_39 = None
        convert_element_type_default_59: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_24, torch.bfloat16);  inductor_random_default_24 = None
        gt_40: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_59, 0.1);  convert_element_type_default_59 = None
        mul_121: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_40, convert_element_type_313);  convert_element_type_313 = None
        mul_122: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_121, 1.1111111111111112);  mul_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_45: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_122, [8, 8, 1024, 1024]);  mul_122 = None
        view_259: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_45, [64, 1024, 1024]);  expand_45 = None
        expand_46: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_113, [8, 8, 1024, 64]);  permute_113 = None
        clone_42: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_260: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_42, [64, 1024, 64]);  clone_42 = None
        bmm_21: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_259, view_260)
        view_261: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_21, [8, 8, 1024, 64]);  bmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_115: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_261, [0, 2, 1, 3]);  view_261 = None
        clone_43: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_262: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_43, [8, 1024, -1]);  clone_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_316: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_85, torch.bfloat16);  primals_85 = None
        permute_116: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_316, [1, 0]);  convert_element_type_316 = None
        view_263: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_262, [8192, 512]);  view_262 = None
        mm_59: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_263, permute_116)
        view_264: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_59, [8, 1024, 512]);  mm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_40: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 40)
        inductor_random_default_23: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_40, 'rand');  inductor_lookup_seed_default_40 = None
        convert_element_type_default_58: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_23, torch.bfloat16);  inductor_random_default_23 = None
        gt_41: "b8[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_58, 0.1);  convert_element_type_default_58 = None
        mul_123: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_41, view_264);  view_264 = None
        mul_124: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_123, 1.1111111111111112);  mul_123 = None
        add_65: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_62, mul_124);  mul_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_21: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_65, 2)
        mean_20: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_21, [-1], True);  pow_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_66: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_20, 1e-06);  mean_20 = None
        rsqrt_20: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        mul_125: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_65, rsqrt_20)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_126: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_86, mul_125);  mul_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_319: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_87, torch.bfloat16);  primals_87 = None
        convert_element_type_320: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_126, torch.bfloat16);  mul_126 = None
        permute_117: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_319, [1, 0]);  convert_element_type_319 = None
        view_265: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_320, [8192, 512]);  convert_element_type_320 = None
        mm_60: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_265, permute_117)
        view_266: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_60, [8, 1024, 512]);  mm_60 = None
        view_267: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_266, [8, 1024, -1, 64]);  view_266 = None
        permute_118: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_267, [0, 2, 1, 3]);  view_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_323: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_88, torch.bfloat16);  primals_88 = None
        permute_119: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_323, [1, 0]);  convert_element_type_323 = None
        mm_61: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_176, permute_119)
        view_269: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_61, [8, 1024, 512]);  mm_61 = None
        view_270: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_269, [8, 1024, -1, 64]);  view_269 = None
        permute_120: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_270, [0, 2, 1, 3]);  view_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_327: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_89, torch.bfloat16);  primals_89 = None
        permute_121: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_327, [1, 0]);  convert_element_type_327 = None
        mm_62: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_176, permute_121)
        view_272: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_62, [8, 1024, 512]);  mm_62 = None
        view_273: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_272, [8, 1024, -1, 64]);  view_272 = None
        permute_122: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_273, [0, 2, 1, 3]);  view_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_123: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_120, [0, 1, 3, 2]);  permute_120 = None
        expand_47: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_118, [8, 8, 1024, 64]);  permute_118 = None
        clone_44: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_274: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_44, [64, 1024, 64]);  clone_44 = None
        expand_48: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_123, [8, 8, 64, 1024]);  permute_123 = None
        clone_45: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_275: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_45, [64, 64, 1024]);  clone_45 = None
        bmm_22: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_274, view_275)
        view_276: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_22, [8, 8, 1024, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        convert_element_type_default_25: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_276, torch.bfloat16);  view_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_334: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_default_25, torch.float32);  convert_element_type_default_25 = None
        amax_11: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_334, [-1], True)
        sub_13: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_334, amax_11);  convert_element_type_334 = None
        exp_11: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_12: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_15: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = None
        convert_element_type_335: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_41: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 41)
        inductor_random_default_22: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_41, 'rand');  inductor_lookup_seed_default_41 = None
        convert_element_type_default_57: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_22, torch.bfloat16);  inductor_random_default_22 = None
        gt_42: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_57, 0.1);  convert_element_type_default_57 = None
        mul_127: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_42, convert_element_type_335);  convert_element_type_335 = None
        mul_128: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_127, 1.1111111111111112);  mul_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_49: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_128, [8, 8, 1024, 1024]);  mul_128 = None
        view_280: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_49, [64, 1024, 1024]);  expand_49 = None
        expand_50: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_122, [8, 8, 1024, 64]);  permute_122 = None
        clone_46: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_281: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_46, [64, 1024, 64]);  clone_46 = None
        bmm_23: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_280, view_281)
        view_282: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_23, [8, 8, 1024, 64]);  bmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_124: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_282, [0, 2, 1, 3]);  view_282 = None
        clone_47: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_283: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_47, [8, 1024, -1]);  clone_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_338: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_90, torch.bfloat16);  primals_90 = None
        permute_125: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_338, [1, 0]);  convert_element_type_338 = None
        view_284: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_283, [8192, 512]);  view_283 = None
        mm_63: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_284, permute_125)
        view_285: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_63, [8, 1024, 512]);  mm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_42: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 42)
        inductor_random_default_21: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_42, 'rand');  inductor_lookup_seed_default_42 = None
        convert_element_type_default_56: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_21, torch.bfloat16);  inductor_random_default_21 = None
        gt_43: "b8[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_56, 0.1);  convert_element_type_default_56 = None
        mul_129: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_43, view_285);  view_285 = None
        mul_130: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_129, 1.1111111111111112);  mul_129 = None
        add_68: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_65, mul_130);  mul_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_22: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_68, 2)
        mean_21: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_22, [-1], True);  pow_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_69: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_21, 1e-06);  mean_21 = None
        rsqrt_21: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        mul_131: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_68, rsqrt_21)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_132: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_91, mul_131);  mul_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        convert_element_type_341: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_92, torch.bfloat16);  primals_92 = None
        convert_element_type_342: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_132, torch.bfloat16);  mul_132 = None
        permute_126: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_341, [1, 0]);  convert_element_type_341 = None
        view_286: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_342, [8192, 512]);  convert_element_type_342 = None
        mm_64: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_286, permute_126)
        view_287: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_64, [8, 1024, 2048]);  mm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_8: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.relu.default(view_287);  view_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_43: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 43)
        inductor_random_default_20: "f32[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 2048], inductor_lookup_seed_default_43, 'rand');  inductor_lookup_seed_default_43 = None
        convert_element_type_default_55: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_20, torch.bfloat16);  inductor_random_default_20 = None
        gt_44: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_55, 0.1);  convert_element_type_default_55 = None
        mul_133: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_44, relu_8)
        mul_134: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_133, 1.1111111111111112);  mul_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        convert_element_type_346: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_93, torch.bfloat16);  primals_93 = None
        convert_element_type_default_24: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_134, torch.bfloat16);  mul_134 = None
        permute_127: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_346, [1, 0]);  convert_element_type_346 = None
        view_288: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_default_24, [8192, 2048]);  convert_element_type_default_24 = None
        mm_65: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_288, permute_127)
        view_289: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_65, [8, 1024, 512]);  mm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_44: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 44)
        inductor_random_default_19: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_44, 'rand');  inductor_lookup_seed_default_44 = None
        convert_element_type_default_54: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_19, torch.bfloat16);  inductor_random_default_19 = None
        gt_45: "b8[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_54, 0.1);  convert_element_type_default_54 = None
        mul_135: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_45, view_289);  view_289 = None
        mul_136: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_135, 1.1111111111111112);  mul_135 = None
        add_70: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_68, mul_136);  mul_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_23: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_70, 2)
        mean_22: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_23, [-1], True);  pow_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_71: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_22, 1e-06);  mean_22 = None
        rsqrt_22: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        mul_137: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_70, rsqrt_22)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_138: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_94, mul_137);  mul_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_350: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_95, torch.bfloat16);  primals_95 = None
        convert_element_type_351: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_138, torch.bfloat16);  mul_138 = None
        permute_128: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_350, [1, 0]);  convert_element_type_350 = None
        view_290: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_351, [8192, 512]);  convert_element_type_351 = None
        mm_66: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_290, permute_128)
        view_291: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_66, [8, 1024, 512]);  mm_66 = None
        view_292: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_291, [8, 1024, -1, 64]);  view_291 = None
        permute_129: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_354: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_96, torch.bfloat16);  primals_96 = None
        permute_130: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_354, [1, 0]);  convert_element_type_354 = None
        mm_67: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_290, permute_130)
        view_294: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_67, [8, 1024, 512]);  mm_67 = None
        view_295: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_294, [8, 1024, -1, 64]);  view_294 = None
        permute_131: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_295, [0, 2, 1, 3]);  view_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_358: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_97, torch.bfloat16);  primals_97 = None
        permute_132: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_358, [1, 0]);  convert_element_type_358 = None
        mm_68: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_290, permute_132)
        view_297: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_68, [8, 1024, 512]);  mm_68 = None
        view_298: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_297, [8, 1024, -1, 64]);  view_297 = None
        permute_133: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_298, [0, 2, 1, 3]);  view_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_134: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_131, [0, 1, 3, 2]);  permute_131 = None
        expand_51: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_129, [8, 8, 1024, 64]);  permute_129 = None
        clone_48: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        view_299: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_48, [64, 1024, 64]);  clone_48 = None
        expand_52: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_134, [8, 8, 64, 1024]);  permute_134 = None
        clone_49: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_300: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_49, [64, 64, 1024]);  clone_49 = None
        bmm_24: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_299, view_300)
        view_301: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_24, [8, 8, 1024, 1024]);  bmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_72: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_301, add_46);  view_301 = None
        convert_element_type_364: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_72, torch.bfloat16);  add_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_365: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_364, torch.float32)
        amax_12: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_365, [-1], True)
        sub_14: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_365, amax_12);  convert_element_type_365 = None
        exp_12: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
        sum_13: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_16: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = None
        convert_element_type_366: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_16, torch.bfloat16);  div_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_45: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 45)
        inductor_random_default_18: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_45, 'rand');  inductor_lookup_seed_default_45 = None
        convert_element_type_default_53: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_18, torch.bfloat16);  inductor_random_default_18 = None
        gt_46: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_53, 0.1);  convert_element_type_default_53 = None
        mul_139: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_46, convert_element_type_366);  convert_element_type_366 = None
        mul_140: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_139, 1.1111111111111112);  mul_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_53: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_140, [8, 8, 1024, 1024]);  mul_140 = None
        view_305: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_53, [64, 1024, 1024]);  expand_53 = None
        expand_54: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_133, [8, 8, 1024, 64]);  permute_133 = None
        clone_50: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_54, memory_format = torch.contiguous_format);  expand_54 = None
        view_306: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_50, [64, 1024, 64]);  clone_50 = None
        bmm_25: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_305, view_306)
        view_307: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_25, [8, 8, 1024, 64]);  bmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_135: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_307, [0, 2, 1, 3]);  view_307 = None
        clone_51: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_308: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_51, [8, 1024, -1]);  clone_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_369: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_98, torch.bfloat16);  primals_98 = None
        permute_136: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_369, [1, 0]);  convert_element_type_369 = None
        view_309: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_308, [8192, 512]);  view_308 = None
        mm_69: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_309, permute_136)
        view_310: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_69, [8, 1024, 512]);  mm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_46: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 46)
        inductor_random_default_17: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_46, 'rand');  inductor_lookup_seed_default_46 = None
        convert_element_type_default_52: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_17, torch.bfloat16);  inductor_random_default_17 = None
        gt_47: "b8[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_52, 0.1);  convert_element_type_default_52 = None
        mul_141: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_47, view_310);  view_310 = None
        mul_142: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_141, 1.1111111111111112);  mul_141 = None
        add_73: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_70, mul_142);  mul_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_24: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_73, 2)
        mean_23: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_24, [-1], True);  pow_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_74: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_23, 1e-06);  mean_23 = None
        rsqrt_23: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        mul_143: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_73, rsqrt_23)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_144: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_99, mul_143);  mul_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_372: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_100, torch.bfloat16);  primals_100 = None
        convert_element_type_373: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_144, torch.bfloat16);  mul_144 = None
        permute_137: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_372, [1, 0]);  convert_element_type_372 = None
        view_311: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_373, [8192, 512]);  convert_element_type_373 = None
        mm_70: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_311, permute_137)
        view_312: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_70, [8, 1024, 512]);  mm_70 = None
        view_313: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_312, [8, 1024, -1, 64]);  view_312 = None
        permute_138: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_313, [0, 2, 1, 3]);  view_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_376: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_101, torch.bfloat16);  primals_101 = None
        permute_139: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_376, [1, 0]);  convert_element_type_376 = None
        mm_71: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_176, permute_139)
        view_315: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_71, [8, 1024, 512]);  mm_71 = None
        view_316: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_315, [8, 1024, -1, 64]);  view_315 = None
        permute_140: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_380: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_102, torch.bfloat16);  primals_102 = None
        permute_141: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_380, [1, 0]);  convert_element_type_380 = None
        mm_72: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_176, permute_141)
        view_318: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_72, [8, 1024, 512]);  mm_72 = None
        view_319: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_318, [8, 1024, -1, 64]);  view_318 = None
        permute_142: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_319, [0, 2, 1, 3]);  view_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_143: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_140, [0, 1, 3, 2]);  permute_140 = None
        expand_55: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_138, [8, 8, 1024, 64]);  permute_138 = None
        clone_52: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
        view_320: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_52, [64, 1024, 64]);  clone_52 = None
        expand_56: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_143, [8, 8, 64, 1024]);  permute_143 = None
        clone_53: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_321: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_53, [64, 64, 1024]);  clone_53 = None
        bmm_26: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_320, view_321)
        view_322: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_26, [8, 8, 1024, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        convert_element_type_default_23: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_322, torch.bfloat16);  view_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_387: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_default_23, torch.float32);  convert_element_type_default_23 = None
        amax_13: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_387, [-1], True)
        sub_15: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_387, amax_13);  convert_element_type_387 = None
        exp_13: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
        sum_14: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_17: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = None
        convert_element_type_388: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_17, torch.bfloat16);  div_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_47: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 47)
        inductor_random_default_16: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_47, 'rand');  inductor_lookup_seed_default_47 = None
        convert_element_type_default_51: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_16, torch.bfloat16);  inductor_random_default_16 = None
        gt_48: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_51, 0.1);  convert_element_type_default_51 = None
        mul_145: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_48, convert_element_type_388);  convert_element_type_388 = None
        mul_146: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_145, 1.1111111111111112);  mul_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_57: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_146, [8, 8, 1024, 1024]);  mul_146 = None
        view_326: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_57, [64, 1024, 1024]);  expand_57 = None
        expand_58: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_142, [8, 8, 1024, 64]);  permute_142 = None
        clone_54: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
        view_327: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_54, [64, 1024, 64]);  clone_54 = None
        bmm_27: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_326, view_327)
        view_328: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_27, [8, 8, 1024, 64]);  bmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_144: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_328, [0, 2, 1, 3]);  view_328 = None
        clone_55: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_329: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_55, [8, 1024, -1]);  clone_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_391: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_103, torch.bfloat16);  primals_103 = None
        permute_145: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_391, [1, 0]);  convert_element_type_391 = None
        view_330: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_329, [8192, 512]);  view_329 = None
        mm_73: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_330, permute_145)
        view_331: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_73, [8, 1024, 512]);  mm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_48: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 48)
        inductor_random_default_15: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_48, 'rand');  inductor_lookup_seed_default_48 = None
        convert_element_type_default_50: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_15, torch.bfloat16);  inductor_random_default_15 = None
        gt_49: "b8[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_50, 0.1);  convert_element_type_default_50 = None
        mul_147: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_49, view_331);  view_331 = None
        mul_148: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_147, 1.1111111111111112);  mul_147 = None
        add_76: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_73, mul_148);  mul_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_25: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_76, 2)
        mean_24: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_77: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_24, 1e-06);  mean_24 = None
        rsqrt_24: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        mul_149: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_76, rsqrt_24)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_150: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_104, mul_149);  mul_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        convert_element_type_394: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_105, torch.bfloat16);  primals_105 = None
        convert_element_type_395: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_150, torch.bfloat16);  mul_150 = None
        permute_146: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_394, [1, 0]);  convert_element_type_394 = None
        view_332: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_395, [8192, 512]);  convert_element_type_395 = None
        mm_74: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_332, permute_146)
        view_333: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_74, [8, 1024, 2048]);  mm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_9: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.relu.default(view_333);  view_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_49: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 49)
        inductor_random_default_14: "f32[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 2048], inductor_lookup_seed_default_49, 'rand');  inductor_lookup_seed_default_49 = None
        convert_element_type_default_49: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_14, torch.bfloat16);  inductor_random_default_14 = None
        gt_50: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_49, 0.1);  convert_element_type_default_49 = None
        mul_151: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_50, relu_9)
        mul_152: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_151, 1.1111111111111112);  mul_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        convert_element_type_399: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_106, torch.bfloat16);  primals_106 = None
        convert_element_type_default_22: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_152, torch.bfloat16);  mul_152 = None
        permute_147: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_399, [1, 0]);  convert_element_type_399 = None
        view_334: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_default_22, [8192, 2048]);  convert_element_type_default_22 = None
        mm_75: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_334, permute_147)
        view_335: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_75, [8, 1024, 512]);  mm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_50: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 50)
        inductor_random_default_13: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_50, 'rand');  inductor_lookup_seed_default_50 = None
        convert_element_type_default_48: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_13, torch.bfloat16);  inductor_random_default_13 = None
        gt_51: "b8[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_48, 0.1);  convert_element_type_default_48 = None
        mul_153: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_51, view_335);  view_335 = None
        mul_154: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_153, 1.1111111111111112);  mul_153 = None
        add_78: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_76, mul_154);  mul_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_26: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_78, 2)
        mean_25: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_26, [-1], True);  pow_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_79: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_25, 1e-06);  mean_25 = None
        rsqrt_25: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
        mul_155: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_78, rsqrt_25)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_156: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_107, mul_155);  mul_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_403: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_108, torch.bfloat16);  primals_108 = None
        convert_element_type_404: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_156, torch.bfloat16);  mul_156 = None
        permute_148: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_403, [1, 0]);  convert_element_type_403 = None
        view_336: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_404, [8192, 512]);  convert_element_type_404 = None
        mm_76: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_336, permute_148)
        view_337: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_76, [8, 1024, 512]);  mm_76 = None
        view_338: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_337, [8, 1024, -1, 64]);  view_337 = None
        permute_149: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_407: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_109, torch.bfloat16);  primals_109 = None
        permute_150: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_407, [1, 0]);  convert_element_type_407 = None
        mm_77: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_336, permute_150)
        view_340: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_77, [8, 1024, 512]);  mm_77 = None
        view_341: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_340, [8, 1024, -1, 64]);  view_340 = None
        permute_151: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_341, [0, 2, 1, 3]);  view_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_411: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_110, torch.bfloat16);  primals_110 = None
        permute_152: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_411, [1, 0]);  convert_element_type_411 = None
        mm_78: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_336, permute_152)
        view_343: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_78, [8, 1024, 512]);  mm_78 = None
        view_344: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_343, [8, 1024, -1, 64]);  view_343 = None
        permute_153: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_154: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_151, [0, 1, 3, 2]);  permute_151 = None
        expand_59: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_149, [8, 8, 1024, 64]);  permute_149 = None
        clone_56: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
        view_345: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_56, [64, 1024, 64]);  clone_56 = None
        expand_60: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_154, [8, 8, 64, 1024]);  permute_154 = None
        clone_57: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_346: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_57, [64, 64, 1024]);  clone_57 = None
        bmm_28: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_345, view_346)
        view_347: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_28, [8, 8, 1024, 1024]);  bmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_80: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_347, add_46);  view_347 = None
        convert_element_type_417: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_80, torch.bfloat16);  add_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_418: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_417, torch.float32)
        amax_14: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_418, [-1], True)
        sub_16: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_418, amax_14);  convert_element_type_418 = None
        exp_14: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_15: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_18: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = None
        convert_element_type_419: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_18, torch.bfloat16);  div_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_51: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 51)
        inductor_random_default_12: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_51, 'rand');  inductor_lookup_seed_default_51 = None
        convert_element_type_default_47: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_12, torch.bfloat16);  inductor_random_default_12 = None
        gt_52: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_47, 0.1);  convert_element_type_default_47 = None
        mul_157: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_52, convert_element_type_419);  convert_element_type_419 = None
        mul_158: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_157, 1.1111111111111112);  mul_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_61: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_158, [8, 8, 1024, 1024]);  mul_158 = None
        view_351: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_61, [64, 1024, 1024]);  expand_61 = None
        expand_62: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_153, [8, 8, 1024, 64]);  permute_153 = None
        clone_58: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_62, memory_format = torch.contiguous_format);  expand_62 = None
        view_352: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_58, [64, 1024, 64]);  clone_58 = None
        bmm_29: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_351, view_352)
        view_353: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_29, [8, 8, 1024, 64]);  bmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_155: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_353, [0, 2, 1, 3]);  view_353 = None
        clone_59: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_354: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_59, [8, 1024, -1]);  clone_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_422: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_111, torch.bfloat16);  primals_111 = None
        permute_156: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_422, [1, 0]);  convert_element_type_422 = None
        view_355: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_354, [8192, 512]);  view_354 = None
        mm_79: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_355, permute_156)
        view_356: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_79, [8, 1024, 512]);  mm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_52: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 52)
        inductor_random_default_11: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_52, 'rand');  inductor_lookup_seed_default_52 = None
        convert_element_type_default_46: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_11, torch.bfloat16);  inductor_random_default_11 = None
        gt_53: "b8[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_46, 0.1);  convert_element_type_default_46 = None
        mul_159: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_53, view_356);  view_356 = None
        mul_160: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_159, 1.1111111111111112);  mul_159 = None
        add_81: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_78, mul_160);  mul_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_27: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_81, 2)
        mean_26: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_27, [-1], True);  pow_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_82: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_26, 1e-06);  mean_26 = None
        rsqrt_26: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        mul_161: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_81, rsqrt_26)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_162: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_112, mul_161);  mul_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_425: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_113, torch.bfloat16);  primals_113 = None
        convert_element_type_426: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_162, torch.bfloat16);  mul_162 = None
        permute_157: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_425, [1, 0]);  convert_element_type_425 = None
        view_357: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_426, [8192, 512]);  convert_element_type_426 = None
        mm_80: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_357, permute_157)
        view_358: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_80, [8, 1024, 512]);  mm_80 = None
        view_359: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_358, [8, 1024, -1, 64]);  view_358 = None
        permute_158: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_359, [0, 2, 1, 3]);  view_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_429: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_114, torch.bfloat16);  primals_114 = None
        permute_159: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_429, [1, 0]);  convert_element_type_429 = None
        mm_81: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_176, permute_159)
        view_361: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_81, [8, 1024, 512]);  mm_81 = None
        view_362: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_361, [8, 1024, -1, 64]);  view_361 = None
        permute_160: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_362, [0, 2, 1, 3]);  view_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_433: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_115, torch.bfloat16);  primals_115 = None
        permute_161: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_433, [1, 0]);  convert_element_type_433 = None
        mm_82: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_176, permute_161)
        view_364: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_82, [8, 1024, 512]);  mm_82 = None
        view_365: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_364, [8, 1024, -1, 64]);  view_364 = None
        permute_162: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_365, [0, 2, 1, 3]);  view_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_163: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_160, [0, 1, 3, 2]);  permute_160 = None
        expand_63: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_158, [8, 8, 1024, 64]);  permute_158 = None
        clone_60: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
        view_366: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_60, [64, 1024, 64]);  clone_60 = None
        expand_64: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_163, [8, 8, 64, 1024]);  permute_163 = None
        clone_61: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
        view_367: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_61, [64, 64, 1024]);  clone_61 = None
        bmm_30: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_366, view_367)
        view_368: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_30, [8, 8, 1024, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        convert_element_type_default_21: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_368, torch.bfloat16);  view_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_440: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_default_21, torch.float32);  convert_element_type_default_21 = None
        amax_15: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_440, [-1], True)
        sub_17: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_440, amax_15);  convert_element_type_440 = None
        exp_15: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
        sum_16: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_19: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = None
        convert_element_type_441: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16);  div_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_53: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 53)
        inductor_random_default_10: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_53, 'rand');  inductor_lookup_seed_default_53 = None
        convert_element_type_default_45: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_10, torch.bfloat16);  inductor_random_default_10 = None
        gt_54: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_45, 0.1);  convert_element_type_default_45 = None
        mul_163: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_54, convert_element_type_441);  convert_element_type_441 = None
        mul_164: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_163, 1.1111111111111112);  mul_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_65: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_164, [8, 8, 1024, 1024]);  mul_164 = None
        view_372: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_65, [64, 1024, 1024]);  expand_65 = None
        expand_66: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_162, [8, 8, 1024, 64]);  permute_162 = None
        clone_62: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
        view_373: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_62, [64, 1024, 64]);  clone_62 = None
        bmm_31: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_372, view_373)
        view_374: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_31, [8, 8, 1024, 64]);  bmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_164: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_374, [0, 2, 1, 3]);  view_374 = None
        clone_63: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_164, memory_format = torch.contiguous_format);  permute_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_375: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_63, [8, 1024, -1]);  clone_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_444: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_116, torch.bfloat16);  primals_116 = None
        permute_165: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_444, [1, 0]);  convert_element_type_444 = None
        view_376: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_375, [8192, 512]);  view_375 = None
        mm_83: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_376, permute_165)
        view_377: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_83, [8, 1024, 512]);  mm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_54: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 54)
        inductor_random_default_9: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_54, 'rand');  inductor_lookup_seed_default_54 = None
        convert_element_type_default_44: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_9, torch.bfloat16);  inductor_random_default_9 = None
        gt_55: "b8[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_44, 0.1);  convert_element_type_default_44 = None
        mul_165: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_55, view_377);  view_377 = None
        mul_166: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_165, 1.1111111111111112);  mul_165 = None
        add_84: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_81, mul_166);  mul_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_28: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_84, 2)
        mean_27: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_28, [-1], True);  pow_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_85: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_27, 1e-06);  mean_27 = None
        rsqrt_27: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        mul_167: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_84, rsqrt_27)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_168: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_117, mul_167);  mul_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        convert_element_type_447: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_118, torch.bfloat16);  primals_118 = None
        convert_element_type_448: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_168, torch.bfloat16);  mul_168 = None
        permute_166: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_447, [1, 0]);  convert_element_type_447 = None
        view_378: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_448, [8192, 512]);  convert_element_type_448 = None
        mm_84: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_378, permute_166)
        view_379: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_84, [8, 1024, 2048]);  mm_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_10: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.relu.default(view_379);  view_379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_55: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 55)
        inductor_random_default_8: "f32[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 2048], inductor_lookup_seed_default_55, 'rand');  inductor_lookup_seed_default_55 = None
        convert_element_type_default_43: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_8, torch.bfloat16);  inductor_random_default_8 = None
        gt_56: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_43, 0.1);  convert_element_type_default_43 = None
        mul_169: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_56, relu_10)
        mul_170: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_169, 1.1111111111111112);  mul_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        convert_element_type_452: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_119, torch.bfloat16);  primals_119 = None
        convert_element_type_default_20: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_170, torch.bfloat16);  mul_170 = None
        permute_167: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_452, [1, 0]);  convert_element_type_452 = None
        view_380: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_default_20, [8192, 2048]);  convert_element_type_default_20 = None
        mm_85: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_380, permute_167)
        view_381: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_85, [8, 1024, 512]);  mm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_56: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 56)
        inductor_random_default_7: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_56, 'rand');  inductor_lookup_seed_default_56 = None
        convert_element_type_default_42: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_7, torch.bfloat16);  inductor_random_default_7 = None
        gt_57: "b8[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_42, 0.1);  convert_element_type_default_42 = None
        mul_171: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_57, view_381);  view_381 = None
        mul_172: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_171, 1.1111111111111112);  mul_171 = None
        add_86: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_84, mul_172);  mul_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_29: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_86, 2)
        mean_28: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_29, [-1], True);  pow_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_87: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_28, 1e-06);  mean_28 = None
        rsqrt_28: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
        mul_173: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_86, rsqrt_28)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_174: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_120, mul_173);  mul_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_456: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_121, torch.bfloat16);  primals_121 = None
        convert_element_type_457: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_174, torch.bfloat16);  mul_174 = None
        permute_168: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_456, [1, 0]);  convert_element_type_456 = None
        view_382: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_457, [8192, 512]);  convert_element_type_457 = None
        mm_86: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_382, permute_168)
        view_383: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_86, [8, 1024, 512]);  mm_86 = None
        view_384: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_383, [8, 1024, -1, 64]);  view_383 = None
        permute_169: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_384, [0, 2, 1, 3]);  view_384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_460: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_122, torch.bfloat16);  primals_122 = None
        permute_170: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_460, [1, 0]);  convert_element_type_460 = None
        mm_87: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_382, permute_170)
        view_386: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_87, [8, 1024, 512]);  mm_87 = None
        view_387: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_386, [8, 1024, -1, 64]);  view_386 = None
        permute_171: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_387, [0, 2, 1, 3]);  view_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_464: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_123, torch.bfloat16);  primals_123 = None
        permute_172: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_464, [1, 0]);  convert_element_type_464 = None
        mm_88: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_382, permute_172)
        view_389: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_88, [8, 1024, 512]);  mm_88 = None
        view_390: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_389, [8, 1024, -1, 64]);  view_389 = None
        permute_173: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_390, [0, 2, 1, 3]);  view_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_174: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_171, [0, 1, 3, 2]);  permute_171 = None
        expand_67: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_169, [8, 8, 1024, 64]);  permute_169 = None
        clone_64: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
        view_391: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_64, [64, 1024, 64]);  clone_64 = None
        expand_68: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_174, [8, 8, 64, 1024]);  permute_174 = None
        clone_65: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
        view_392: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_65, [64, 64, 1024]);  clone_65 = None
        bmm_32: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_391, view_392)
        view_393: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_32, [8, 8, 1024, 1024]);  bmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_88: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_393, add_46);  view_393 = add_46 = None
        convert_element_type_470: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_88, torch.bfloat16);  add_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_471: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_470, torch.float32)
        amax_16: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_471, [-1], True)
        sub_18: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_471, amax_16);  convert_element_type_471 = None
        exp_16: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
        sum_17: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_20: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = None
        convert_element_type_472: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_20, torch.bfloat16);  div_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_57: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 57)
        inductor_random_default_6: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_57, 'rand');  inductor_lookup_seed_default_57 = None
        convert_element_type_default_41: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_6, torch.bfloat16);  inductor_random_default_6 = None
        gt_58: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_41, 0.1);  convert_element_type_default_41 = None
        mul_175: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_58, convert_element_type_472);  convert_element_type_472 = None
        mul_176: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_175, 1.1111111111111112);  mul_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_69: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_176, [8, 8, 1024, 1024]);  mul_176 = None
        view_397: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_69, [64, 1024, 1024]);  expand_69 = None
        expand_70: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_173, [8, 8, 1024, 64]);  permute_173 = None
        clone_66: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_70, memory_format = torch.contiguous_format);  expand_70 = None
        view_398: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_66, [64, 1024, 64]);  clone_66 = None
        bmm_33: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_397, view_398)
        view_399: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_33, [8, 8, 1024, 64]);  bmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_175: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_399, [0, 2, 1, 3]);  view_399 = None
        clone_67: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_175, memory_format = torch.contiguous_format);  permute_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_400: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_67, [8, 1024, -1]);  clone_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_475: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_124, torch.bfloat16);  primals_124 = None
        permute_176: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_475, [1, 0]);  convert_element_type_475 = None
        view_401: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_400, [8192, 512]);  view_400 = None
        mm_89: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_401, permute_176)
        view_402: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_89, [8, 1024, 512]);  mm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_58: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 58)
        inductor_random_default_5: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_58, 'rand');  inductor_lookup_seed_default_58 = None
        convert_element_type_default_40: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_5, torch.bfloat16);  inductor_random_default_5 = None
        gt_59: "b8[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_40, 0.1);  convert_element_type_default_40 = None
        mul_177: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_59, view_402);  view_402 = None
        mul_178: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_177, 1.1111111111111112);  mul_177 = None
        add_89: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_86, mul_178);  mul_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_30: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_89, 2)
        mean_29: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_30, [-1], True);  pow_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_90: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_29, 1e-06);  mean_29 = None
        rsqrt_29: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        mul_179: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_89, rsqrt_29)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_180: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_125, mul_179);  mul_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        convert_element_type_478: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_126, torch.bfloat16);  primals_126 = None
        convert_element_type_479: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_180, torch.bfloat16);  mul_180 = None
        permute_177: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_478, [1, 0]);  convert_element_type_478 = None
        view_403: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_479, [8192, 512]);  convert_element_type_479 = None
        mm_90: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_403, permute_177)
        view_404: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_90, [8, 1024, 512]);  mm_90 = None
        view_405: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_404, [8, 1024, -1, 64]);  view_404 = None
        permute_178: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_405, [0, 2, 1, 3]);  view_405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_482: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_127, torch.bfloat16);  primals_127 = None
        permute_179: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_482, [1, 0]);  convert_element_type_482 = None
        mm_91: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_176, permute_179)
        view_407: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_91, [8, 1024, 512]);  mm_91 = None
        view_408: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_407, [8, 1024, -1, 64]);  view_407 = None
        permute_180: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_408, [0, 2, 1, 3]);  view_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        convert_element_type_486: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_128, torch.bfloat16);  primals_128 = None
        permute_181: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_486, [1, 0]);  convert_element_type_486 = None
        mm_92: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_176, permute_181)
        view_410: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_92, [8, 1024, 512]);  mm_92 = None
        view_411: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_410, [8, 1024, -1, 64]);  view_410 = None
        permute_182: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_411, [0, 2, 1, 3]);  view_411 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_183: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_180, [0, 1, 3, 2]);  permute_180 = None
        expand_71: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_178, [8, 8, 1024, 64]);  permute_178 = None
        clone_68: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
        view_412: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_68, [64, 1024, 64]);  clone_68 = None
        expand_72: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_183, [8, 8, 64, 1024]);  permute_183 = None
        clone_69: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
        view_413: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_69, [64, 64, 1024]);  clone_69 = None
        bmm_34: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_412, view_413)
        view_414: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_34, [8, 8, 1024, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        convert_element_type_default_19: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_414, torch.bfloat16);  view_414 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_493: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_default_19, torch.float32);  convert_element_type_default_19 = None
        amax_17: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_493, [-1], True)
        sub_19: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_493, amax_17);  convert_element_type_493 = None
        exp_17: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_18: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_21: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = None
        convert_element_type_494: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16);  div_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_59: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 59)
        inductor_random_default_4: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 8, 1024, 1024], inductor_lookup_seed_default_59, 'rand');  inductor_lookup_seed_default_59 = None
        convert_element_type_default_39: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_4, torch.bfloat16);  inductor_random_default_4 = None
        gt_60: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_39, 0.1);  convert_element_type_default_39 = None
        mul_181: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_60, convert_element_type_494);  convert_element_type_494 = None
        mul_182: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_181, 1.1111111111111112);  mul_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_73: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(mul_182, [8, 8, 1024, 1024]);  mul_182 = None
        view_418: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_73, [64, 1024, 1024]);  expand_73 = None
        expand_74: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_182, [8, 8, 1024, 64]);  permute_182 = None
        clone_70: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
        view_419: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_70, [64, 1024, 64]);  clone_70 = None
        bmm_35: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_418, view_419)
        view_420: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_35, [8, 8, 1024, 64]);  bmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_184: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_420, [0, 2, 1, 3]);  view_420 = None
        clone_71: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_184, memory_format = torch.contiguous_format);  permute_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_421: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_71, [8, 1024, -1]);  clone_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        convert_element_type_497: "bf16[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_129, torch.bfloat16);  primals_129 = None
        permute_185: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_497, [1, 0]);  convert_element_type_497 = None
        view_422: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_421, [8192, 512]);  view_421 = None
        mm_93: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_422, permute_185)
        view_423: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_93, [8, 1024, 512]);  mm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default_60: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 60)
        inductor_random_default_3: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_60, 'rand');  inductor_lookup_seed_default_60 = None
        convert_element_type_default_38: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_3, torch.bfloat16);  inductor_random_default_3 = None
        gt_61: "b8[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_38, 0.1);  convert_element_type_default_38 = None
        mul_183: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_61, view_423);  view_423 = None
        mul_184: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_183, 1.1111111111111112);  mul_183 = None
        add_92: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_89, mul_184);  mul_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_31: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_92, 2)
        mean_30: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_31, [-1], True);  pow_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_93: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_30, 1e-06);  mean_30 = None
        rsqrt_30: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        mul_185: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_92, rsqrt_30)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_186: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_130, mul_185);  mul_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        convert_element_type_500: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_131, torch.bfloat16);  primals_131 = None
        convert_element_type_501: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_186, torch.bfloat16);  mul_186 = None
        permute_186: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_500, [1, 0]);  convert_element_type_500 = None
        view_424: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_501, [8192, 512]);  convert_element_type_501 = None
        mm_94: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_424, permute_186)
        view_425: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_94, [8, 1024, 2048]);  mm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_11: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.relu.default(view_425);  view_425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_61: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 61)
        inductor_random_default_2: "f32[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 2048], inductor_lookup_seed_default_61, 'rand');  inductor_lookup_seed_default_61 = None
        convert_element_type_default_37: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_2, torch.bfloat16);  inductor_random_default_2 = None
        gt_62: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_37, 0.1);  convert_element_type_default_37 = None
        mul_187: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_62, relu_11)
        mul_188: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_187, 1.1111111111111112);  mul_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        convert_element_type_505: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_132, torch.bfloat16);  primals_132 = None
        convert_element_type_default_18: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_188, torch.bfloat16);  mul_188 = None
        permute_187: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_505, [1, 0]);  convert_element_type_505 = None
        view_426: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_default_18, [8192, 2048]);  convert_element_type_default_18 = None
        mm_95: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_426, permute_187)
        view_427: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_95, [8, 1024, 512]);  mm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default_62: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 62)
        inductor_random_default_1: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_62, 'rand');  inductor_lookup_seed_default_62 = None
        convert_element_type_default_36: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_1, torch.bfloat16);  inductor_random_default_1 = None
        gt_63: "b8[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_36, 0.1);  convert_element_type_default_36 = None
        mul_189: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_63, view_427);  view_427 = None
        mul_190: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_189, 1.1111111111111112);  mul_189 = None
        add_94: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_92, mul_190);  mul_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_32: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_94, 2)
        mean_31: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_32, [-1], True);  pow_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_95: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_31, 1e-06);  mean_31 = None
        rsqrt_31: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
        mul_191: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_94, rsqrt_31)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_192: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_133, mul_191);  mul_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:755 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_63: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 63);  inductor_seeds_default = None
        inductor_random_default: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_63, 'rand');  inductor_lookup_seed_default_63 = None
        gt_64: "b8[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_193: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_64, mul_192);  mul_192 = None
        mul_194: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_193, 1.1111111111111112);  mul_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1095 in forward, code: sequence_output = sequence_output * (self.model_dim**-0.5)
        mul_195: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_194, 0.04419417382415922);  mul_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1097 in forward, code: lm_logits = self.lm_head(sequence_output)
        convert_element_type_509: "bf16[32128, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16);  primals_2 = None
        convert_element_type_510: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_195, torch.bfloat16);  mul_195 = None
        permute_188: "bf16[512, 32128][1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_509, [1, 0]);  convert_element_type_509 = None
        view_428: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_510, [8192, 512]);  convert_element_type_510 = None
        mm_96: "bf16[8192, 32128][32128, 1]cuda:0" = torch.ops.aten.mm.default(view_428, permute_188)
        view_429: "bf16[8, 1024, 32128][32899072, 32128, 1]cuda:0" = torch.ops.aten.reshape.default(mm_96, [8, 1024, 32128]);  mm_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1104 in forward, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        view_430: "bf16[8192, 32128][32128, 1]cuda:0" = torch.ops.aten.reshape.default(view_429, [-1, 32128])
        view_431: "i64[8192][1]cuda:0" = torch.ops.aten.reshape.default(primals_53, [-1])
        convert_element_type_513: "f32[8192, 32128][32128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_430, torch.float32);  view_430 = None
        amax_18: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_513, [1], True)
        sub_20: "f32[8192, 32128][32128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_513, amax_18);  convert_element_type_513 = None
        exp_18: "f32[8192, 32128][32128, 1]cuda:0" = torch.ops.aten.exp.default(sub_20)
        sum_19: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_18, [1], True);  exp_18 = None
        log_2: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.log.default(sum_19);  sum_19 = None
        sub_21: "f32[8192, 32128][32128, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_20, log_2);  sub_20 = None
        convert_element_type_514: "bf16[8192, 32128][32128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_21, torch.bfloat16);  sub_21 = None
        convert_element_type_515: "f32[8192, 32128][32128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_514, torch.float32);  convert_element_type_514 = None
        ne: "b8[8192][1]cuda:0" = torch.ops.aten.ne.Scalar(view_431, -100)
        full_default_4: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5: "i64[8192][1]cuda:0" = torch.ops.aten.where.self(ne, view_431, full_default_4);  view_431 = full_default_4 = None
        unsqueeze_18: "i64[8192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(where_5, 1);  where_5 = None
        gather: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.gather.default(convert_element_type_515, 1, unsqueeze_18);  convert_element_type_515 = unsqueeze_18 = None
        squeeze: "f32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg_1: "f32[8192][1]cuda:0" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        where_6: "f32[8192][1]cuda:0" = torch.ops.aten.where.self(ne, neg_1, full_default_1);  neg_1 = full_default_1 = None
        sum_20: "i64[][]cuda:0" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type_516: "f32[][]cuda:0" = torch.ops.prims.convert_element_type.default(sum_20, torch.float32);  sum_20 = None
        sum_21: "f32[][]cuda:0" = torch.ops.aten.sum.default(where_6);  where_6 = None
        div_22: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(sum_21, convert_element_type_516);  sum_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1097 in forward, code: lm_logits = self.lm_head(sequence_output)
        permute_191: "bf16[32128, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_188, [1, 0]);  permute_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_195: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_1: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_199: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_203: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_206: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_418, [0, 2, 1]);  view_418 = None
        permute_207: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_419, [0, 2, 1]);  view_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_208: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_412, [0, 2, 1]);  view_412 = None
        permute_209: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_413, [0, 2, 1]);  view_413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_214: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_219: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_179, [1, 0]);  permute_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_224: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_228: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_231: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_397, [0, 2, 1]);  view_397 = None
        permute_232: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_398, [0, 2, 1]);  view_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_233: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_391, [0, 2, 1]);  view_391 = None
        permute_234: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_392, [0, 2, 1]);  view_392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_239: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_244: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_249: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_168, [1, 0]);  permute_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_253: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_2: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_257: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_261: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_264: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_372, [0, 2, 1]);  view_372 = None
        permute_265: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_373, [0, 2, 1]);  view_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_266: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_366, [0, 2, 1]);  view_366 = None
        permute_267: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_367, [0, 2, 1]);  view_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_272: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_277: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_159, [1, 0]);  permute_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_282: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_157, [1, 0]);  permute_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_286: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_289: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_351, [0, 2, 1]);  view_351 = None
        permute_290: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_352, [0, 2, 1]);  view_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_291: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_345, [0, 2, 1]);  view_345 = None
        permute_292: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_346, [0, 2, 1]);  view_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_297: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_302: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_150, [1, 0]);  permute_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_307: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_311: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_147, [1, 0]);  permute_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_3: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_315: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_146, [1, 0]);  permute_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_319: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_322: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_326, [0, 2, 1]);  view_326 = None
        permute_323: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_327, [0, 2, 1]);  view_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_324: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_320, [0, 2, 1]);  view_320 = None
        permute_325: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_321, [0, 2, 1]);  view_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_330: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_335: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_139, [1, 0]);  permute_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_340: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_137, [1, 0]);  permute_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_344: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_347: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_305, [0, 2, 1]);  view_305 = None
        permute_348: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_306, [0, 2, 1]);  view_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_349: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_299, [0, 2, 1]);  view_299 = None
        permute_350: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_300, [0, 2, 1]);  view_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_355: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_360: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_365: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_369: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_127, [1, 0]);  permute_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_4: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_373: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_126, [1, 0]);  permute_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_377: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_380: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_280, [0, 2, 1]);  view_280 = None
        permute_381: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_281, [0, 2, 1]);  view_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_382: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_274, [0, 2, 1]);  view_274 = None
        permute_383: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_275, [0, 2, 1]);  view_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_388: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_393: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_398: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_117, [1, 0]);  permute_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_402: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_405: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_259, [0, 2, 1]);  view_259 = None
        permute_406: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_260, [0, 2, 1]);  view_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_407: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_253, [0, 2, 1]);  view_253 = None
        permute_408: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_413: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_418: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_423: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_427: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_5: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_431: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_435: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_105, [1, 0]);  permute_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_438: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_234, [0, 2, 1]);  view_234 = None
        permute_439: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_235, [0, 2, 1]);  view_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_440: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_228, [0, 2, 1]);  view_228 = None
        permute_441: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_229, [0, 2, 1]);  view_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_446: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_451: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_456: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_460: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_463: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_213, [0, 2, 1]);  view_213 = None
        permute_464: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_214, [0, 2, 1]);  view_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_465: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_207, [0, 2, 1]);  view_207 = None
        permute_466: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_208, [0, 2, 1]);  view_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_471: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_476: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_481: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_485: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_6: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_489: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_493: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_496: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
        permute_497: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_498: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_182, [0, 2, 1]);  view_182 = None
        permute_499: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_183, [0, 2, 1]);  view_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_504: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_509: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_514: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_518: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_521: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None
        permute_522: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_168, [0, 2, 1]);  view_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_524: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_161, [0, 2, 1]);  view_161 = None
        permute_525: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_162, [0, 2, 1]);  view_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_530: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_535: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_540: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_544: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_7: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_548: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_552: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_555: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_141, [0, 2, 1]);  view_141 = None
        permute_556: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_142, [0, 2, 1]);  view_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_557: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_135, [0, 2, 1]);  view_135 = None
        permute_558: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_136, [0, 2, 1]);  view_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_563: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_568: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_573: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_577: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_8: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_581: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_585: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_588: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_116, [0, 2, 1]);  view_116 = None
        permute_589: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_117, [0, 2, 1]);  view_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_590: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_110, [0, 2, 1]);  view_110 = None
        permute_591: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_111, [0, 2, 1]);  view_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_596: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_601: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_606: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_610: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_9: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_614: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_618: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_621: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_91, [0, 2, 1]);  view_91 = None
        permute_622: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_92, [0, 2, 1]);  view_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_623: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_85, [0, 2, 1]);  view_85 = None
        permute_624: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_86, [0, 2, 1]);  view_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_629: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_634: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_639: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_643: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_10: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_647: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_651: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_654: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_66, [0, 2, 1]);  view_66 = None
        permute_655: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_67, [0, 2, 1]);  view_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_656: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_60, [0, 2, 1]);  view_60 = None
        permute_657: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_61, [0, 2, 1]);  view_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_662: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_667: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_672: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_676: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_11: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_680: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_684: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_687: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_41, [0, 2, 1]);  view_41 = None
        permute_688: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_42, [0, 2, 1]);  view_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_689: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_35, [0, 2, 1]);  view_35 = None
        permute_690: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_36, [0, 2, 1]);  view_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_695: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_700: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_705: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_709: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_12: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.le.Scalar(relu, 0);  relu = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_713: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        permute_717: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_720: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_16, [0, 2, 1]);  view_16 = None
        permute_721: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_17, [0, 2, 1]);  view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_723: "bf16[64, 64, 1024][65536, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
        permute_724: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_729: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_734: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_739: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (div_22, view_429, mul_79, primals_1, primals_3, primals_9, primals_12, primals_17, primals_20, primals_25, primals_28, primals_33, primals_36, primals_41, primals_44, primals_49, primals_52, primals_53, primals_54, primals_60, primals_65, primals_68, primals_73, primals_78, primals_81, primals_86, primals_91, primals_94, primals_99, primals_104, primals_107, primals_112, primals_117, primals_120, primals_125, primals_130, primals_133, embedding, gt, rsqrt, view_1, bmm, add_6, embedding_1, amax, sum_1, gt_2, view_20, gt_3, add_9, rsqrt_1, view_22, gt_4, view_24, gt_5, add_11, rsqrt_2, view_26, convert_element_type_48, amax_1, sum_2, gt_6, view_45, gt_7, add_14, rsqrt_3, view_47, gt_8, view_49, gt_9, add_16, rsqrt_4, view_51, convert_element_type_79, amax_2, sum_3, gt_10, view_70, gt_11, add_19, rsqrt_5, view_72, gt_12, view_74, gt_13, add_21, rsqrt_6, view_76, convert_element_type_110, amax_3, sum_4, gt_14, view_95, gt_15, add_24, rsqrt_7, view_97, gt_16, view_99, gt_17, add_26, rsqrt_8, view_101, convert_element_type_141, amax_4, sum_5, gt_18, view_120, gt_19, add_29, rsqrt_9, view_122, gt_20, view_124, gt_21, add_31, rsqrt_10, view_126, convert_element_type_172, amax_5, sum_6, gt_22, view_145, gt_23, add_34, rsqrt_11, view_147, gt_24, view_149, gt_25, add_36, rsqrt_12, gt_26, unsqueeze_7, gt_27, rsqrt_13, view_152, bmm_12, add_45, embedding_3, amax_6, sum_7, gt_28, view_171, gt_29, add_48, rsqrt_14, view_173, view_176, bmm_14, amax_7, sum_8, gt_30, view_192, gt_31, add_52, rsqrt_15, view_194, gt_32, view_196, gt_33, add_54, rsqrt_16, view_198, convert_element_type_258, amax_8, sum_9, gt_34, view_217, gt_35, add_57, rsqrt_17, view_219, bmm_18, amax_9, sum_10, gt_36, view_238, gt_37, add_60, rsqrt_18, view_240, gt_38, view_242, gt_39, add_62, rsqrt_19, view_244, convert_element_type_311, amax_10, sum_11, gt_40, view_263, gt_41, add_65, rsqrt_20, view_265, bmm_22, amax_11, sum_12, gt_42, view_284, gt_43, add_68, rsqrt_21, view_286, gt_44, view_288, gt_45, add_70, rsqrt_22, view_290, convert_element_type_364, amax_12, sum_13, gt_46, view_309, gt_47, add_73, rsqrt_23, view_311, bmm_26, amax_13, sum_14, gt_48, view_330, gt_49, add_76, rsqrt_24, view_332, gt_50, view_334, gt_51, add_78, rsqrt_25, view_336, convert_element_type_417, amax_14, sum_15, gt_52, view_355, gt_53, add_81, rsqrt_26, view_357, bmm_30, amax_15, sum_16, gt_54, view_376, gt_55, add_84, rsqrt_27, view_378, gt_56, view_380, gt_57, add_86, rsqrt_28, view_382, convert_element_type_470, amax_16, sum_17, gt_58, view_401, gt_59, add_89, rsqrt_29, view_403, bmm_34, amax_17, sum_18, gt_60, view_422, gt_61, add_92, rsqrt_30, view_424, gt_62, view_426, gt_63, add_94, rsqrt_31, gt_64, view_428, view_429, amax_18, log_2, convert_element_type_516, permute_191, permute_195, le_1, permute_199, permute_203, permute_206, permute_207, permute_208, permute_209, permute_214, permute_219, permute_224, permute_228, permute_231, permute_232, permute_233, permute_234, permute_239, permute_244, permute_249, permute_253, le_2, permute_257, permute_261, permute_264, permute_265, permute_266, permute_267, permute_272, permute_277, permute_282, permute_286, permute_289, permute_290, permute_291, permute_292, permute_297, permute_302, permute_307, permute_311, le_3, permute_315, permute_319, permute_322, permute_323, permute_324, permute_325, permute_330, permute_335, permute_340, permute_344, permute_347, permute_348, permute_349, permute_350, permute_355, permute_360, permute_365, permute_369, le_4, permute_373, permute_377, permute_380, permute_381, permute_382, permute_383, permute_388, permute_393, permute_398, permute_402, permute_405, permute_406, permute_407, permute_408, permute_413, permute_418, permute_423, permute_427, le_5, permute_431, permute_435, permute_438, permute_439, permute_440, permute_441, permute_446, permute_451, permute_456, permute_460, permute_463, permute_464, permute_465, permute_466, permute_471, permute_476, permute_481, permute_485, le_6, permute_489, permute_493, permute_496, permute_497, permute_498, permute_499, permute_504, permute_509, permute_514, permute_518, permute_521, permute_522, permute_524, permute_525, permute_530, permute_535, permute_540, permute_544, le_7, permute_548, permute_552, permute_555, permute_556, permute_557, permute_558, permute_563, permute_568, permute_573, permute_577, le_8, permute_581, permute_585, permute_588, permute_589, permute_590, permute_591, permute_596, permute_601, permute_606, permute_610, le_9, permute_614, permute_618, permute_621, permute_622, permute_623, permute_624, permute_629, permute_634, permute_639, permute_643, le_10, permute_647, permute_651, permute_654, permute_655, permute_656, permute_657, permute_662, permute_667, permute_672, permute_676, le_11, permute_680, permute_684, permute_687, permute_688, permute_689, permute_690, permute_695, permute_700, permute_705, permute_709, le_12, permute_713, permute_717, permute_720, permute_721, permute_723, permute_724, permute_729, permute_734, permute_739)
