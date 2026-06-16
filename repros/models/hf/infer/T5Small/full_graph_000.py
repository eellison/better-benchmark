class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[8, 1024][1024, 1]cuda:0", arg1_1: "bf16[32128, 512][512, 1]cuda:0", arg2_1: "bf16[512][1]cuda:0", arg3_1: "bf16[512, 512][512, 1]cuda:0", arg4_1: "bf16[512, 512][512, 1]cuda:0", arg5_1: "bf16[512, 512][512, 1]cuda:0", arg6_1: "bf16[32, 8][8, 1]cuda:0", arg7_1: "bf16[512, 512][512, 1]cuda:0", arg8_1: "bf16[512][1]cuda:0", arg9_1: "bf16[2048, 512][512, 1]cuda:0", arg10_1: "bf16[512, 2048][2048, 1]cuda:0", arg11_1: "bf16[512][1]cuda:0", arg12_1: "bf16[512, 512][512, 1]cuda:0", arg13_1: "bf16[512, 512][512, 1]cuda:0", arg14_1: "bf16[512, 512][512, 1]cuda:0", arg15_1: "bf16[512, 512][512, 1]cuda:0", arg16_1: "bf16[512][1]cuda:0", arg17_1: "bf16[2048, 512][512, 1]cuda:0", arg18_1: "bf16[512, 2048][2048, 1]cuda:0", arg19_1: "bf16[512][1]cuda:0", arg20_1: "bf16[512, 512][512, 1]cuda:0", arg21_1: "bf16[512, 512][512, 1]cuda:0", arg22_1: "bf16[512, 512][512, 1]cuda:0", arg23_1: "bf16[512, 512][512, 1]cuda:0", arg24_1: "bf16[512][1]cuda:0", arg25_1: "bf16[2048, 512][512, 1]cuda:0", arg26_1: "bf16[512, 2048][2048, 1]cuda:0", arg27_1: "bf16[512][1]cuda:0", arg28_1: "bf16[512, 512][512, 1]cuda:0", arg29_1: "bf16[512, 512][512, 1]cuda:0", arg30_1: "bf16[512, 512][512, 1]cuda:0", arg31_1: "bf16[512, 512][512, 1]cuda:0", arg32_1: "bf16[512][1]cuda:0", arg33_1: "bf16[2048, 512][512, 1]cuda:0", arg34_1: "bf16[512, 2048][2048, 1]cuda:0", arg35_1: "bf16[512][1]cuda:0", arg36_1: "bf16[512, 512][512, 1]cuda:0", arg37_1: "bf16[512, 512][512, 1]cuda:0", arg38_1: "bf16[512, 512][512, 1]cuda:0", arg39_1: "bf16[512, 512][512, 1]cuda:0", arg40_1: "bf16[512][1]cuda:0", arg41_1: "bf16[2048, 512][512, 1]cuda:0", arg42_1: "bf16[512, 2048][2048, 1]cuda:0", arg43_1: "bf16[512][1]cuda:0", arg44_1: "bf16[512, 512][512, 1]cuda:0", arg45_1: "bf16[512, 512][512, 1]cuda:0", arg46_1: "bf16[512, 512][512, 1]cuda:0", arg47_1: "bf16[512, 512][512, 1]cuda:0", arg48_1: "bf16[512][1]cuda:0", arg49_1: "bf16[2048, 512][512, 1]cuda:0", arg50_1: "bf16[512, 2048][2048, 1]cuda:0", arg51_1: "bf16[512][1]cuda:0", arg52_1: "i64[8, 1024][1024, 1]cuda:0", arg53_1: "bf16[512][1]cuda:0", arg54_1: "bf16[512, 512][512, 1]cuda:0", arg55_1: "bf16[512, 512][512, 1]cuda:0", arg56_1: "bf16[512, 512][512, 1]cuda:0", arg57_1: "bf16[32, 8][8, 1]cuda:0", arg58_1: "bf16[512, 512][512, 1]cuda:0", arg59_1: "bf16[512][1]cuda:0", arg60_1: "bf16[512, 512][512, 1]cuda:0", arg61_1: "bf16[512, 512][512, 1]cuda:0", arg62_1: "bf16[512, 512][512, 1]cuda:0", arg63_1: "bf16[512, 512][512, 1]cuda:0", arg64_1: "bf16[512][1]cuda:0", arg65_1: "bf16[2048, 512][512, 1]cuda:0", arg66_1: "bf16[512, 2048][2048, 1]cuda:0", arg67_1: "bf16[512][1]cuda:0", arg68_1: "bf16[512, 512][512, 1]cuda:0", arg69_1: "bf16[512, 512][512, 1]cuda:0", arg70_1: "bf16[512, 512][512, 1]cuda:0", arg71_1: "bf16[512, 512][512, 1]cuda:0", arg72_1: "bf16[512][1]cuda:0", arg73_1: "bf16[512, 512][512, 1]cuda:0", arg74_1: "bf16[512, 512][512, 1]cuda:0", arg75_1: "bf16[512, 512][512, 1]cuda:0", arg76_1: "bf16[512, 512][512, 1]cuda:0", arg77_1: "bf16[512][1]cuda:0", arg78_1: "bf16[2048, 512][512, 1]cuda:0", arg79_1: "bf16[512, 2048][2048, 1]cuda:0", arg80_1: "bf16[512][1]cuda:0", arg81_1: "bf16[512, 512][512, 1]cuda:0", arg82_1: "bf16[512, 512][512, 1]cuda:0", arg83_1: "bf16[512, 512][512, 1]cuda:0", arg84_1: "bf16[512, 512][512, 1]cuda:0", arg85_1: "bf16[512][1]cuda:0", arg86_1: "bf16[512, 512][512, 1]cuda:0", arg87_1: "bf16[512, 512][512, 1]cuda:0", arg88_1: "bf16[512, 512][512, 1]cuda:0", arg89_1: "bf16[512, 512][512, 1]cuda:0", arg90_1: "bf16[512][1]cuda:0", arg91_1: "bf16[2048, 512][512, 1]cuda:0", arg92_1: "bf16[512, 2048][2048, 1]cuda:0", arg93_1: "bf16[512][1]cuda:0", arg94_1: "bf16[512, 512][512, 1]cuda:0", arg95_1: "bf16[512, 512][512, 1]cuda:0", arg96_1: "bf16[512, 512][512, 1]cuda:0", arg97_1: "bf16[512, 512][512, 1]cuda:0", arg98_1: "bf16[512][1]cuda:0", arg99_1: "bf16[512, 512][512, 1]cuda:0", arg100_1: "bf16[512, 512][512, 1]cuda:0", arg101_1: "bf16[512, 512][512, 1]cuda:0", arg102_1: "bf16[512, 512][512, 1]cuda:0", arg103_1: "bf16[512][1]cuda:0", arg104_1: "bf16[2048, 512][512, 1]cuda:0", arg105_1: "bf16[512, 2048][2048, 1]cuda:0", arg106_1: "bf16[512][1]cuda:0", arg107_1: "bf16[512, 512][512, 1]cuda:0", arg108_1: "bf16[512, 512][512, 1]cuda:0", arg109_1: "bf16[512, 512][512, 1]cuda:0", arg110_1: "bf16[512, 512][512, 1]cuda:0", arg111_1: "bf16[512][1]cuda:0", arg112_1: "bf16[512, 512][512, 1]cuda:0", arg113_1: "bf16[512, 512][512, 1]cuda:0", arg114_1: "bf16[512, 512][512, 1]cuda:0", arg115_1: "bf16[512, 512][512, 1]cuda:0", arg116_1: "bf16[512][1]cuda:0", arg117_1: "bf16[2048, 512][512, 1]cuda:0", arg118_1: "bf16[512, 2048][2048, 1]cuda:0", arg119_1: "bf16[512][1]cuda:0", arg120_1: "bf16[512, 512][512, 1]cuda:0", arg121_1: "bf16[512, 512][512, 1]cuda:0", arg122_1: "bf16[512, 512][512, 1]cuda:0", arg123_1: "bf16[512, 512][512, 1]cuda:0", arg124_1: "bf16[512][1]cuda:0", arg125_1: "bf16[512, 512][512, 1]cuda:0", arg126_1: "bf16[512, 512][512, 1]cuda:0", arg127_1: "bf16[512, 512][512, 1]cuda:0", arg128_1: "bf16[512, 512][512, 1]cuda:0", arg129_1: "bf16[512][1]cuda:0", arg130_1: "bf16[2048, 512][512, 1]cuda:0", arg131_1: "bf16[512, 2048][2048, 1]cuda:0", arg132_1: "bf16[512][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_2: "i64[1024][1]cuda:0" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[1024][1]cuda:0" = torch.ops.aten.add.Tensor(iota_2, 0);  iota_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze: "i64[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add, 0);  add = None
        unsqueeze_1: "i64[1, 1, 1024][1024, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "i64[1, 1, 1024, 1][1024, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:87 in bidirectional_mask_function, code: return q_idx >= 0
        ge: "b8[1, 1, 1024, 1][1024, 1024, 1, 1]cuda:0" = torch.ops.aten.ge.Scalar(unsqueeze_2, 0);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand: "b8[8, 1, 1024, 1024][0, 1024, 1, 0]cuda:0" = torch.ops.aten.expand.default(ge, [8, -1, 1024, 1024]);  ge = expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        _tensor_constant0: "bf16[][]cuda:0" = self._tensor_constant0
        lift_fresh_copy: "bf16[][]cuda:0" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = lift_fresh_copy = None
        scalar_tensor: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor = None
        _tensor_constant1: "bf16[][]cuda:0" = self._tensor_constant1;  _tensor_constant1 = None
        _tensor_constant2: "bf16[][]cuda:0" = self._tensor_constant2

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_12: "i64[1024][1]cuda:0" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_40: "i64[1024][1]cuda:0" = torch.ops.aten.add.Tensor(iota_12, 0);  iota_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_12: "i64[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_40, 0);  add_40 = None
        unsqueeze_13: "i64[1, 1, 1024][1024, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_12, 1);  unsqueeze_12 = None
        unsqueeze_14: "i64[1, 1, 1024, 1][1024, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 3);  unsqueeze_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:87 in bidirectional_mask_function, code: return q_idx >= 0
        ge_1: "b8[1, 1, 1024, 1][1024, 1024, 1, 1]cuda:0" = torch.ops.aten.ge.Scalar(unsqueeze_14, 0);  unsqueeze_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_26: "b8[8, 1, 1024, 1024][0, 1024, 1, 0]cuda:0" = torch.ops.aten.expand.default(ge_1, [8, -1, 1024, 1024]);  ge_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        lift_fresh_copy_2: "bf16[][]cuda:0" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant2);  _tensor_constant2 = None
        scalar_tensor_2: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_3: "bf16[8, 1, 1024, 1024][1048576, 1048576, 1024, 1]cuda:0" = torch.ops.aten.where.self(expand_26, lift_fresh_copy_2, scalar_tensor_2);  expand_26 = lift_fresh_copy_2 = scalar_tensor_2 = where_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:300 in forward, code: position_bias = torch.zeros(
        full_3: "bf16[1, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.full.default([1, 8, 1024, 1024], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        full_default_3: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.full.default([8, 8, 1024, 1024], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1104 in forward, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        view_413: "i64[8192][1]cuda:0" = torch.ops.aten.reshape.default(arg52_1, [-1]);  arg52_1 = None
        ne_1: "b8[8192][1]cuda:0" = torch.ops.aten.ne.Scalar(view_413, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:669 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding_2: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.embedding.default(arg1_1, arg0_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_137: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(embedding_2, torch.float32)
        pow_14: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_137, 2);  convert_element_type_137 = None
        mean_13: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_14, [-1], True);  pow_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_42: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_13, 1e-06);  mean_13 = None
        rsqrt_13: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_28: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(embedding_2, rsqrt_13);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_138: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_28, torch.bfloat16);  mul_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_29: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg53_1, convert_element_type_138);  arg53_1 = convert_element_type_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_146: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_29, [8192, 512])
        permute_67: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        mm_36: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_146, permute_67);  view_146 = permute_67 = None
        view_147: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_36, [8, 1024, 512]);  mm_36 = None
        view_148: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_147, [8, 1024, -1, 64]);  view_147 = None
        permute_68: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_27: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_68, [8, 8, 1024, 64]);  permute_68 = None
        clone_51: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_155: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_51, [64, 1024, 64]);  clone_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_149: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_29, [8192, 512])
        permute_69: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        mm_37: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_149, permute_69);  view_149 = permute_69 = None
        view_150: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_37, [8, 1024, 512]);  mm_37 = None
        view_151: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_150, [8, 1024, -1, 64]);  view_150 = None
        permute_70: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_151, [0, 2, 1, 3]);  view_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_73: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_70, [0, 1, 3, 2]);  permute_70 = None
        expand_28: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_73, [8, 8, 64, 1024]);  permute_73 = None
        clone_52: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_156: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_52, [64, 64, 1024]);  clone_52 = None
        bmm_12: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_155, view_156);  view_155 = view_156 = None
        view_157: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_12, [8, 8, 1024, 1024]);  bmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:228 in compute_bias, code: memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        iota_15: "i64[1024][1]cuda:0" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_16: "i64[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_15, 0);  iota_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:227 in compute_bias, code: context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None] + past_seen_tokens
        iota_14: "i64[1024][1]cuda:0" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_15: "i64[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_14, 1);  iota_14 = None
        add_43: "i64[1024, 1][1, 1]cuda:0" = torch.ops.aten.add.Tensor(unsqueeze_15, 0);  unsqueeze_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:229 in compute_bias, code: relative_position = memory_position - context_position  # shape (query_length, key_length)
        sub_7: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(unsqueeze_16, add_43);  unsqueeze_16 = add_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:203 in _relative_position_bucket, code: relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        full_1: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.full.default([1024, 1024], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_1: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.minimum.default(sub_7, full_1);  sub_7 = full_1 = None
        neg: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.neg.default(minimum_1);  minimum_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:208 in _relative_position_bucket, code: is_small = relative_position < max_exact
        lt_1: "b8[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.lt.Scalar(neg, 16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:212 in _relative_position_bucket, code: torch.log(relative_position.float() / max_exact)
        convert_element_type_147: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(neg, torch.float32)
        div_8: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_147, 16);  convert_element_type_147 = None
        log_1: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.log.default(div_8);  div_8 = None
        div_9: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.div.Tensor(log_1, 2.0794415416798357);  log_1 = None
        mul_30: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_9, 16);  div_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:215 in _relative_position_bucket, code: ).to(torch.long)
        convert_element_type_148: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_30, torch.int64);  mul_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:211 in _relative_position_bucket, code: relative_position_if_large = max_exact + (
        add_44: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_148, 16);  convert_element_type_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:217 in _relative_position_bucket, code: relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        full_2: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.full.default([1024, 1024], 31, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:216 in _relative_position_bucket, code: relative_position_if_large = torch.min(
        minimum_2: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.minimum.default(add_44, full_2);  add_44 = full_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:220 in _relative_position_bucket, code: relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        where_4: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.where.self(lt_1, neg, minimum_2);  lt_1 = neg = minimum_2 = None
        add_45: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.add.Tensor(where_4, 0);  where_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:236 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        embedding_3: "bf16[1024, 1024, 8][8192, 8, 1]cuda:0" = torch.ops.aten.embedding.default(arg57_1, add_45);  arg57_1 = add_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:237 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        permute_74: "bf16[8, 1024, 1024][1, 8192, 8]cuda:0" = torch.ops.aten.permute.default(embedding_3, [2, 0, 1]);  embedding_3 = None
        unsqueeze_17: "bf16[1, 8, 1024, 1024][8, 1, 8192, 8]cuda:0" = torch.ops.aten.unsqueeze.default(permute_74, 0);  permute_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:512 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        iota_9: "i64[1024][1]cuda:0" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_39: "i64[1024][1]cuda:0" = torch.ops.aten.add.Tensor(iota_9, 0);  iota_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_9: "i64[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_39, 0);  add_39 = None
        unsqueeze_10: "i64[1, 1, 1024][1024, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 1);  unsqueeze_9 = None
        unsqueeze_11: "i64[1, 1, 1, 1024][1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 2);  unsqueeze_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_8: "i64[1024][1]cuda:0" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_38: "i64[1024][1]cuda:0" = torch.ops.aten.add.Tensor(iota_8, 0);  iota_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_6: "i64[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_38, 0);  add_38 = None
        unsqueeze_7: "i64[1, 1, 1024][1024, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 1);  unsqueeze_6 = None
        unsqueeze_8: "i64[1, 1, 1024, 1][1024, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 1024, 1024][1048576, 1048576, 1024, 1]cuda:0" = torch.ops.aten.le.Tensor(unsqueeze_11, unsqueeze_8);  unsqueeze_11 = unsqueeze_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_25: "b8[8, 1, 1024, 1024][0, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(le, [8, -1, 1024, 1024]);  le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default_1: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "bf16[8, 1, 1024, 1024][1048576, 1048576, 1024, 1]cuda:0" = torch.ops.aten.where.self(expand_25, full_default_1, full_default_2);  expand_25 = full_default_1 = full_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        add_46: "bf16[8, 8, 1024, 1024][8388608, 1, 8192, 8]cuda:0" = torch.ops.aten.add.Tensor(unsqueeze_17, where_2);  unsqueeze_17 = where_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_47: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_157, add_46);  view_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_149: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_47, torch.float32);  add_47 = None
        amax_6: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_149, [-1], True)
        sub_8: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_149, amax_6);  convert_element_type_149 = amax_6 = None
        exp_6: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
        sum_7: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_10: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        convert_element_type_150: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_29: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_150, [8, 8, 1024, 1024]);  convert_element_type_150 = None
        view_160: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_29, [64, 1024, 1024]);  expand_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_152: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_29, [8192, 512]);  mul_29 = None
        permute_71: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        mm_38: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_152, permute_71);  view_152 = permute_71 = None
        view_153: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_38, [8, 1024, 512]);  mm_38 = None
        view_154: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_153, [8, 1024, -1, 64]);  view_153 = None
        permute_72: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_154, [0, 2, 1, 3]);  view_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_30: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_72, [8, 8, 1024, 64]);  permute_72 = None
        clone_54: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_161: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_54, [64, 1024, 64]);  clone_54 = None
        bmm_13: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_160, view_161);  view_160 = view_161 = None
        view_162: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_13, [8, 8, 1024, 64]);  bmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_75: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
        clone_55: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_75, memory_format = torch.contiguous_format);  permute_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_163: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_55, [8, 1024, -1]);  clone_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_164: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_163, [8192, 512]);  view_163 = None
        permute_76: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        mm_39: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_164, permute_76);  view_164 = permute_76 = None
        view_165: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_39, [8, 1024, 512]);  mm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_48: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(embedding_2, view_165);  embedding_2 = view_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_155: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_48, torch.float32)
        pow_15: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_155, 2);  convert_element_type_155 = None
        mean_14: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_15, [-1], True);  pow_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_49: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_14, 1e-06);  mean_14 = None
        rsqrt_14: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        mul_31: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_48, rsqrt_14);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_156: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_31, torch.bfloat16);  mul_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_32: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg59_1, convert_element_type_156);  arg59_1 = convert_element_type_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_166: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_32, [8192, 512]);  mul_32 = None
        permute_77: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        mm_40: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_166, permute_77);  view_166 = permute_77 = None
        view_167: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_40, [8, 1024, 512]);  mm_40 = None
        view_168: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_167, [8, 1024, -1, 64]);  view_167 = None
        permute_78: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_31: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_78, [8, 8, 1024, 64]);  permute_78 = None
        clone_57: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_175: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_57, [64, 1024, 64]);  clone_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:669 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(embedding, torch.float32)
        pow_1: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2);  convert_element_type = None
        mean: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_2: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(embedding, rsqrt);  rsqrt = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_1: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_1: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg2_1, convert_element_type_1);  arg2_1 = convert_element_type_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_1: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_1, [8192, 512])
        permute: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        mm: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_1, permute);  view_1 = permute = None
        view_2: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [8, 1024, 512]);  mm = None
        view_3: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_2, [8, 1024, -1, 64]);  view_2 = None
        permute_1: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_1: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_1, [8, 8, 1024, 64]);  permute_1 = None
        clone_1: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_10: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_1, [64, 1024, 64]);  clone_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_4: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_1, [8192, 512])
        permute_2: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        mm_1: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_4, permute_2);  view_4 = permute_2 = None
        view_5: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_1, [8, 1024, 512]);  mm_1 = None
        view_6: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_5, [8, 1024, -1, 64]);  view_5 = None
        permute_3: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_6: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        expand_2: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_6, [8, 8, 64, 1024]);  permute_6 = None
        clone_2: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_11: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [64, 64, 1024]);  clone_2 = None
        bmm: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_10, view_11);  view_10 = view_11 = None
        view_12: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm, [8, 8, 1024, 1024]);  bmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:228 in compute_bias, code: memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        iota_5: "i64[1024][1]cuda:0" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_4: "i64[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_5, 0);  iota_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:227 in compute_bias, code: context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None] + past_seen_tokens
        iota_4: "i64[1024][1]cuda:0" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_3: "i64[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_4, 1);  iota_4 = None
        add_3: "i64[1024, 1][1, 1]cuda:0" = torch.ops.aten.add.Tensor(unsqueeze_3, 0);  unsqueeze_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:229 in compute_bias, code: relative_position = memory_position - context_position  # shape (query_length, key_length)
        sub: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(unsqueeze_4, add_3);  unsqueeze_4 = add_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:200 in _relative_position_bucket, code: relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        gt: "b8[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(sub, 0)
        convert_element_type_10: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt, torch.int64);  gt = None
        mul_2: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_10, 16);  convert_element_type_10 = None
        add_4: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2, 0);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:201 in _relative_position_bucket, code: relative_position = torch.abs(relative_position)
        abs_1: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.abs.default(sub);  sub = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:208 in _relative_position_bucket, code: is_small = relative_position < max_exact
        lt: "b8[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.lt.Scalar(abs_1, 8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:212 in _relative_position_bucket, code: torch.log(relative_position.float() / max_exact)
        convert_element_type_11: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(abs_1, torch.float32)
        div: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_11, 8);  convert_element_type_11 = None
        log: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.log.default(div);  div = None
        div_1: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.div.Tensor(log, 2.772588722239781);  log = None
        mul_3: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_1, 8);  div_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:215 in _relative_position_bucket, code: ).to(torch.long)
        convert_element_type_12: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3, torch.int64);  mul_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:211 in _relative_position_bucket, code: relative_position_if_large = max_exact + (
        add_5: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_12, 8);  convert_element_type_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:217 in _relative_position_bucket, code: relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        full: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.full.default([1024, 1024], 15, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:216 in _relative_position_bucket, code: relative_position_if_large = torch.min(
        minimum: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.minimum.default(add_5, full);  add_5 = full = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:220 in _relative_position_bucket, code: relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        where_1: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.where.self(lt, abs_1, minimum);  lt = abs_1 = minimum = None
        add_6: "i64[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4, where_1);  add_4 = where_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:236 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        embedding_1: "bf16[1024, 1024, 8][8192, 8, 1]cuda:0" = torch.ops.aten.embedding.default(arg6_1, add_6);  arg6_1 = add_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:237 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        permute_7: "bf16[8, 1024, 1024][1, 8192, 8]cuda:0" = torch.ops.aten.permute.default(embedding_1, [2, 0, 1]);  embedding_1 = None
        unsqueeze_5: "bf16[1, 8, 1024, 1024][8, 1, 8192, 8]cuda:0" = torch.ops.aten.unsqueeze.default(permute_7, 0);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default: "bf16[8, 1, 1024, 1024][1048576, 1048576, 1024, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 1024, 1024], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        add_7: "bf16[8, 8, 1024, 1024][8388608, 1, 8192, 8]cuda:0" = torch.ops.aten.add.Tensor(unsqueeze_5, full_default);  unsqueeze_5 = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_8: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_12, add_7);  view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_13: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_8, torch.float32);  add_8 = None
        amax: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_13, [-1], True)
        sub_1: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_13, amax);  convert_element_type_13 = amax = None
        exp: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_2: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_14: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_3: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_14, [8, 8, 1024, 1024]);  convert_element_type_14 = None
        view_15: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_3, [64, 1024, 1024]);  expand_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_7: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_1, [8192, 512]);  mul_1 = None
        permute_4: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        mm_2: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_7, permute_4);  view_7 = permute_4 = None
        view_8: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [8, 1024, 512]);  mm_2 = None
        view_9: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_8, [8, 1024, -1, 64]);  view_8 = None
        permute_5: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_4: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_5, [8, 8, 1024, 64]);  permute_5 = None
        clone_4: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_16: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_4, [64, 1024, 64]);  clone_4 = None
        bmm_1: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_15, view_16);  view_15 = view_16 = None
        view_17: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_1, [8, 8, 1024, 64]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_8: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None
        clone_5: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_18: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_5, [8, 1024, -1]);  clone_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_19: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_18, [8192, 512]);  view_18 = None
        permute_9: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        mm_3: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_19, permute_9);  view_19 = permute_9 = None
        view_20: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_3, [8, 1024, 512]);  mm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_9: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(embedding, view_20);  embedding = view_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_19: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_9, torch.float32)
        pow_2: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_19, 2);  convert_element_type_19 = None
        mean_1: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_10: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        rsqrt_1: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_4: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_9, rsqrt_1);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_20: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_5: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg8_1, convert_element_type_20);  arg8_1 = convert_element_type_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_21: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_5, [8192, 512]);  mul_5 = None
        permute_10: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        mm_4: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_21, permute_10);  view_21 = permute_10 = None
        view_22: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [8, 1024, 2048]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.relu.default(view_22);  view_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_23: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(relu, [8192, 2048]);  relu = None
        permute_11: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        mm_5: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_23, permute_11);  view_23 = permute_11 = None
        view_24: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_5, [8, 1024, 512]);  mm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_11: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_9, view_24);  add_9 = view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_25: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_11, torch.float32)
        pow_3: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_25, 2);  convert_element_type_25 = None
        mean_2: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_12: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
        rsqrt_2: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_6: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_11, rsqrt_2);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_26: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_7: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg11_1, convert_element_type_26);  arg11_1 = convert_element_type_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_25: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_7, [8192, 512])
        permute_12: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        mm_6: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_25, permute_12);  view_25 = permute_12 = None
        view_26: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [8, 1024, 512]);  mm_6 = None
        view_27: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_26, [8, 1024, -1, 64]);  view_26 = None
        permute_13: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_5: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_13, [8, 8, 1024, 64]);  permute_13 = None
        clone_9: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_34: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_9, [64, 1024, 64]);  clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_28: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_7, [8192, 512])
        permute_14: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        mm_7: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_28, permute_14);  view_28 = permute_14 = None
        view_29: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_7, [8, 1024, 512]);  mm_7 = None
        view_30: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_29, [8, 1024, -1, 64]);  view_29 = None
        permute_15: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_18: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_15, [0, 1, 3, 2]);  permute_15 = None
        expand_6: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_18, [8, 8, 64, 1024]);  permute_18 = None
        clone_10: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_35: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_10, [64, 64, 1024]);  clone_10 = None
        bmm_2: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_34, view_35);  view_34 = view_35 = None
        view_36: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_2, [8, 8, 1024, 1024]);  bmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_13: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_36, add_7);  view_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_35: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_13, torch.float32);  add_13 = None
        amax_1: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_35, [-1], True)
        sub_2: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_35, amax_1);  convert_element_type_35 = amax_1 = None
        exp_1: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_2: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_3: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        convert_element_type_36: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_7: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_36, [8, 8, 1024, 1024]);  convert_element_type_36 = None
        view_39: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_7, [64, 1024, 1024]);  expand_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_31: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_7, [8192, 512]);  mul_7 = None
        permute_16: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        mm_8: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_31, permute_16);  view_31 = permute_16 = None
        view_32: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [8, 1024, 512]);  mm_8 = None
        view_33: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_32, [8, 1024, -1, 64]);  view_32 = None
        permute_17: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_8: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_17, [8, 8, 1024, 64]);  permute_17 = None
        clone_12: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_40: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_12, [64, 1024, 64]);  clone_12 = None
        bmm_3: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_39, view_40);  view_39 = view_40 = None
        view_41: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_3, [8, 8, 1024, 64]);  bmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_19: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
        clone_13: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_42: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_13, [8, 1024, -1]);  clone_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_43: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_42, [8192, 512]);  view_42 = None
        permute_20: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        mm_9: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_43, permute_20);  view_43 = permute_20 = None
        view_44: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_9, [8, 1024, 512]);  mm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_14: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_11, view_44);  add_11 = view_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_41: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_14, torch.float32)
        pow_4: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_41, 2);  convert_element_type_41 = None
        mean_3: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_15: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None
        rsqrt_3: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        mul_8: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_14, rsqrt_3);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_42: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16);  mul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_9: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg16_1, convert_element_type_42);  arg16_1 = convert_element_type_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_45: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_9, [8192, 512]);  mul_9 = None
        permute_21: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        mm_10: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_45, permute_21);  view_45 = permute_21 = None
        view_46: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [8, 1024, 2048]);  mm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_1: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.relu.default(view_46);  view_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_47: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(relu_1, [8192, 2048]);  relu_1 = None
        permute_22: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        mm_11: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_47, permute_22);  view_47 = permute_22 = None
        view_48: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_11, [8, 1024, 512]);  mm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_16: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_14, view_48);  add_14 = view_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_47: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_16, torch.float32)
        pow_5: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_47, 2);  convert_element_type_47 = None
        mean_4: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_17: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_4, 1e-06);  mean_4 = None
        rsqrt_4: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        mul_10: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_16, rsqrt_4);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_48: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_10, torch.bfloat16);  mul_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_11: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg19_1, convert_element_type_48);  arg19_1 = convert_element_type_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_49: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_11, [8192, 512])
        permute_23: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        mm_12: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_49, permute_23);  view_49 = permute_23 = None
        view_50: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_12, [8, 1024, 512]);  mm_12 = None
        view_51: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_50, [8, 1024, -1, 64]);  view_50 = None
        permute_24: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_9: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_24, [8, 8, 1024, 64]);  permute_24 = None
        clone_17: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_58: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_17, [64, 1024, 64]);  clone_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_52: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_11, [8192, 512])
        permute_25: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        mm_13: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_52, permute_25);  view_52 = permute_25 = None
        view_53: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_13, [8, 1024, 512]);  mm_13 = None
        view_54: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_53, [8, 1024, -1, 64]);  view_53 = None
        permute_26: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_29: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_26, [0, 1, 3, 2]);  permute_26 = None
        expand_10: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_29, [8, 8, 64, 1024]);  permute_29 = None
        clone_18: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_59: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_18, [64, 64, 1024]);  clone_18 = None
        bmm_4: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_58, view_59);  view_58 = view_59 = None
        view_60: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_4, [8, 8, 1024, 1024]);  bmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_18: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_60, add_7);  view_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_57: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_18, torch.float32);  add_18 = None
        amax_2: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_57, [-1], True)
        sub_3: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_57, amax_2);  convert_element_type_57 = amax_2 = None
        exp_2: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        sum_3: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_4: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        convert_element_type_58: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_11: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_58, [8, 8, 1024, 1024]);  convert_element_type_58 = None
        view_63: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_11, [64, 1024, 1024]);  expand_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_55: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_11, [8192, 512]);  mul_11 = None
        permute_27: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        mm_14: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_55, permute_27);  view_55 = permute_27 = None
        view_56: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_14, [8, 1024, 512]);  mm_14 = None
        view_57: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_56, [8, 1024, -1, 64]);  view_56 = None
        permute_28: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_57, [0, 2, 1, 3]);  view_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_12: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_28, [8, 8, 1024, 64]);  permute_28 = None
        clone_20: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_64: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_20, [64, 1024, 64]);  clone_20 = None
        bmm_5: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_63, view_64);  view_63 = view_64 = None
        view_65: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_5, [8, 8, 1024, 64]);  bmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_30: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None
        clone_21: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_66: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_21, [8, 1024, -1]);  clone_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_67: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_66, [8192, 512]);  view_66 = None
        permute_31: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        mm_15: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_67, permute_31);  view_67 = permute_31 = None
        view_68: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_15, [8, 1024, 512]);  mm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_19: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_16, view_68);  add_16 = view_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_63: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_19, torch.float32)
        pow_6: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_63, 2);  convert_element_type_63 = None
        mean_5: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_6, [-1], True);  pow_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_20: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_5, 1e-06);  mean_5 = None
        rsqrt_5: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_12: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_19, rsqrt_5);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_64: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_12, torch.bfloat16);  mul_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_13: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg24_1, convert_element_type_64);  arg24_1 = convert_element_type_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_69: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_13, [8192, 512]);  mul_13 = None
        permute_32: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        mm_16: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_69, permute_32);  view_69 = permute_32 = None
        view_70: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_16, [8, 1024, 2048]);  mm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_2: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.relu.default(view_70);  view_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_71: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(relu_2, [8192, 2048]);  relu_2 = None
        permute_33: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        mm_17: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_71, permute_33);  view_71 = permute_33 = None
        view_72: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_17, [8, 1024, 512]);  mm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_21: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_19, view_72);  add_19 = view_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_69: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_21, torch.float32)
        pow_7: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_69, 2);  convert_element_type_69 = None
        mean_6: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_22: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_6, 1e-06);  mean_6 = None
        rsqrt_6: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        mul_14: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_21, rsqrt_6);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_70: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_14, torch.bfloat16);  mul_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_15: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg27_1, convert_element_type_70);  arg27_1 = convert_element_type_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_73: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_15, [8192, 512])
        permute_34: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        mm_18: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_73, permute_34);  view_73 = permute_34 = None
        view_74: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_18, [8, 1024, 512]);  mm_18 = None
        view_75: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_74, [8, 1024, -1, 64]);  view_74 = None
        permute_35: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_13: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_35, [8, 8, 1024, 64]);  permute_35 = None
        clone_25: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_82: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_25, [64, 1024, 64]);  clone_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_76: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_15, [8192, 512])
        permute_36: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        mm_19: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_76, permute_36);  view_76 = permute_36 = None
        view_77: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_19, [8, 1024, 512]);  mm_19 = None
        view_78: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_77, [8, 1024, -1, 64]);  view_77 = None
        permute_37: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_40: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_37, [0, 1, 3, 2]);  permute_37 = None
        expand_14: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_40, [8, 8, 64, 1024]);  permute_40 = None
        clone_26: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_83: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_26, [64, 64, 1024]);  clone_26 = None
        bmm_6: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_82, view_83);  view_82 = view_83 = None
        view_84: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_6, [8, 8, 1024, 1024]);  bmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_23: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_84, add_7);  view_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_79: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_23, torch.float32);  add_23 = None
        amax_3: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_79, [-1], True)
        sub_4: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_79, amax_3);  convert_element_type_79 = amax_3 = None
        exp_3: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_4: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_5: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        convert_element_type_80: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_15: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_80, [8, 8, 1024, 1024]);  convert_element_type_80 = None
        view_87: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_15, [64, 1024, 1024]);  expand_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_79: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_15, [8192, 512]);  mul_15 = None
        permute_38: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        mm_20: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_79, permute_38);  view_79 = permute_38 = None
        view_80: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_20, [8, 1024, 512]);  mm_20 = None
        view_81: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_80, [8, 1024, -1, 64]);  view_80 = None
        permute_39: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_81, [0, 2, 1, 3]);  view_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_16: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_39, [8, 8, 1024, 64]);  permute_39 = None
        clone_28: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_88: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_28, [64, 1024, 64]);  clone_28 = None
        bmm_7: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_87, view_88);  view_87 = view_88 = None
        view_89: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_7, [8, 8, 1024, 64]);  bmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_41: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None
        clone_29: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_90: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_29, [8, 1024, -1]);  clone_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_91: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_90, [8192, 512]);  view_90 = None
        permute_42: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        mm_21: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_91, permute_42);  view_91 = permute_42 = None
        view_92: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_21, [8, 1024, 512]);  mm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_24: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_21, view_92);  add_21 = view_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_85: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_24, torch.float32)
        pow_8: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_85, 2);  convert_element_type_85 = None
        mean_7: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_8, [-1], True);  pow_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_25: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_7, 1e-06);  mean_7 = None
        rsqrt_7: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        mul_16: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_24, rsqrt_7);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_86: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_16, torch.bfloat16);  mul_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_17: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg32_1, convert_element_type_86);  arg32_1 = convert_element_type_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_93: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_17, [8192, 512]);  mul_17 = None
        permute_43: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        mm_22: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_93, permute_43);  view_93 = permute_43 = None
        view_94: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_22, [8, 1024, 2048]);  mm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_3: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.relu.default(view_94);  view_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_95: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(relu_3, [8192, 2048]);  relu_3 = None
        permute_44: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        mm_23: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_95, permute_44);  view_95 = permute_44 = None
        view_96: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_23, [8, 1024, 512]);  mm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_26: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_24, view_96);  add_24 = view_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_91: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_26, torch.float32)
        pow_9: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_91, 2);  convert_element_type_91 = None
        mean_8: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_9, [-1], True);  pow_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_27: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_8, 1e-06);  mean_8 = None
        rsqrt_8: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        mul_18: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_26, rsqrt_8);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_92: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_18, torch.bfloat16);  mul_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_19: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg35_1, convert_element_type_92);  arg35_1 = convert_element_type_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_97: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_19, [8192, 512])
        permute_45: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        mm_24: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_97, permute_45);  view_97 = permute_45 = None
        view_98: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_24, [8, 1024, 512]);  mm_24 = None
        view_99: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_98, [8, 1024, -1, 64]);  view_98 = None
        permute_46: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_17: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_46, [8, 8, 1024, 64]);  permute_46 = None
        clone_33: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_106: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_33, [64, 1024, 64]);  clone_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_100: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_19, [8192, 512])
        permute_47: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        mm_25: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_100, permute_47);  view_100 = permute_47 = None
        view_101: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_25, [8, 1024, 512]);  mm_25 = None
        view_102: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_101, [8, 1024, -1, 64]);  view_101 = None
        permute_48: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_51: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_48, [0, 1, 3, 2]);  permute_48 = None
        expand_18: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_51, [8, 8, 64, 1024]);  permute_51 = None
        clone_34: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_107: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_34, [64, 64, 1024]);  clone_34 = None
        bmm_8: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_106, view_107);  view_106 = view_107 = None
        view_108: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_8, [8, 8, 1024, 1024]);  bmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_28: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_108, add_7);  view_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_101: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_28, torch.float32);  add_28 = None
        amax_4: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_101, [-1], True)
        sub_5: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_101, amax_4);  convert_element_type_101 = amax_4 = None
        exp_4: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        sum_5: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_6: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        convert_element_type_102: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_19: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_102, [8, 8, 1024, 1024]);  convert_element_type_102 = None
        view_111: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_19, [64, 1024, 1024]);  expand_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_103: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_19, [8192, 512]);  mul_19 = None
        permute_49: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        mm_26: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_103, permute_49);  view_103 = permute_49 = None
        view_104: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_26, [8, 1024, 512]);  mm_26 = None
        view_105: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_104, [8, 1024, -1, 64]);  view_104 = None
        permute_50: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_105, [0, 2, 1, 3]);  view_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_20: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_50, [8, 8, 1024, 64]);  permute_50 = None
        clone_36: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_112: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_36, [64, 1024, 64]);  clone_36 = None
        bmm_9: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_111, view_112);  view_111 = view_112 = None
        view_113: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_9, [8, 8, 1024, 64]);  bmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_52: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
        clone_37: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_114: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_37, [8, 1024, -1]);  clone_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_115: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_114, [8192, 512]);  view_114 = None
        permute_53: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        mm_27: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_115, permute_53);  view_115 = permute_53 = None
        view_116: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_27, [8, 1024, 512]);  mm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_29: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_26, view_116);  add_26 = view_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_107: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_29, torch.float32)
        pow_10: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_107, 2);  convert_element_type_107 = None
        mean_9: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_30: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_9, 1e-06);  mean_9 = None
        rsqrt_9: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_20: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_29, rsqrt_9);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_108: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_20, torch.bfloat16);  mul_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_21: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg40_1, convert_element_type_108);  arg40_1 = convert_element_type_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_117: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_21, [8192, 512]);  mul_21 = None
        permute_54: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        mm_28: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_117, permute_54);  view_117 = permute_54 = None
        view_118: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_28, [8, 1024, 2048]);  mm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_4: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.relu.default(view_118);  view_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_119: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(relu_4, [8192, 2048]);  relu_4 = None
        permute_55: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        mm_29: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_119, permute_55);  view_119 = permute_55 = None
        view_120: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_29, [8, 1024, 512]);  mm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_31: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_29, view_120);  add_29 = view_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_113: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_31, torch.float32)
        pow_11: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_113, 2);  convert_element_type_113 = None
        mean_10: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_32: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_10, 1e-06);  mean_10 = None
        rsqrt_10: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        mul_22: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_31, rsqrt_10);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_114: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_22, torch.bfloat16);  mul_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_23: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg43_1, convert_element_type_114);  arg43_1 = convert_element_type_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_121: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_23, [8192, 512])
        permute_56: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        mm_30: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_121, permute_56);  view_121 = permute_56 = None
        view_122: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_30, [8, 1024, 512]);  mm_30 = None
        view_123: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_122, [8, 1024, -1, 64]);  view_122 = None
        permute_57: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_21: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_57, [8, 8, 1024, 64]);  permute_57 = None
        clone_41: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_130: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_41, [64, 1024, 64]);  clone_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_124: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_23, [8192, 512])
        permute_58: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        mm_31: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_124, permute_58);  view_124 = permute_58 = None
        view_125: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_31, [8, 1024, 512]);  mm_31 = None
        view_126: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_125, [8, 1024, -1, 64]);  view_125 = None
        permute_59: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_62: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_59, [0, 1, 3, 2]);  permute_59 = None
        expand_22: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_62, [8, 8, 64, 1024]);  permute_62 = None
        clone_42: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_131: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_42, [64, 64, 1024]);  clone_42 = None
        bmm_10: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_130, view_131);  view_130 = view_131 = None
        view_132: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_10, [8, 8, 1024, 1024]);  bmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_33: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_132, add_7);  view_132 = add_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_123: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_33, torch.float32);  add_33 = None
        amax_5: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_123, [-1], True)
        sub_6: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_123, amax_5);  convert_element_type_123 = amax_5 = None
        exp_5: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_6: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_7: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        convert_element_type_124: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_23: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_124, [8, 8, 1024, 1024]);  convert_element_type_124 = None
        view_135: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_23, [64, 1024, 1024]);  expand_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_127: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_23, [8192, 512]);  mul_23 = None
        permute_60: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        mm_32: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_127, permute_60);  view_127 = permute_60 = None
        view_128: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_32, [8, 1024, 512]);  mm_32 = None
        view_129: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_128, [8, 1024, -1, 64]);  view_128 = None
        permute_61: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_24: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_61, [8, 8, 1024, 64]);  permute_61 = None
        clone_44: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_136: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_44, [64, 1024, 64]);  clone_44 = None
        bmm_11: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_135, view_136);  view_135 = view_136 = None
        view_137: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_11, [8, 8, 1024, 64]);  bmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_63: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None
        clone_45: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_138: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_45, [8, 1024, -1]);  clone_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_139: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_138, [8192, 512]);  view_138 = None
        permute_64: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        mm_33: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_139, permute_64);  view_139 = permute_64 = None
        view_140: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_33, [8, 1024, 512]);  mm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_34: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_31, view_140);  add_31 = view_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_129: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_34, torch.float32)
        pow_12: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_129, 2);  convert_element_type_129 = None
        mean_11: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_12, [-1], True);  pow_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_35: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_11, 1e-06);  mean_11 = None
        rsqrt_11: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        mul_24: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_34, rsqrt_11);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_130: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_24, torch.bfloat16);  mul_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_25: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg48_1, convert_element_type_130);  arg48_1 = convert_element_type_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_141: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_25, [8192, 512]);  mul_25 = None
        permute_65: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        mm_34: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_141, permute_65);  view_141 = permute_65 = None
        view_142: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_34, [8, 1024, 2048]);  mm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_5: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.relu.default(view_142);  view_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_143: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(relu_5, [8192, 2048]);  relu_5 = None
        permute_66: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        mm_35: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_143, permute_66);  view_143 = permute_66 = None
        view_144: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_35, [8, 1024, 512]);  mm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_36: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_34, view_144);  add_34 = view_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_135: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_36, torch.float32)
        pow_13: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_135, 2);  convert_element_type_135 = None
        mean_12: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_37: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_12, 1e-06);  mean_12 = None
        rsqrt_12: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        mul_26: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_36, rsqrt_12);  add_36 = rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_136: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_26, torch.bfloat16);  mul_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_27: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg51_1, convert_element_type_136);  arg51_1 = convert_element_type_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_169: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_27, [8192, 512])
        permute_79: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        mm_41: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_169, permute_79);  view_169 = permute_79 = None
        view_170: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_41, [8, 1024, 512]);  mm_41 = None
        view_171: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_170, [8, 1024, -1, 64]);  view_170 = None
        permute_80: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_171, [0, 2, 1, 3]);  view_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_83: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_80, [0, 1, 3, 2]);  permute_80 = None
        expand_32: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_83, [8, 8, 64, 1024]);  permute_83 = None
        clone_58: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_176: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_58, [64, 64, 1024]);  clone_58 = None
        bmm_14: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_175, view_176);  view_175 = view_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_177: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_14, [8, 8, 1024, 1024]);  bmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_165: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_177, torch.float32);  view_177 = None
        amax_7: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_165, [-1], True)
        sub_9: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_165, amax_7);  convert_element_type_165 = amax_7 = None
        exp_7: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        sum_8: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_11: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        convert_element_type_166: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_33: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_166, [8, 8, 1024, 1024]);  convert_element_type_166 = None
        view_180: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_33, [64, 1024, 1024]);  expand_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_172: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_27, [8192, 512])
        permute_81: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
        mm_42: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_172, permute_81);  view_172 = permute_81 = None
        view_173: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_42, [8, 1024, 512]);  mm_42 = None
        view_174: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_173, [8, 1024, -1, 64]);  view_173 = None
        permute_82: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_174, [0, 2, 1, 3]);  view_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_34: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_82, [8, 8, 1024, 64]);  permute_82 = None
        clone_60: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_181: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_60, [64, 1024, 64]);  clone_60 = None
        bmm_15: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_180, view_181);  view_180 = view_181 = None
        view_182: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_15, [8, 8, 1024, 64]);  bmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_84: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
        clone_61: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_183: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_61, [8, 1024, -1]);  clone_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_184: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_183, [8192, 512]);  view_183 = None
        permute_85: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        mm_43: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_184, permute_85);  view_184 = permute_85 = None
        view_185: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_43, [8, 1024, 512]);  mm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_52: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_48, view_185);  add_48 = view_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_171: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_52, torch.float32)
        pow_16: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_171, 2);  convert_element_type_171 = None
        mean_15: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_53: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_15, 1e-06);  mean_15 = None
        rsqrt_15: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        mul_33: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_52, rsqrt_15);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_172: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_33, torch.bfloat16);  mul_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_34: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg64_1, convert_element_type_172);  arg64_1 = convert_element_type_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_186: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_34, [8192, 512]);  mul_34 = None
        permute_86: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        mm_44: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_186, permute_86);  view_186 = permute_86 = None
        view_187: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_44, [8, 1024, 2048]);  mm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_6: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.relu.default(view_187);  view_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_188: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(relu_6, [8192, 2048]);  relu_6 = None
        permute_87: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        mm_45: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_188, permute_87);  view_188 = permute_87 = None
        view_189: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_45, [8, 1024, 512]);  mm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_54: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_52, view_189);  add_52 = view_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_177: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_54, torch.float32)
        pow_17: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_177, 2);  convert_element_type_177 = None
        mean_16: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_55: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_16, 1e-06);  mean_16 = None
        rsqrt_16: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
        mul_35: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_54, rsqrt_16);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_178: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_35, torch.bfloat16);  mul_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_36: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg67_1, convert_element_type_178);  arg67_1 = convert_element_type_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_190: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_36, [8192, 512])
        permute_88: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        mm_46: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_190, permute_88);  view_190 = permute_88 = None
        view_191: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_46, [8, 1024, 512]);  mm_46 = None
        view_192: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_191, [8, 1024, -1, 64]);  view_191 = None
        permute_89: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_35: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_89, [8, 8, 1024, 64]);  permute_89 = None
        clone_65: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_199: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_65, [64, 1024, 64]);  clone_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_193: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_36, [8192, 512])
        permute_90: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        mm_47: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_193, permute_90);  view_193 = permute_90 = None
        view_194: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_47, [8, 1024, 512]);  mm_47 = None
        view_195: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_194, [8, 1024, -1, 64]);  view_194 = None
        permute_91: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_195, [0, 2, 1, 3]);  view_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_94: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_91, [0, 1, 3, 2]);  permute_91 = None
        expand_36: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_94, [8, 8, 64, 1024]);  permute_94 = None
        clone_66: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_200: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_66, [64, 64, 1024]);  clone_66 = None
        bmm_16: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_199, view_200);  view_199 = view_200 = None
        view_201: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_16, [8, 8, 1024, 1024]);  bmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_56: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_201, add_46);  view_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_187: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_56, torch.float32);  add_56 = None
        amax_8: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_187, [-1], True)
        sub_10: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_187, amax_8);  convert_element_type_187 = amax_8 = None
        exp_8: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_9: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_12: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        convert_element_type_188: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_12, torch.bfloat16);  div_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_37: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_188, [8, 8, 1024, 1024]);  convert_element_type_188 = None
        view_204: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_37, [64, 1024, 1024]);  expand_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_196: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_36, [8192, 512]);  mul_36 = None
        permute_92: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        mm_48: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_196, permute_92);  view_196 = permute_92 = None
        view_197: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_48, [8, 1024, 512]);  mm_48 = None
        view_198: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_197, [8, 1024, -1, 64]);  view_197 = None
        permute_93: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_198, [0, 2, 1, 3]);  view_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_38: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_93, [8, 8, 1024, 64]);  permute_93 = None
        clone_68: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_205: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_68, [64, 1024, 64]);  clone_68 = None
        bmm_17: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_204, view_205);  view_204 = view_205 = None
        view_206: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_17, [8, 8, 1024, 64]);  bmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_95: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
        clone_69: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_207: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_69, [8, 1024, -1]);  clone_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_208: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_207, [8192, 512]);  view_207 = None
        permute_96: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        mm_49: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_208, permute_96);  view_208 = permute_96 = None
        view_209: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_49, [8, 1024, 512]);  mm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_57: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_54, view_209);  add_54 = view_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_193: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_57, torch.float32)
        pow_18: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_193, 2);  convert_element_type_193 = None
        mean_17: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_18, [-1], True);  pow_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_58: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_17, 1e-06);  mean_17 = None
        rsqrt_17: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_37: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_57, rsqrt_17);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_194: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_37, torch.bfloat16);  mul_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_38: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg72_1, convert_element_type_194);  arg72_1 = convert_element_type_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_210: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_38, [8192, 512]);  mul_38 = None
        permute_97: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        mm_50: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_210, permute_97);  view_210 = permute_97 = None
        view_211: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_50, [8, 1024, 512]);  mm_50 = None
        view_212: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_211, [8, 1024, -1, 64]);  view_211 = None
        permute_98: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_39: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_98, [8, 8, 1024, 64]);  permute_98 = None
        clone_71: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_219: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_71, [64, 1024, 64]);  clone_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_213: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_27, [8192, 512])
        permute_99: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        mm_51: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_213, permute_99);  view_213 = permute_99 = None
        view_214: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_51, [8, 1024, 512]);  mm_51 = None
        view_215: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_214, [8, 1024, -1, 64]);  view_214 = None
        permute_100: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_215, [0, 2, 1, 3]);  view_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_103: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_100, [0, 1, 3, 2]);  permute_100 = None
        expand_40: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_103, [8, 8, 64, 1024]);  permute_103 = None
        clone_72: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_220: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_72, [64, 64, 1024]);  clone_72 = None
        bmm_18: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_219, view_220);  view_219 = view_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_221: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_18, [8, 8, 1024, 1024]);  bmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_203: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_221, torch.float32);  view_221 = None
        amax_9: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_203, [-1], True)
        sub_11: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_203, amax_9);  convert_element_type_203 = amax_9 = None
        exp_9: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        sum_10: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_13: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        convert_element_type_204: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_41: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_204, [8, 8, 1024, 1024]);  convert_element_type_204 = None
        view_224: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_41, [64, 1024, 1024]);  expand_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_216: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_27, [8192, 512])
        permute_101: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        mm_52: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_216, permute_101);  view_216 = permute_101 = None
        view_217: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_52, [8, 1024, 512]);  mm_52 = None
        view_218: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_217, [8, 1024, -1, 64]);  view_217 = None
        permute_102: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_218, [0, 2, 1, 3]);  view_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_42: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_102, [8, 8, 1024, 64]);  permute_102 = None
        clone_74: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_225: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_74, [64, 1024, 64]);  clone_74 = None
        bmm_19: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_224, view_225);  view_224 = view_225 = None
        view_226: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_19, [8, 8, 1024, 64]);  bmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_104: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
        clone_75: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_227: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_75, [8, 1024, -1]);  clone_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_228: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_227, [8192, 512]);  view_227 = None
        permute_105: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        mm_53: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_228, permute_105);  view_228 = permute_105 = None
        view_229: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_53, [8, 1024, 512]);  mm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_60: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_57, view_229);  add_57 = view_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_209: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_60, torch.float32)
        pow_19: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_209, 2);  convert_element_type_209 = None
        mean_18: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_61: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_18, 1e-06);  mean_18 = None
        rsqrt_18: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        mul_39: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_60, rsqrt_18);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_210: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_39, torch.bfloat16);  mul_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_40: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg77_1, convert_element_type_210);  arg77_1 = convert_element_type_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_230: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_40, [8192, 512]);  mul_40 = None
        permute_106: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        mm_54: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_230, permute_106);  view_230 = permute_106 = None
        view_231: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_54, [8, 1024, 2048]);  mm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_7: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.relu.default(view_231);  view_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_232: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(relu_7, [8192, 2048]);  relu_7 = None
        permute_107: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        mm_55: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_232, permute_107);  view_232 = permute_107 = None
        view_233: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_55, [8, 1024, 512]);  mm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_62: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_60, view_233);  add_60 = view_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_215: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_62, torch.float32)
        pow_20: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_215, 2);  convert_element_type_215 = None
        mean_19: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_20, [-1], True);  pow_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_63: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_19, 1e-06);  mean_19 = None
        rsqrt_19: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
        mul_41: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_62, rsqrt_19);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_216: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_41, torch.bfloat16);  mul_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_42: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg80_1, convert_element_type_216);  arg80_1 = convert_element_type_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_234: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_42, [8192, 512])
        permute_108: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        mm_56: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_234, permute_108);  view_234 = permute_108 = None
        view_235: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_56, [8, 1024, 512]);  mm_56 = None
        view_236: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_235, [8, 1024, -1, 64]);  view_235 = None
        permute_109: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_43: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_109, [8, 8, 1024, 64]);  permute_109 = None
        clone_79: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_243: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_79, [64, 1024, 64]);  clone_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_237: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_42, [8192, 512])
        permute_110: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        mm_57: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_237, permute_110);  view_237 = permute_110 = None
        view_238: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_57, [8, 1024, 512]);  mm_57 = None
        view_239: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_238, [8, 1024, -1, 64]);  view_238 = None
        permute_111: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_239, [0, 2, 1, 3]);  view_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_114: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_111, [0, 1, 3, 2]);  permute_111 = None
        expand_44: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_114, [8, 8, 64, 1024]);  permute_114 = None
        clone_80: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_244: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_80, [64, 64, 1024]);  clone_80 = None
        bmm_20: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_243, view_244);  view_243 = view_244 = None
        view_245: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_20, [8, 8, 1024, 1024]);  bmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_64: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_245, add_46);  view_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_225: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_64, torch.float32);  add_64 = None
        amax_10: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_225, [-1], True)
        sub_12: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_225, amax_10);  convert_element_type_225 = amax_10 = None
        exp_10: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_11: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_14: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        convert_element_type_226: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_14, torch.bfloat16);  div_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_45: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_226, [8, 8, 1024, 1024]);  convert_element_type_226 = None
        view_248: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_45, [64, 1024, 1024]);  expand_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_240: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_42, [8192, 512]);  mul_42 = None
        permute_112: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        mm_58: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_240, permute_112);  view_240 = permute_112 = None
        view_241: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_58, [8, 1024, 512]);  mm_58 = None
        view_242: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_241, [8, 1024, -1, 64]);  view_241 = None
        permute_113: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_242, [0, 2, 1, 3]);  view_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_46: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_113, [8, 8, 1024, 64]);  permute_113 = None
        clone_82: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_249: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_82, [64, 1024, 64]);  clone_82 = None
        bmm_21: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_248, view_249);  view_248 = view_249 = None
        view_250: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_21, [8, 8, 1024, 64]);  bmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_115: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
        clone_83: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_251: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_83, [8, 1024, -1]);  clone_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_252: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_251, [8192, 512]);  view_251 = None
        permute_116: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
        mm_59: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_252, permute_116);  view_252 = permute_116 = None
        view_253: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_59, [8, 1024, 512]);  mm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_65: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_62, view_253);  add_62 = view_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_231: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_65, torch.float32)
        pow_21: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_231, 2);  convert_element_type_231 = None
        mean_20: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_21, [-1], True);  pow_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_66: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_20, 1e-06);  mean_20 = None
        rsqrt_20: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        mul_43: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_65, rsqrt_20);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_232: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_43, torch.bfloat16);  mul_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_44: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg85_1, convert_element_type_232);  arg85_1 = convert_element_type_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_254: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_44, [8192, 512]);  mul_44 = None
        permute_117: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        mm_60: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_254, permute_117);  view_254 = permute_117 = None
        view_255: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_60, [8, 1024, 512]);  mm_60 = None
        view_256: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_255, [8, 1024, -1, 64]);  view_255 = None
        permute_118: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_47: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_118, [8, 8, 1024, 64]);  permute_118 = None
        clone_85: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_263: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_85, [64, 1024, 64]);  clone_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_257: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_27, [8192, 512])
        permute_119: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        mm_61: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_257, permute_119);  view_257 = permute_119 = None
        view_258: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_61, [8, 1024, 512]);  mm_61 = None
        view_259: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_258, [8, 1024, -1, 64]);  view_258 = None
        permute_120: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_259, [0, 2, 1, 3]);  view_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_123: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_120, [0, 1, 3, 2]);  permute_120 = None
        expand_48: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_123, [8, 8, 64, 1024]);  permute_123 = None
        clone_86: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_264: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_86, [64, 64, 1024]);  clone_86 = None
        bmm_22: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_263, view_264);  view_263 = view_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_265: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_22, [8, 8, 1024, 1024]);  bmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_241: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_265, torch.float32);  view_265 = None
        amax_11: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_241, [-1], True)
        sub_13: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_241, amax_11);  convert_element_type_241 = amax_11 = None
        exp_11: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_12: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_15: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        convert_element_type_242: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_49: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_242, [8, 8, 1024, 1024]);  convert_element_type_242 = None
        view_268: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_49, [64, 1024, 1024]);  expand_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_260: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_27, [8192, 512])
        permute_121: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        mm_62: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_260, permute_121);  view_260 = permute_121 = None
        view_261: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_62, [8, 1024, 512]);  mm_62 = None
        view_262: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_261, [8, 1024, -1, 64]);  view_261 = None
        permute_122: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_262, [0, 2, 1, 3]);  view_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_50: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_122, [8, 8, 1024, 64]);  permute_122 = None
        clone_88: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_269: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_88, [64, 1024, 64]);  clone_88 = None
        bmm_23: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_268, view_269);  view_268 = view_269 = None
        view_270: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_23, [8, 8, 1024, 64]);  bmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_124: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_270, [0, 2, 1, 3]);  view_270 = None
        clone_89: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_271: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_89, [8, 1024, -1]);  clone_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_272: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_271, [8192, 512]);  view_271 = None
        permute_125: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        mm_63: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_272, permute_125);  view_272 = permute_125 = None
        view_273: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_63, [8, 1024, 512]);  mm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_68: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_65, view_273);  add_65 = view_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_247: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_68, torch.float32)
        pow_22: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_247, 2);  convert_element_type_247 = None
        mean_21: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_22, [-1], True);  pow_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_69: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_21, 1e-06);  mean_21 = None
        rsqrt_21: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        mul_45: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_68, rsqrt_21);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_248: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_45, torch.bfloat16);  mul_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_46: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg90_1, convert_element_type_248);  arg90_1 = convert_element_type_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_274: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_46, [8192, 512]);  mul_46 = None
        permute_126: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        mm_64: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_274, permute_126);  view_274 = permute_126 = None
        view_275: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_64, [8, 1024, 2048]);  mm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_8: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.relu.default(view_275);  view_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_276: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(relu_8, [8192, 2048]);  relu_8 = None
        permute_127: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        mm_65: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_276, permute_127);  view_276 = permute_127 = None
        view_277: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_65, [8, 1024, 512]);  mm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_70: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_68, view_277);  add_68 = view_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_253: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_70, torch.float32)
        pow_23: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_253, 2);  convert_element_type_253 = None
        mean_22: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_23, [-1], True);  pow_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_71: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_22, 1e-06);  mean_22 = None
        rsqrt_22: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        mul_47: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_70, rsqrt_22);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_254: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_47, torch.bfloat16);  mul_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_48: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg93_1, convert_element_type_254);  arg93_1 = convert_element_type_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_278: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_48, [8192, 512])
        permute_128: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        mm_66: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_278, permute_128);  view_278 = permute_128 = None
        view_279: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_66, [8, 1024, 512]);  mm_66 = None
        view_280: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_279, [8, 1024, -1, 64]);  view_279 = None
        permute_129: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_280, [0, 2, 1, 3]);  view_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_51: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_129, [8, 8, 1024, 64]);  permute_129 = None
        clone_93: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        view_287: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_93, [64, 1024, 64]);  clone_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_281: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_48, [8192, 512])
        permute_130: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        mm_67: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_281, permute_130);  view_281 = permute_130 = None
        view_282: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_67, [8, 1024, 512]);  mm_67 = None
        view_283: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_282, [8, 1024, -1, 64]);  view_282 = None
        permute_131: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_283, [0, 2, 1, 3]);  view_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_134: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_131, [0, 1, 3, 2]);  permute_131 = None
        expand_52: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_134, [8, 8, 64, 1024]);  permute_134 = None
        clone_94: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_288: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_94, [64, 64, 1024]);  clone_94 = None
        bmm_24: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_287, view_288);  view_287 = view_288 = None
        view_289: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_24, [8, 8, 1024, 1024]);  bmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_72: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_289, add_46);  view_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_263: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_72, torch.float32);  add_72 = None
        amax_12: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_263, [-1], True)
        sub_14: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_263, amax_12);  convert_element_type_263 = amax_12 = None
        exp_12: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
        sum_13: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_16: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        convert_element_type_264: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_16, torch.bfloat16);  div_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_53: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_264, [8, 8, 1024, 1024]);  convert_element_type_264 = None
        view_292: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_53, [64, 1024, 1024]);  expand_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_284: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_48, [8192, 512]);  mul_48 = None
        permute_132: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        mm_68: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_284, permute_132);  view_284 = permute_132 = None
        view_285: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_68, [8, 1024, 512]);  mm_68 = None
        view_286: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_285, [8, 1024, -1, 64]);  view_285 = None
        permute_133: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_286, [0, 2, 1, 3]);  view_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_54: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_133, [8, 8, 1024, 64]);  permute_133 = None
        clone_96: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_54, memory_format = torch.contiguous_format);  expand_54 = None
        view_293: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_96, [64, 1024, 64]);  clone_96 = None
        bmm_25: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_292, view_293);  view_292 = view_293 = None
        view_294: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_25, [8, 8, 1024, 64]);  bmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_135: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
        clone_97: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_295: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_97, [8, 1024, -1]);  clone_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_296: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_295, [8192, 512]);  view_295 = None
        permute_136: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        mm_69: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_296, permute_136);  view_296 = permute_136 = None
        view_297: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_69, [8, 1024, 512]);  mm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_73: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_70, view_297);  add_70 = view_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_269: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_73, torch.float32)
        pow_24: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_269, 2);  convert_element_type_269 = None
        mean_23: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_24, [-1], True);  pow_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_74: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_23, 1e-06);  mean_23 = None
        rsqrt_23: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        mul_49: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_73, rsqrt_23);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_270: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_49, torch.bfloat16);  mul_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_50: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg98_1, convert_element_type_270);  arg98_1 = convert_element_type_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_298: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_50, [8192, 512]);  mul_50 = None
        permute_137: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        mm_70: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_298, permute_137);  view_298 = permute_137 = None
        view_299: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_70, [8, 1024, 512]);  mm_70 = None
        view_300: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_299, [8, 1024, -1, 64]);  view_299 = None
        permute_138: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_300, [0, 2, 1, 3]);  view_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_55: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_138, [8, 8, 1024, 64]);  permute_138 = None
        clone_99: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
        view_307: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_99, [64, 1024, 64]);  clone_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_301: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_27, [8192, 512])
        permute_139: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        mm_71: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_301, permute_139);  view_301 = permute_139 = None
        view_302: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_71, [8, 1024, 512]);  mm_71 = None
        view_303: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_302, [8, 1024, -1, 64]);  view_302 = None
        permute_140: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_303, [0, 2, 1, 3]);  view_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_143: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_140, [0, 1, 3, 2]);  permute_140 = None
        expand_56: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_143, [8, 8, 64, 1024]);  permute_143 = None
        clone_100: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_308: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_100, [64, 64, 1024]);  clone_100 = None
        bmm_26: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_307, view_308);  view_307 = view_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_309: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_26, [8, 8, 1024, 1024]);  bmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_279: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_309, torch.float32);  view_309 = None
        amax_13: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_279, [-1], True)
        sub_15: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_279, amax_13);  convert_element_type_279 = amax_13 = None
        exp_13: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
        sum_14: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_17: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        convert_element_type_280: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_17, torch.bfloat16);  div_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_57: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_280, [8, 8, 1024, 1024]);  convert_element_type_280 = None
        view_312: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_57, [64, 1024, 1024]);  expand_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_304: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_27, [8192, 512])
        permute_141: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        mm_72: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_304, permute_141);  view_304 = permute_141 = None
        view_305: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_72, [8, 1024, 512]);  mm_72 = None
        view_306: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_305, [8, 1024, -1, 64]);  view_305 = None
        permute_142: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_306, [0, 2, 1, 3]);  view_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_58: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_142, [8, 8, 1024, 64]);  permute_142 = None
        clone_102: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
        view_313: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_102, [64, 1024, 64]);  clone_102 = None
        bmm_27: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_312, view_313);  view_312 = view_313 = None
        view_314: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_27, [8, 8, 1024, 64]);  bmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_144: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_314, [0, 2, 1, 3]);  view_314 = None
        clone_103: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_315: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_103, [8, 1024, -1]);  clone_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_316: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_315, [8192, 512]);  view_315 = None
        permute_145: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
        mm_73: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_316, permute_145);  view_316 = permute_145 = None
        view_317: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_73, [8, 1024, 512]);  mm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_76: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_73, view_317);  add_73 = view_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_285: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_76, torch.float32)
        pow_25: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_285, 2);  convert_element_type_285 = None
        mean_24: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_77: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_24, 1e-06);  mean_24 = None
        rsqrt_24: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        mul_51: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_76, rsqrt_24);  rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_286: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_51, torch.bfloat16);  mul_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_52: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg103_1, convert_element_type_286);  arg103_1 = convert_element_type_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_318: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_52, [8192, 512]);  mul_52 = None
        permute_146: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        mm_74: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_318, permute_146);  view_318 = permute_146 = None
        view_319: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_74, [8, 1024, 2048]);  mm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_9: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.relu.default(view_319);  view_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_320: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(relu_9, [8192, 2048]);  relu_9 = None
        permute_147: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        mm_75: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_320, permute_147);  view_320 = permute_147 = None
        view_321: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_75, [8, 1024, 512]);  mm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_78: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_76, view_321);  add_76 = view_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_291: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_78, torch.float32)
        pow_26: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_291, 2);  convert_element_type_291 = None
        mean_25: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_26, [-1], True);  pow_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_79: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_25, 1e-06);  mean_25 = None
        rsqrt_25: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
        mul_53: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_78, rsqrt_25);  rsqrt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_292: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_53, torch.bfloat16);  mul_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_54: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg106_1, convert_element_type_292);  arg106_1 = convert_element_type_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_322: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_54, [8192, 512])
        permute_148: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        mm_76: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_322, permute_148);  view_322 = permute_148 = None
        view_323: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_76, [8, 1024, 512]);  mm_76 = None
        view_324: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_323, [8, 1024, -1, 64]);  view_323 = None
        permute_149: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_324, [0, 2, 1, 3]);  view_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_59: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_149, [8, 8, 1024, 64]);  permute_149 = None
        clone_107: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
        view_331: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_107, [64, 1024, 64]);  clone_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_325: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_54, [8192, 512])
        permute_150: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        mm_77: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_325, permute_150);  view_325 = permute_150 = None
        view_326: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_77, [8, 1024, 512]);  mm_77 = None
        view_327: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_326, [8, 1024, -1, 64]);  view_326 = None
        permute_151: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_327, [0, 2, 1, 3]);  view_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_154: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_151, [0, 1, 3, 2]);  permute_151 = None
        expand_60: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_154, [8, 8, 64, 1024]);  permute_154 = None
        clone_108: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_332: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_108, [64, 64, 1024]);  clone_108 = None
        bmm_28: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_331, view_332);  view_331 = view_332 = None
        view_333: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_28, [8, 8, 1024, 1024]);  bmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_80: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_333, add_46);  view_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_301: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_80, torch.float32);  add_80 = None
        amax_14: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_301, [-1], True)
        sub_16: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_301, amax_14);  convert_element_type_301 = amax_14 = None
        exp_14: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_15: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_18: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        convert_element_type_302: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_18, torch.bfloat16);  div_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_61: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_302, [8, 8, 1024, 1024]);  convert_element_type_302 = None
        view_336: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_61, [64, 1024, 1024]);  expand_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_328: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_54, [8192, 512]);  mul_54 = None
        permute_152: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        mm_78: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_328, permute_152);  view_328 = permute_152 = None
        view_329: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_78, [8, 1024, 512]);  mm_78 = None
        view_330: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_329, [8, 1024, -1, 64]);  view_329 = None
        permute_153: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_62: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_153, [8, 8, 1024, 64]);  permute_153 = None
        clone_110: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_62, memory_format = torch.contiguous_format);  expand_62 = None
        view_337: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_110, [64, 1024, 64]);  clone_110 = None
        bmm_29: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_336, view_337);  view_336 = view_337 = None
        view_338: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_29, [8, 8, 1024, 64]);  bmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_155: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None
        clone_111: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_339: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_111, [8, 1024, -1]);  clone_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_340: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_339, [8192, 512]);  view_339 = None
        permute_156: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        mm_79: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_340, permute_156);  view_340 = permute_156 = None
        view_341: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_79, [8, 1024, 512]);  mm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_81: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_78, view_341);  add_78 = view_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_307: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_81, torch.float32)
        pow_27: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_307, 2);  convert_element_type_307 = None
        mean_26: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_27, [-1], True);  pow_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_82: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_26, 1e-06);  mean_26 = None
        rsqrt_26: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        mul_55: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_81, rsqrt_26);  rsqrt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_308: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_55, torch.bfloat16);  mul_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_56: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg111_1, convert_element_type_308);  arg111_1 = convert_element_type_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_342: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_56, [8192, 512]);  mul_56 = None
        permute_157: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
        mm_80: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_342, permute_157);  view_342 = permute_157 = None
        view_343: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_80, [8, 1024, 512]);  mm_80 = None
        view_344: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_343, [8, 1024, -1, 64]);  view_343 = None
        permute_158: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_63: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_158, [8, 8, 1024, 64]);  permute_158 = None
        clone_113: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
        view_351: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_113, [64, 1024, 64]);  clone_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_345: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_27, [8192, 512])
        permute_159: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        mm_81: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_345, permute_159);  view_345 = permute_159 = None
        view_346: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_81, [8, 1024, 512]);  mm_81 = None
        view_347: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_346, [8, 1024, -1, 64]);  view_346 = None
        permute_160: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_347, [0, 2, 1, 3]);  view_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_163: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_160, [0, 1, 3, 2]);  permute_160 = None
        expand_64: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_163, [8, 8, 64, 1024]);  permute_163 = None
        clone_114: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
        view_352: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_114, [64, 64, 1024]);  clone_114 = None
        bmm_30: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_351, view_352);  view_351 = view_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_353: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_30, [8, 8, 1024, 1024]);  bmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_317: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_353, torch.float32);  view_353 = None
        amax_15: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_317, [-1], True)
        sub_17: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_317, amax_15);  convert_element_type_317 = amax_15 = None
        exp_15: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
        sum_16: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_19: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        convert_element_type_318: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16);  div_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_65: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_318, [8, 8, 1024, 1024]);  convert_element_type_318 = None
        view_356: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_65, [64, 1024, 1024]);  expand_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_348: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_27, [8192, 512])
        permute_161: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        mm_82: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_348, permute_161);  view_348 = permute_161 = None
        view_349: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_82, [8, 1024, 512]);  mm_82 = None
        view_350: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_349, [8, 1024, -1, 64]);  view_349 = None
        permute_162: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_350, [0, 2, 1, 3]);  view_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_66: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_162, [8, 8, 1024, 64]);  permute_162 = None
        clone_116: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
        view_357: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_116, [64, 1024, 64]);  clone_116 = None
        bmm_31: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_356, view_357);  view_356 = view_357 = None
        view_358: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_31, [8, 8, 1024, 64]);  bmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_164: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_358, [0, 2, 1, 3]);  view_358 = None
        clone_117: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_164, memory_format = torch.contiguous_format);  permute_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_359: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_117, [8, 1024, -1]);  clone_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_360: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_359, [8192, 512]);  view_359 = None
        permute_165: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        mm_83: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_360, permute_165);  view_360 = permute_165 = None
        view_361: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_83, [8, 1024, 512]);  mm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_84: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_81, view_361);  add_81 = view_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_323: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_84, torch.float32)
        pow_28: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_323, 2);  convert_element_type_323 = None
        mean_27: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_28, [-1], True);  pow_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_85: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_27, 1e-06);  mean_27 = None
        rsqrt_27: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        mul_57: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_84, rsqrt_27);  rsqrt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_324: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_57, torch.bfloat16);  mul_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_58: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg116_1, convert_element_type_324);  arg116_1 = convert_element_type_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_362: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_58, [8192, 512]);  mul_58 = None
        permute_166: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        mm_84: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_362, permute_166);  view_362 = permute_166 = None
        view_363: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_84, [8, 1024, 2048]);  mm_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_10: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.relu.default(view_363);  view_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_364: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(relu_10, [8192, 2048]);  relu_10 = None
        permute_167: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        mm_85: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_364, permute_167);  view_364 = permute_167 = None
        view_365: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_85, [8, 1024, 512]);  mm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_86: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_84, view_365);  add_84 = view_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_329: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_86, torch.float32)
        pow_29: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_329, 2);  convert_element_type_329 = None
        mean_28: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_29, [-1], True);  pow_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_87: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_28, 1e-06);  mean_28 = None
        rsqrt_28: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
        mul_59: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_86, rsqrt_28);  rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_330: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_59, torch.bfloat16);  mul_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_60: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg119_1, convert_element_type_330);  arg119_1 = convert_element_type_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_366: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_60, [8192, 512])
        permute_168: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
        mm_86: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_366, permute_168);  view_366 = permute_168 = None
        view_367: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_86, [8, 1024, 512]);  mm_86 = None
        view_368: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_367, [8, 1024, -1, 64]);  view_367 = None
        permute_169: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_368, [0, 2, 1, 3]);  view_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_67: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_169, [8, 8, 1024, 64]);  permute_169 = None
        clone_121: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
        view_375: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_121, [64, 1024, 64]);  clone_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_369: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_60, [8192, 512])
        permute_170: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        mm_87: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_369, permute_170);  view_369 = permute_170 = None
        view_370: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_87, [8, 1024, 512]);  mm_87 = None
        view_371: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_370, [8, 1024, -1, 64]);  view_370 = None
        permute_171: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_371, [0, 2, 1, 3]);  view_371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_174: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_171, [0, 1, 3, 2]);  permute_171 = None
        expand_68: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_174, [8, 8, 64, 1024]);  permute_174 = None
        clone_122: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
        view_376: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_122, [64, 64, 1024]);  clone_122 = None
        bmm_32: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_375, view_376);  view_375 = view_376 = None
        view_377: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_32, [8, 8, 1024, 1024]);  bmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_88: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_377, add_46);  view_377 = add_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_339: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_88, torch.float32);  add_88 = None
        amax_16: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_339, [-1], True)
        sub_18: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_339, amax_16);  convert_element_type_339 = amax_16 = None
        exp_16: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
        sum_17: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_20: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        convert_element_type_340: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_20, torch.bfloat16);  div_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_69: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_340, [8, 8, 1024, 1024]);  convert_element_type_340 = None
        view_380: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_69, [64, 1024, 1024]);  expand_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_372: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_60, [8192, 512]);  mul_60 = None
        permute_172: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        mm_88: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_372, permute_172);  view_372 = permute_172 = None
        view_373: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_88, [8, 1024, 512]);  mm_88 = None
        view_374: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_373, [8, 1024, -1, 64]);  view_373 = None
        permute_173: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_374, [0, 2, 1, 3]);  view_374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_70: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_173, [8, 8, 1024, 64]);  permute_173 = None
        clone_124: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_70, memory_format = torch.contiguous_format);  expand_70 = None
        view_381: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_124, [64, 1024, 64]);  clone_124 = None
        bmm_33: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_380, view_381);  view_380 = view_381 = None
        view_382: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_33, [8, 8, 1024, 64]);  bmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_175: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_382, [0, 2, 1, 3]);  view_382 = None
        clone_125: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_175, memory_format = torch.contiguous_format);  permute_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_383: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_125, [8, 1024, -1]);  clone_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_384: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_383, [8192, 512]);  view_383 = None
        permute_176: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        mm_89: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_384, permute_176);  view_384 = permute_176 = None
        view_385: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_89, [8, 1024, 512]);  mm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_89: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_86, view_385);  add_86 = view_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_345: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_89, torch.float32)
        pow_30: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_345, 2);  convert_element_type_345 = None
        mean_29: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_30, [-1], True);  pow_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_90: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_29, 1e-06);  mean_29 = None
        rsqrt_29: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        mul_61: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_89, rsqrt_29);  rsqrt_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_346: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_61, torch.bfloat16);  mul_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_62: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg124_1, convert_element_type_346);  arg124_1 = convert_element_type_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_386: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_62, [8192, 512]);  mul_62 = None
        permute_177: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        mm_90: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_386, permute_177);  view_386 = permute_177 = None
        view_387: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_90, [8, 1024, 512]);  mm_90 = None
        view_388: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_387, [8, 1024, -1, 64]);  view_387 = None
        permute_178: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_388, [0, 2, 1, 3]);  view_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_71: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_178, [8, 8, 1024, 64]);  permute_178 = None
        clone_127: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
        view_395: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_127, [64, 1024, 64]);  clone_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_389: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_27, [8192, 512])
        permute_179: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        mm_91: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_389, permute_179);  view_389 = permute_179 = None
        view_390: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_91, [8, 1024, 512]);  mm_91 = None
        view_391: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_390, [8, 1024, -1, 64]);  view_390 = None
        permute_180: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_391, [0, 2, 1, 3]);  view_391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_183: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.permute.default(permute_180, [0, 1, 3, 2]);  permute_180 = None
        expand_72: "bf16[8, 8, 64, 1024][524288, 64, 1, 512]cuda:0" = torch.ops.aten.expand.default(permute_183, [8, 8, 64, 1024]);  permute_183 = None
        clone_128: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
        view_396: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_128, [64, 64, 1024]);  clone_128 = None
        bmm_34: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_395, view_396);  view_395 = view_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_397: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_34, [8, 8, 1024, 1024]);  bmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_355: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_397, torch.float32);  view_397 = None
        amax_17: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_355, [-1], True)
        sub_19: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_355, amax_17);  convert_element_type_355 = amax_17 = None
        exp_17: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_18: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_21: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
        convert_element_type_356: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16);  div_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_73: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_356, [8, 8, 1024, 1024]);  convert_element_type_356 = None
        view_400: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(expand_73, [64, 1024, 1024]);  expand_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_392: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_27, [8192, 512])
        permute_181: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        mm_92: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_392, permute_181);  view_392 = permute_181 = None
        view_393: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_92, [8, 1024, 512]);  mm_92 = None
        view_394: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_393, [8, 1024, -1, 64]);  view_393 = None
        permute_182: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_394, [0, 2, 1, 3]);  view_394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_74: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.expand.default(permute_182, [8, 8, 1024, 64]);  permute_182 = None
        clone_130: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
        view_401: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_130, [64, 1024, 64]);  clone_130 = None
        bmm_35: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_400, view_401);  view_400 = view_401 = None
        view_402: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_35, [8, 8, 1024, 64]);  bmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_184: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
        clone_131: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_184, memory_format = torch.contiguous_format);  permute_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_403: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_131, [8, 1024, -1]);  clone_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_404: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_403, [8192, 512]);  view_403 = None
        permute_185: "bf16[512, 512][1, 512]cuda:0" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
        mm_93: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_404, permute_185);  view_404 = permute_185 = None
        view_405: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_93, [8, 1024, 512]);  mm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_92: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_89, view_405);  add_89 = view_405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_361: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_92, torch.float32)
        pow_31: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_361, 2);  convert_element_type_361 = None
        mean_30: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_31, [-1], True);  pow_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_93: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_30, 1e-06);  mean_30 = None
        rsqrt_30: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        mul_63: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_92, rsqrt_30);  rsqrt_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_362: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_63, torch.bfloat16);  mul_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_64: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg129_1, convert_element_type_362);  arg129_1 = convert_element_type_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_406: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_64, [8192, 512]);  mul_64 = None
        permute_186: "bf16[512, 2048][1, 512]cuda:0" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        mm_94: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_406, permute_186);  view_406 = permute_186 = None
        view_407: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_94, [8, 1024, 2048]);  mm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_11: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.relu.default(view_407);  view_407 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_408: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(relu_11, [8192, 2048]);  relu_11 = None
        permute_187: "bf16[2048, 512][1, 2048]cuda:0" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        mm_95: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_408, permute_187);  view_408 = permute_187 = None
        view_409: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_95, [8, 1024, 512]);  mm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_94: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_92, view_409);  add_92 = view_409 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_367: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_94, torch.float32)
        pow_32: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_367, 2);  convert_element_type_367 = None
        mean_31: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_32, [-1], True);  pow_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_95: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_31, 1e-06);  mean_31 = None
        rsqrt_31: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
        mul_65: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_94, rsqrt_31);  add_94 = rsqrt_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_368: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_65, torch.bfloat16);  mul_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_66: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg132_1, convert_element_type_368);  arg132_1 = convert_element_type_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1095 in forward, code: sequence_output = sequence_output * (self.model_dim**-0.5)
        mul_67: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, 0.04419417382415922);  mul_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1097 in forward, code: lm_logits = self.lm_head(sequence_output)
        view_410: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_67, [8192, 512]);  mul_67 = None
        permute_188: "bf16[512, 32128][1, 512]cuda:0" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        mm_96: "bf16[8192, 32128][32128, 1]cuda:0" = torch.ops.aten.mm.default(view_410, permute_188);  view_410 = permute_188 = None
        view_411: "bf16[8, 1024, 32128][32899072, 32128, 1]cuda:0" = torch.ops.aten.reshape.default(mm_96, [8, 1024, 32128]);  mm_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1104 in forward, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        view_412: "bf16[8192, 32128][32128, 1]cuda:0" = torch.ops.aten.reshape.default(view_411, [-1, 32128])
        convert_element_type_371: "f32[8192, 32128][32128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_412, torch.float32);  view_412 = None
        amax_18: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_371, [1], True)
        sub_20: "f32[8192, 32128][32128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_371, amax_18);  convert_element_type_371 = amax_18 = None
        exp_18: "f32[8192, 32128][32128, 1]cuda:0" = torch.ops.aten.exp.default(sub_20)
        sum_19: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_18, [1], True);  exp_18 = None
        log_2: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.log.default(sum_19);  sum_19 = None
        sub_21: "f32[8192, 32128][32128, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_20, log_2);  sub_20 = log_2 = None
        convert_element_type_372: "bf16[8192, 32128][32128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_21, torch.bfloat16);  sub_21 = None
        ne: "b8[8192][1]cuda:0" = torch.ops.aten.ne.Scalar(view_413, -100)
        full_default_4: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5: "i64[8192][1]cuda:0" = torch.ops.aten.where.self(ne, view_413, full_default_4);  ne = full_default_4 = None
        unsqueeze_18: "i64[8192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(where_5, 1);  where_5 = None
        gather: "bf16[8192, 1][1, 1]cuda:0" = torch.ops.aten.gather.default(convert_element_type_372, 1, unsqueeze_18);  convert_element_type_372 = unsqueeze_18 = None
        squeeze: "bf16[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg_1: "bf16[8192][1]cuda:0" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_5: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "bf16[8192][1]cuda:0" = torch.ops.aten.where.self(ne_1, neg_1, full_default_5);  ne_1 = neg_1 = full_default_5 = None
        sum_21: "bf16[][]cuda:0" = torch.ops.aten.sum.default(where_6);  where_6 = None
        ne_2: "b8[8192][1]cuda:0" = torch.ops.aten.ne.Scalar(view_413, -100);  view_413 = None
        sum_20: "i64[][]cuda:0" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_373: "bf16[][]cuda:0" = torch.ops.prims.convert_element_type.default(sum_20, torch.bfloat16);  sum_20 = None
        div_22: "bf16[][]cuda:0" = torch.ops.aten.div.Tensor(sum_21, convert_element_type_373);  sum_21 = convert_element_type_373 = None
        return (div_22, view_411, mul_27)
