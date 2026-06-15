class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[32, 128][128, 1]cuda:0", arg1_1: "bf16[250112, 512][512, 1]cuda:0", arg2_1: "bf16[512][1]cuda:0", arg3_1: "bf16[384, 512][512, 1]cuda:0", arg4_1: "bf16[384, 512][512, 1]cuda:0", arg5_1: "bf16[384, 512][512, 1]cuda:0", arg6_1: "bf16[32, 6][6, 1]cuda:0", arg7_1: "bf16[512, 384][384, 1]cuda:0", arg8_1: "bf16[512][1]cuda:0", arg9_1: "bf16[1024, 512][512, 1]cuda:0", arg10_1: "bf16[1024, 512][512, 1]cuda:0", arg11_1: "bf16[512, 1024][1024, 1]cuda:0", arg12_1: "bf16[512][1]cuda:0", arg13_1: "bf16[384, 512][512, 1]cuda:0", arg14_1: "bf16[384, 512][512, 1]cuda:0", arg15_1: "bf16[384, 512][512, 1]cuda:0", arg16_1: "bf16[512, 384][384, 1]cuda:0", arg17_1: "bf16[512][1]cuda:0", arg18_1: "bf16[1024, 512][512, 1]cuda:0", arg19_1: "bf16[1024, 512][512, 1]cuda:0", arg20_1: "bf16[512, 1024][1024, 1]cuda:0", arg21_1: "bf16[512][1]cuda:0", arg22_1: "bf16[384, 512][512, 1]cuda:0", arg23_1: "bf16[384, 512][512, 1]cuda:0", arg24_1: "bf16[384, 512][512, 1]cuda:0", arg25_1: "bf16[512, 384][384, 1]cuda:0", arg26_1: "bf16[512][1]cuda:0", arg27_1: "bf16[1024, 512][512, 1]cuda:0", arg28_1: "bf16[1024, 512][512, 1]cuda:0", arg29_1: "bf16[512, 1024][1024, 1]cuda:0", arg30_1: "bf16[512][1]cuda:0", arg31_1: "bf16[384, 512][512, 1]cuda:0", arg32_1: "bf16[384, 512][512, 1]cuda:0", arg33_1: "bf16[384, 512][512, 1]cuda:0", arg34_1: "bf16[512, 384][384, 1]cuda:0", arg35_1: "bf16[512][1]cuda:0", arg36_1: "bf16[1024, 512][512, 1]cuda:0", arg37_1: "bf16[1024, 512][512, 1]cuda:0", arg38_1: "bf16[512, 1024][1024, 1]cuda:0", arg39_1: "bf16[512][1]cuda:0", arg40_1: "bf16[384, 512][512, 1]cuda:0", arg41_1: "bf16[384, 512][512, 1]cuda:0", arg42_1: "bf16[384, 512][512, 1]cuda:0", arg43_1: "bf16[512, 384][384, 1]cuda:0", arg44_1: "bf16[512][1]cuda:0", arg45_1: "bf16[1024, 512][512, 1]cuda:0", arg46_1: "bf16[1024, 512][512, 1]cuda:0", arg47_1: "bf16[512, 1024][1024, 1]cuda:0", arg48_1: "bf16[512][1]cuda:0", arg49_1: "bf16[384, 512][512, 1]cuda:0", arg50_1: "bf16[384, 512][512, 1]cuda:0", arg51_1: "bf16[384, 512][512, 1]cuda:0", arg52_1: "bf16[512, 384][384, 1]cuda:0", arg53_1: "bf16[512][1]cuda:0", arg54_1: "bf16[1024, 512][512, 1]cuda:0", arg55_1: "bf16[1024, 512][512, 1]cuda:0", arg56_1: "bf16[512, 1024][1024, 1]cuda:0", arg57_1: "bf16[512][1]cuda:0", arg58_1: "bf16[384, 512][512, 1]cuda:0", arg59_1: "bf16[384, 512][512, 1]cuda:0", arg60_1: "bf16[384, 512][512, 1]cuda:0", arg61_1: "bf16[512, 384][384, 1]cuda:0", arg62_1: "bf16[512][1]cuda:0", arg63_1: "bf16[1024, 512][512, 1]cuda:0", arg64_1: "bf16[1024, 512][512, 1]cuda:0", arg65_1: "bf16[512, 1024][1024, 1]cuda:0", arg66_1: "bf16[512][1]cuda:0", arg67_1: "bf16[384, 512][512, 1]cuda:0", arg68_1: "bf16[384, 512][512, 1]cuda:0", arg69_1: "bf16[384, 512][512, 1]cuda:0", arg70_1: "bf16[512, 384][384, 1]cuda:0", arg71_1: "bf16[512][1]cuda:0", arg72_1: "bf16[1024, 512][512, 1]cuda:0", arg73_1: "bf16[1024, 512][512, 1]cuda:0", arg74_1: "bf16[512, 1024][1024, 1]cuda:0", arg75_1: "bf16[512][1]cuda:0", arg76_1: "i64[32, 128][128, 1]cuda:0", arg77_1: "bf16[512][1]cuda:0", arg78_1: "bf16[384, 512][512, 1]cuda:0", arg79_1: "bf16[384, 512][512, 1]cuda:0", arg80_1: "bf16[384, 512][512, 1]cuda:0", arg81_1: "bf16[32, 6][6, 1]cuda:0", arg82_1: "bf16[512, 384][384, 1]cuda:0", arg83_1: "bf16[512][1]cuda:0", arg84_1: "bf16[384, 512][512, 1]cuda:0", arg85_1: "bf16[384, 512][512, 1]cuda:0", arg86_1: "bf16[384, 512][512, 1]cuda:0", arg87_1: "bf16[512, 384][384, 1]cuda:0", arg88_1: "bf16[512][1]cuda:0", arg89_1: "bf16[1024, 512][512, 1]cuda:0", arg90_1: "bf16[1024, 512][512, 1]cuda:0", arg91_1: "bf16[512, 1024][1024, 1]cuda:0", arg92_1: "bf16[512][1]cuda:0", arg93_1: "bf16[384, 512][512, 1]cuda:0", arg94_1: "bf16[384, 512][512, 1]cuda:0", arg95_1: "bf16[384, 512][512, 1]cuda:0", arg96_1: "bf16[512, 384][384, 1]cuda:0", arg97_1: "bf16[512][1]cuda:0", arg98_1: "bf16[384, 512][512, 1]cuda:0", arg99_1: "bf16[384, 512][512, 1]cuda:0", arg100_1: "bf16[384, 512][512, 1]cuda:0", arg101_1: "bf16[512, 384][384, 1]cuda:0", arg102_1: "bf16[512][1]cuda:0", arg103_1: "bf16[1024, 512][512, 1]cuda:0", arg104_1: "bf16[1024, 512][512, 1]cuda:0", arg105_1: "bf16[512, 1024][1024, 1]cuda:0", arg106_1: "bf16[512][1]cuda:0", arg107_1: "bf16[384, 512][512, 1]cuda:0", arg108_1: "bf16[384, 512][512, 1]cuda:0", arg109_1: "bf16[384, 512][512, 1]cuda:0", arg110_1: "bf16[512, 384][384, 1]cuda:0", arg111_1: "bf16[512][1]cuda:0", arg112_1: "bf16[384, 512][512, 1]cuda:0", arg113_1: "bf16[384, 512][512, 1]cuda:0", arg114_1: "bf16[384, 512][512, 1]cuda:0", arg115_1: "bf16[512, 384][384, 1]cuda:0", arg116_1: "bf16[512][1]cuda:0", arg117_1: "bf16[1024, 512][512, 1]cuda:0", arg118_1: "bf16[1024, 512][512, 1]cuda:0", arg119_1: "bf16[512, 1024][1024, 1]cuda:0", arg120_1: "bf16[512][1]cuda:0", arg121_1: "bf16[384, 512][512, 1]cuda:0", arg122_1: "bf16[384, 512][512, 1]cuda:0", arg123_1: "bf16[384, 512][512, 1]cuda:0", arg124_1: "bf16[512, 384][384, 1]cuda:0", arg125_1: "bf16[512][1]cuda:0", arg126_1: "bf16[384, 512][512, 1]cuda:0", arg127_1: "bf16[384, 512][512, 1]cuda:0", arg128_1: "bf16[384, 512][512, 1]cuda:0", arg129_1: "bf16[512, 384][384, 1]cuda:0", arg130_1: "bf16[512][1]cuda:0", arg131_1: "bf16[1024, 512][512, 1]cuda:0", arg132_1: "bf16[1024, 512][512, 1]cuda:0", arg133_1: "bf16[512, 1024][1024, 1]cuda:0", arg134_1: "bf16[512][1]cuda:0", arg135_1: "bf16[384, 512][512, 1]cuda:0", arg136_1: "bf16[384, 512][512, 1]cuda:0", arg137_1: "bf16[384, 512][512, 1]cuda:0", arg138_1: "bf16[512, 384][384, 1]cuda:0", arg139_1: "bf16[512][1]cuda:0", arg140_1: "bf16[384, 512][512, 1]cuda:0", arg141_1: "bf16[384, 512][512, 1]cuda:0", arg142_1: "bf16[384, 512][512, 1]cuda:0", arg143_1: "bf16[512, 384][384, 1]cuda:0", arg144_1: "bf16[512][1]cuda:0", arg145_1: "bf16[1024, 512][512, 1]cuda:0", arg146_1: "bf16[1024, 512][512, 1]cuda:0", arg147_1: "bf16[512, 1024][1024, 1]cuda:0", arg148_1: "bf16[512][1]cuda:0", arg149_1: "bf16[384, 512][512, 1]cuda:0", arg150_1: "bf16[384, 512][512, 1]cuda:0", arg151_1: "bf16[384, 512][512, 1]cuda:0", arg152_1: "bf16[512, 384][384, 1]cuda:0", arg153_1: "bf16[512][1]cuda:0", arg154_1: "bf16[384, 512][512, 1]cuda:0", arg155_1: "bf16[384, 512][512, 1]cuda:0", arg156_1: "bf16[384, 512][512, 1]cuda:0", arg157_1: "bf16[512, 384][384, 1]cuda:0", arg158_1: "bf16[512][1]cuda:0", arg159_1: "bf16[1024, 512][512, 1]cuda:0", arg160_1: "bf16[1024, 512][512, 1]cuda:0", arg161_1: "bf16[512, 1024][1024, 1]cuda:0", arg162_1: "bf16[512][1]cuda:0", arg163_1: "bf16[384, 512][512, 1]cuda:0", arg164_1: "bf16[384, 512][512, 1]cuda:0", arg165_1: "bf16[384, 512][512, 1]cuda:0", arg166_1: "bf16[512, 384][384, 1]cuda:0", arg167_1: "bf16[512][1]cuda:0", arg168_1: "bf16[384, 512][512, 1]cuda:0", arg169_1: "bf16[384, 512][512, 1]cuda:0", arg170_1: "bf16[384, 512][512, 1]cuda:0", arg171_1: "bf16[512, 384][384, 1]cuda:0", arg172_1: "bf16[512][1]cuda:0", arg173_1: "bf16[1024, 512][512, 1]cuda:0", arg174_1: "bf16[1024, 512][512, 1]cuda:0", arg175_1: "bf16[512, 1024][1024, 1]cuda:0", arg176_1: "bf16[512][1]cuda:0", arg177_1: "bf16[384, 512][512, 1]cuda:0", arg178_1: "bf16[384, 512][512, 1]cuda:0", arg179_1: "bf16[384, 512][512, 1]cuda:0", arg180_1: "bf16[512, 384][384, 1]cuda:0", arg181_1: "bf16[512][1]cuda:0", arg182_1: "bf16[384, 512][512, 1]cuda:0", arg183_1: "bf16[384, 512][512, 1]cuda:0", arg184_1: "bf16[384, 512][512, 1]cuda:0", arg185_1: "bf16[512, 384][384, 1]cuda:0", arg186_1: "bf16[512][1]cuda:0", arg187_1: "bf16[1024, 512][512, 1]cuda:0", arg188_1: "bf16[1024, 512][512, 1]cuda:0", arg189_1: "bf16[512, 1024][1024, 1]cuda:0", arg190_1: "bf16[512][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_2: "i64[128][1]cuda:0" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[128][1]cuda:0" = torch.ops.aten.add.Tensor(iota_2, 0);  iota_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze: "i64[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add, 0);  add = None
        unsqueeze_1: "i64[1, 1, 128][128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "i64[1, 1, 128, 1][128, 128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:87 in bidirectional_mask_function, code: return q_idx >= 0
        ge: "b8[1, 1, 128, 1][128, 128, 1, 1]cuda:0" = torch.ops.aten.ge.Scalar(unsqueeze_2, 0);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand: "b8[32, 1, 128, 128][0, 128, 1, 0]cuda:0" = torch.ops.aten.expand.default(ge, [32, -1, 128, 128]);  ge = expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        _tensor_constant0: "bf16[][]cuda:0" = self._tensor_constant0
        lift_fresh_copy: "bf16[][]cuda:0" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = lift_fresh_copy = None
        scalar_tensor: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  scalar_tensor = None
        _tensor_constant1: "bf16[][]cuda:0" = self._tensor_constant1;  _tensor_constant1 = None
        _tensor_constant2: "bf16[][]cuda:0" = self._tensor_constant2

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_12: "i64[128][1]cuda:0" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_66: "i64[128][1]cuda:0" = torch.ops.aten.add.Tensor(iota_12, 0);  iota_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_12: "i64[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_66, 0);  add_66 = None
        unsqueeze_13: "i64[1, 1, 128][128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_12, 1);  unsqueeze_12 = None
        unsqueeze_14: "i64[1, 1, 128, 1][128, 128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 3);  unsqueeze_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:87 in bidirectional_mask_function, code: return q_idx >= 0
        ge_1: "b8[1, 1, 128, 1][128, 128, 1, 1]cuda:0" = torch.ops.aten.ge.Scalar(unsqueeze_14, 0);  unsqueeze_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_34: "b8[32, 1, 128, 128][0, 128, 1, 0]cuda:0" = torch.ops.aten.expand.default(ge_1, [32, -1, 128, 128]);  ge_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        lift_fresh_copy_2: "bf16[][]cuda:0" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant2);  _tensor_constant2 = None
        scalar_tensor_2: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_3: "bf16[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_34, lift_fresh_copy_2, scalar_tensor_2);  expand_34 = lift_fresh_copy_2 = scalar_tensor_2 = where_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:305 in forward, code: position_bias = torch.zeros(
        full_3: "bf16[1, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([1, 6, 128, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:317 in forward, code: position_bias = position_bias + causal_mask
        full_default_3: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([32, 6, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:1150 in forward, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        view_581: "i64[4096][1]cuda:0" = torch.ops.aten.reshape.default(arg76_1, [-1]);  arg76_1 = None
        ne_1: "b8[4096][1]cuda:0" = torch.ops.aten.ne.Scalar(view_581, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:680 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding_2: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.embedding.default(arg1_1, arg0_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_197: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(embedding_2, torch.float32)
        pow_26: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_197, 2);  convert_element_type_197 = None
        mean_17: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_26, [-1], True);  pow_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_68: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_17, 1e-06);  mean_17 = None
        rsqrt_17: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        mul_76: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(embedding_2, rsqrt_17);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_198: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_76, torch.bfloat16);  mul_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_77: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg77_1, convert_element_type_198);  arg77_1 = convert_element_type_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_210: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_77, [4096, 512])
        permute_97: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        mm_56: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_210, permute_97);  view_210 = permute_97 = None
        view_211: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_56, [32, 128, 384]);  mm_56 = None
        view_212: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_211, [32, 128, -1, 64]);  view_211 = None
        permute_98: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_35: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_98, [32, 6, 128, 64]);  permute_98 = None
        clone_67: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_219: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_67, [192, 128, 64]);  clone_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_213: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_77, [4096, 512])
        permute_99: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        mm_57: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_213, permute_99);  view_213 = permute_99 = None
        view_214: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_57, [32, 128, 384]);  mm_57 = None
        view_215: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_214, [32, 128, -1, 64]);  view_214 = None
        permute_100: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_215, [0, 2, 1, 3]);  view_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_103: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_100, [0, 1, 3, 2]);  permute_100 = None
        expand_36: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_103, [32, 6, 64, 128]);  permute_103 = None
        clone_68: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_220: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_68, [192, 64, 128]);  clone_68 = None
        bmm_16: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_219, view_220);  view_219 = view_220 = None
        view_221: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_16, [32, 6, 128, 128]);  bmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:233 in compute_bias, code: memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        iota_15: "i64[128][1]cuda:0" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_16: "i64[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_15, 0);  iota_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:232 in compute_bias, code: context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None] + past_seen_tokens
        iota_14: "i64[128][1]cuda:0" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_15: "i64[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_14, 1);  iota_14 = None
        add_69: "i64[128, 1][1, 1]cuda:0" = torch.ops.aten.add.Tensor(unsqueeze_15, 0);  unsqueeze_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:234 in compute_bias, code: relative_position = memory_position - context_position  # shape (query_length, key_length)
        sub_9: "i64[128, 128][128, 1]cuda:0" = torch.ops.aten.sub.Tensor(unsqueeze_16, add_69);  unsqueeze_16 = add_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:208 in _relative_position_bucket, code: relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        full_1: "i64[128, 128][128, 1]cuda:0" = torch.ops.aten.full.default([128, 128], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_1: "i64[128, 128][128, 1]cuda:0" = torch.ops.aten.minimum.default(sub_9, full_1);  sub_9 = full_1 = None
        neg: "i64[128, 128][128, 1]cuda:0" = torch.ops.aten.neg.default(minimum_1);  minimum_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:213 in _relative_position_bucket, code: is_small = relative_position < max_exact
        lt_1: "b8[128, 128][128, 1]cuda:0" = torch.ops.aten.lt.Scalar(neg, 16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:217 in _relative_position_bucket, code: torch.log(relative_position.float() / max_exact)
        convert_element_type_207: "f32[128, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(neg, torch.float32)
        div_10: "f32[128, 128][128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_207, 16);  convert_element_type_207 = None
        log_1: "f32[128, 128][128, 1]cuda:0" = torch.ops.aten.log.default(div_10);  div_10 = None
        div_11: "f32[128, 128][128, 1]cuda:0" = torch.ops.aten.div.Tensor(log_1, 2.0794415416798357);  log_1 = None
        mul_78: "f32[128, 128][128, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_11, 16);  div_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:220 in _relative_position_bucket, code: ).to(torch.long)
        convert_element_type_208: "i64[128, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_78, torch.int64);  mul_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:216 in _relative_position_bucket, code: relative_position_if_large = max_exact + (
        add_70: "i64[128, 128][128, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_208, 16);  convert_element_type_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:222 in _relative_position_bucket, code: relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        full_2: "i64[128, 128][128, 1]cuda:0" = torch.ops.aten.full.default([128, 128], 31, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:221 in _relative_position_bucket, code: relative_position_if_large = torch.min(
        minimum_2: "i64[128, 128][128, 1]cuda:0" = torch.ops.aten.minimum.default(add_70, full_2);  add_70 = full_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:225 in _relative_position_bucket, code: relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        where_4: "i64[128, 128][128, 1]cuda:0" = torch.ops.aten.where.self(lt_1, neg, minimum_2);  lt_1 = neg = minimum_2 = None
        add_71: "i64[128, 128][128, 1]cuda:0" = torch.ops.aten.add.Tensor(where_4, 0);  where_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:241 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        embedding_3: "bf16[128, 128, 6][768, 6, 1]cuda:0" = torch.ops.aten.embedding.default(arg81_1, add_71);  arg81_1 = add_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:242 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        permute_104: "bf16[6, 128, 128][1, 768, 6]cuda:0" = torch.ops.aten.permute.default(embedding_3, [2, 0, 1]);  embedding_3 = None
        unsqueeze_17: "bf16[1, 6, 128, 128][6, 1, 768, 6]cuda:0" = torch.ops.aten.unsqueeze.default(permute_104, 0);  permute_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:512 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        iota_9: "i64[128][1]cuda:0" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_65: "i64[128][1]cuda:0" = torch.ops.aten.add.Tensor(iota_9, 0);  iota_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_9: "i64[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_65, 0);  add_65 = None
        unsqueeze_10: "i64[1, 1, 128][128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 1);  unsqueeze_9 = None
        unsqueeze_11: "i64[1, 1, 1, 128][128, 128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 2);  unsqueeze_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_8: "i64[128][1]cuda:0" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_64: "i64[128][1]cuda:0" = torch.ops.aten.add.Tensor(iota_8, 0);  iota_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_6: "i64[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_64, 0);  add_64 = None
        unsqueeze_7: "i64[1, 1, 128][128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 1);  unsqueeze_6 = None
        unsqueeze_8: "i64[1, 1, 128, 1][128, 128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.le.Tensor(unsqueeze_11, unsqueeze_8);  unsqueeze_11 = unsqueeze_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_33: "b8[32, 1, 128, 128][0, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(le, [32, -1, 128, 128]);  le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default_1: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "bf16[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_33, full_default_1, full_default_2);  expand_33 = full_default_1 = full_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:317 in forward, code: position_bias = position_bias + causal_mask
        add_72: "bf16[32, 6, 128, 128][98304, 1, 768, 6]cuda:0" = torch.ops.aten.add.Tensor(unsqueeze_17, where_2);  unsqueeze_17 = where_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_73: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_221, add_72);  view_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_209: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_73, torch.float32);  add_73 = None
        amax_8: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_209, [-1], True)
        sub_10: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_209, amax_8);  convert_element_type_209 = amax_8 = None
        exp_8: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_9: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_12: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        convert_element_type_210: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_12, torch.bfloat16);  div_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_37: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_210, [32, 6, 128, 128]);  convert_element_type_210 = None
        view_224: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_37, [192, 128, 128]);  expand_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_216: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_77, [4096, 512]);  mul_77 = None
        permute_101: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
        mm_58: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_216, permute_101);  view_216 = permute_101 = None
        view_217: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_58, [32, 128, 384]);  mm_58 = None
        view_218: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_217, [32, 128, -1, 64]);  view_217 = None
        permute_102: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_218, [0, 2, 1, 3]);  view_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_38: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_102, [32, 6, 128, 64]);  permute_102 = None
        clone_70: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_225: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_70, [192, 128, 64]);  clone_70 = None
        bmm_17: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_224, view_225);  view_224 = view_225 = None
        view_226: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_17, [32, 6, 128, 64]);  bmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_105: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
        clone_71: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_105, memory_format = torch.contiguous_format);  permute_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_227: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_71, [32, 128, -1]);  clone_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_228: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_227, [4096, 384]);  view_227 = None
        permute_106: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        mm_59: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_228, permute_106);  view_228 = permute_106 = None
        view_229: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_59, [32, 128, 512]);  mm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_74: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(embedding_2, view_229);  embedding_2 = view_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_215: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_74, torch.float32)
        pow_27: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_215, 2);  convert_element_type_215 = None
        mean_18: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_27, [-1], True);  pow_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_75: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_18, 1e-06);  mean_18 = None
        rsqrt_18: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        mul_79: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_74, rsqrt_18);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_216: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_79, torch.bfloat16);  mul_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_80: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg83_1, convert_element_type_216);  arg83_1 = convert_element_type_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_230: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_80, [4096, 512]);  mul_80 = None
        permute_107: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
        mm_60: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_230, permute_107);  view_230 = permute_107 = None
        view_231: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_60, [32, 128, 384]);  mm_60 = None
        view_232: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_231, [32, 128, -1, 64]);  view_231 = None
        permute_108: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_232, [0, 2, 1, 3]);  view_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_39: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_108, [32, 6, 128, 64]);  permute_108 = None
        clone_73: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_239: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_73, [192, 128, 64]);  clone_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:680 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(embedding, torch.float32)
        pow_1: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2);  convert_element_type = None
        mean: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_2: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(embedding, rsqrt);  rsqrt = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_1: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_1: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg2_1, convert_element_type_1);  arg2_1 = convert_element_type_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_1: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_1, [4096, 512])
        permute: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        mm: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_1, permute);  view_1 = permute = None
        view_2: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [32, 128, 384]);  mm = None
        view_3: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_2, [32, 128, -1, 64]);  view_2 = None
        permute_1: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_1: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_1, [32, 6, 128, 64]);  permute_1 = None
        clone_1: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_10: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_1, [192, 128, 64]);  clone_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_4: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_1, [4096, 512])
        permute_2: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        mm_1: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_4, permute_2);  view_4 = permute_2 = None
        view_5: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_1, [32, 128, 384]);  mm_1 = None
        view_6: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_5, [32, 128, -1, 64]);  view_5 = None
        permute_3: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_6: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        expand_2: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_6, [32, 6, 64, 128]);  permute_6 = None
        clone_2: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_11: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [192, 64, 128]);  clone_2 = None
        bmm: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_10, view_11);  view_10 = view_11 = None
        view_12: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm, [32, 6, 128, 128]);  bmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:233 in compute_bias, code: memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        iota_5: "i64[128][1]cuda:0" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_4: "i64[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_5, 0);  iota_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:232 in compute_bias, code: context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None] + past_seen_tokens
        iota_4: "i64[128][1]cuda:0" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_3: "i64[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_4, 1);  iota_4 = None
        add_3: "i64[128, 1][1, 1]cuda:0" = torch.ops.aten.add.Tensor(unsqueeze_3, 0);  unsqueeze_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:234 in compute_bias, code: relative_position = memory_position - context_position  # shape (query_length, key_length)
        sub: "i64[128, 128][128, 1]cuda:0" = torch.ops.aten.sub.Tensor(unsqueeze_4, add_3);  unsqueeze_4 = add_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:205 in _relative_position_bucket, code: relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        gt: "b8[128, 128][128, 1]cuda:0" = torch.ops.aten.gt.Scalar(sub, 0)
        convert_element_type_10: "i64[128, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt, torch.int64);  gt = None
        mul_2: "i64[128, 128][128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_10, 16);  convert_element_type_10 = None
        add_4: "i64[128, 128][128, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2, 0);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:206 in _relative_position_bucket, code: relative_position = torch.abs(relative_position)
        abs_1: "i64[128, 128][128, 1]cuda:0" = torch.ops.aten.abs.default(sub);  sub = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:213 in _relative_position_bucket, code: is_small = relative_position < max_exact
        lt: "b8[128, 128][128, 1]cuda:0" = torch.ops.aten.lt.Scalar(abs_1, 8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:217 in _relative_position_bucket, code: torch.log(relative_position.float() / max_exact)
        convert_element_type_11: "f32[128, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(abs_1, torch.float32)
        div: "f32[128, 128][128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_11, 8);  convert_element_type_11 = None
        log: "f32[128, 128][128, 1]cuda:0" = torch.ops.aten.log.default(div);  div = None
        div_1: "f32[128, 128][128, 1]cuda:0" = torch.ops.aten.div.Tensor(log, 2.772588722239781);  log = None
        mul_3: "f32[128, 128][128, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_1, 8);  div_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:220 in _relative_position_bucket, code: ).to(torch.long)
        convert_element_type_12: "i64[128, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3, torch.int64);  mul_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:216 in _relative_position_bucket, code: relative_position_if_large = max_exact + (
        add_5: "i64[128, 128][128, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_12, 8);  convert_element_type_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:222 in _relative_position_bucket, code: relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        full: "i64[128, 128][128, 1]cuda:0" = torch.ops.aten.full.default([128, 128], 15, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:221 in _relative_position_bucket, code: relative_position_if_large = torch.min(
        minimum: "i64[128, 128][128, 1]cuda:0" = torch.ops.aten.minimum.default(add_5, full);  add_5 = full = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:225 in _relative_position_bucket, code: relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        where_1: "i64[128, 128][128, 1]cuda:0" = torch.ops.aten.where.self(lt, abs_1, minimum);  lt = abs_1 = minimum = None
        add_6: "i64[128, 128][128, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4, where_1);  add_4 = where_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:241 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        embedding_1: "bf16[128, 128, 6][768, 6, 1]cuda:0" = torch.ops.aten.embedding.default(arg6_1, add_6);  arg6_1 = add_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:242 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        permute_7: "bf16[6, 128, 128][1, 768, 6]cuda:0" = torch.ops.aten.permute.default(embedding_1, [2, 0, 1]);  embedding_1 = None
        unsqueeze_5: "bf16[1, 6, 128, 128][6, 1, 768, 6]cuda:0" = torch.ops.aten.unsqueeze.default(permute_7, 0);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default: "bf16[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.full.default([32, 1, 128, 128], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:317 in forward, code: position_bias = position_bias + causal_mask
        add_7: "bf16[32, 6, 128, 128][98304, 1, 768, 6]cuda:0" = torch.ops.aten.add.Tensor(unsqueeze_5, full_default);  unsqueeze_5 = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_8: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_12, add_7);  view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_13: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_8, torch.float32);  add_8 = None
        amax: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_13, [-1], True)
        sub_1: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_13, amax);  convert_element_type_13 = amax = None
        exp: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_2: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_14: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_3: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_14, [32, 6, 128, 128]);  convert_element_type_14 = None
        view_15: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_3, [192, 128, 128]);  expand_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_7: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_1, [4096, 512]);  mul_1 = None
        permute_4: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        mm_2: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_7, permute_4);  view_7 = permute_4 = None
        view_8: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [32, 128, 384]);  mm_2 = None
        view_9: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_8, [32, 128, -1, 64]);  view_8 = None
        permute_5: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_4: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_5, [32, 6, 128, 64]);  permute_5 = None
        clone_4: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_16: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_4, [192, 128, 64]);  clone_4 = None
        bmm_1: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_15, view_16);  view_15 = view_16 = None
        view_17: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_1, [32, 6, 128, 64]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_8: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None
        clone_5: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_18: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_5, [32, 128, -1]);  clone_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_19: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_18, [4096, 384]);  view_18 = None
        permute_9: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        mm_3: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_19, permute_9);  view_19 = permute_9 = None
        view_20: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_3, [32, 128, 512]);  mm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_9: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(embedding, view_20);  embedding = view_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_19: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_9, torch.float32)
        pow_2: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_19, 2);  convert_element_type_19 = None
        mean_1: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_10: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        rsqrt_1: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_4: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_9, rsqrt_1);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_20: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_5: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg8_1, convert_element_type_20);  arg8_1 = convert_element_type_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_21: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_5, [4096, 512])
        permute_10: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        mm_4: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_21, permute_10);  view_21 = permute_10 = None
        view_22: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [32, 128, 1024]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_6: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_22, 0.5)
        pow_3: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_22, 3.0)
        mul_7: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_11: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_22, mul_7);  view_22 = mul_7 = None
        mul_8: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_11, 0.7978845608028654);  add_11 = None
        tanh: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.tanh.default(mul_8);  mul_8 = None
        add_12: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_9: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_6, add_12);  mul_6 = add_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_23: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_5, [4096, 512]);  mul_5 = None
        permute_11: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        mm_5: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_23, permute_11);  view_23 = permute_11 = None
        view_24: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_5, [32, 128, 1024]);  mm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_10: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_9, view_24);  mul_9 = view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_25: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_10, [4096, 1024]);  mul_10 = None
        permute_12: "bf16[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        mm_6: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_25, permute_12);  view_25 = permute_12 = None
        view_26: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [32, 128, 512]);  mm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_13: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_9, view_26);  add_9 = view_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_27: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_13, torch.float32)
        pow_4: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_27, 2);  convert_element_type_27 = None
        mean_2: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_14: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
        rsqrt_2: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        mul_11: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_13, rsqrt_2);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_28: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_11, torch.bfloat16);  mul_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_12: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg12_1, convert_element_type_28);  arg12_1 = convert_element_type_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_27: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_12, [4096, 512])
        permute_13: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        mm_7: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_27, permute_13);  view_27 = permute_13 = None
        view_28: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_7, [32, 128, 384]);  mm_7 = None
        view_29: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_28, [32, 128, -1, 64]);  view_28 = None
        permute_14: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_5: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_14, [32, 6, 128, 64]);  permute_14 = None
        clone_9: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_36: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_9, [192, 128, 64]);  clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_30: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_12, [4096, 512])
        permute_15: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        mm_8: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_30, permute_15);  view_30 = permute_15 = None
        view_31: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [32, 128, 384]);  mm_8 = None
        view_32: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_31, [32, 128, -1, 64]);  view_31 = None
        permute_16: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_19: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_16, [0, 1, 3, 2]);  permute_16 = None
        expand_6: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_19, [32, 6, 64, 128]);  permute_19 = None
        clone_10: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_37: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_10, [192, 64, 128]);  clone_10 = None
        bmm_2: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_36, view_37);  view_36 = view_37 = None
        view_38: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_2, [32, 6, 128, 128]);  bmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_15: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_38, add_7);  view_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_37: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_15, torch.float32);  add_15 = None
        amax_1: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_37, [-1], True)
        sub_2: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_37, amax_1);  convert_element_type_37 = amax_1 = None
        exp_1: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_2: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_3: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        convert_element_type_38: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_7: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_38, [32, 6, 128, 128]);  convert_element_type_38 = None
        view_41: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_7, [192, 128, 128]);  expand_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_33: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_12, [4096, 512]);  mul_12 = None
        permute_17: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        mm_9: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_33, permute_17);  view_33 = permute_17 = None
        view_34: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_9, [32, 128, 384]);  mm_9 = None
        view_35: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_34, [32, 128, -1, 64]);  view_34 = None
        permute_18: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_35, [0, 2, 1, 3]);  view_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_8: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_18, [32, 6, 128, 64]);  permute_18 = None
        clone_12: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_42: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_12, [192, 128, 64]);  clone_12 = None
        bmm_3: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_41, view_42);  view_41 = view_42 = None
        view_43: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_3, [32, 6, 128, 64]);  bmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_20: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_43, [0, 2, 1, 3]);  view_43 = None
        clone_13: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_20, memory_format = torch.contiguous_format);  permute_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_44: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_13, [32, 128, -1]);  clone_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_45: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_44, [4096, 384]);  view_44 = None
        permute_21: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        mm_10: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_45, permute_21);  view_45 = permute_21 = None
        view_46: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [32, 128, 512]);  mm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_16: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_13, view_46);  add_13 = view_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_43: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_16, torch.float32)
        pow_5: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_43, 2);  convert_element_type_43 = None
        mean_3: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_17: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None
        rsqrt_3: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        mul_13: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_16, rsqrt_3);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_44: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_13, torch.bfloat16);  mul_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_14: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg17_1, convert_element_type_44);  arg17_1 = convert_element_type_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_47: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_14, [4096, 512])
        permute_22: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        mm_11: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_47, permute_22);  view_47 = permute_22 = None
        view_48: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_11, [32, 128, 1024]);  mm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_15: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_48, 0.5)
        pow_6: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_48, 3.0)
        mul_16: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_18: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_48, mul_16);  view_48 = mul_16 = None
        mul_17: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_18, 0.7978845608028654);  add_18 = None
        tanh_1: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.tanh.default(mul_17);  mul_17 = None
        add_19: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_18: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_15, add_19);  mul_15 = add_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_49: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_14, [4096, 512]);  mul_14 = None
        permute_23: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        mm_12: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_49, permute_23);  view_49 = permute_23 = None
        view_50: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_12, [32, 128, 1024]);  mm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_19: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_18, view_50);  mul_18 = view_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_51: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_19, [4096, 1024]);  mul_19 = None
        permute_24: "bf16[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        mm_13: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_51, permute_24);  view_51 = permute_24 = None
        view_52: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_13, [32, 128, 512]);  mm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_20: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_16, view_52);  add_16 = view_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_51: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_20, torch.float32)
        pow_7: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_51, 2);  convert_element_type_51 = None
        mean_4: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_21: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_4, 1e-06);  mean_4 = None
        rsqrt_4: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        mul_20: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_20, rsqrt_4);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_52: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_20, torch.bfloat16);  mul_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_21: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg21_1, convert_element_type_52);  arg21_1 = convert_element_type_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_53: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_21, [4096, 512])
        permute_25: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        mm_14: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_53, permute_25);  view_53 = permute_25 = None
        view_54: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_14, [32, 128, 384]);  mm_14 = None
        view_55: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_54, [32, 128, -1, 64]);  view_54 = None
        permute_26: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_55, [0, 2, 1, 3]);  view_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_9: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_26, [32, 6, 128, 64]);  permute_26 = None
        clone_17: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_62: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_17, [192, 128, 64]);  clone_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_56: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_21, [4096, 512])
        permute_27: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        mm_15: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_56, permute_27);  view_56 = permute_27 = None
        view_57: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_15, [32, 128, 384]);  mm_15 = None
        view_58: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_57, [32, 128, -1, 64]);  view_57 = None
        permute_28: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_31: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_28, [0, 1, 3, 2]);  permute_28 = None
        expand_10: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_31, [32, 6, 64, 128]);  permute_31 = None
        clone_18: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_63: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_18, [192, 64, 128]);  clone_18 = None
        bmm_4: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_62, view_63);  view_62 = view_63 = None
        view_64: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_4, [32, 6, 128, 128]);  bmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_22: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_64, add_7);  view_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_61: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_22, torch.float32);  add_22 = None
        amax_2: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_61, [-1], True)
        sub_3: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_61, amax_2);  convert_element_type_61 = amax_2 = None
        exp_2: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        sum_3: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_4: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        convert_element_type_62: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_11: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_62, [32, 6, 128, 128]);  convert_element_type_62 = None
        view_67: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_11, [192, 128, 128]);  expand_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_59: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_21, [4096, 512]);  mul_21 = None
        permute_29: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        mm_16: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_59, permute_29);  view_59 = permute_29 = None
        view_60: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_16, [32, 128, 384]);  mm_16 = None
        view_61: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_60, [32, 128, -1, 64]);  view_60 = None
        permute_30: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_61, [0, 2, 1, 3]);  view_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_12: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_30, [32, 6, 128, 64]);  permute_30 = None
        clone_20: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_68: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_20, [192, 128, 64]);  clone_20 = None
        bmm_5: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_67, view_68);  view_67 = view_68 = None
        view_69: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_5, [32, 6, 128, 64]);  bmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_32: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None
        clone_21: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_70: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_21, [32, 128, -1]);  clone_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_71: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_70, [4096, 384]);  view_70 = None
        permute_33: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        mm_17: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_71, permute_33);  view_71 = permute_33 = None
        view_72: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_17, [32, 128, 512]);  mm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_23: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_20, view_72);  add_20 = view_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_67: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_23, torch.float32)
        pow_8: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_67, 2);  convert_element_type_67 = None
        mean_5: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_8, [-1], True);  pow_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_24: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_5, 1e-06);  mean_5 = None
        rsqrt_5: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        mul_22: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_23, rsqrt_5);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_68: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_22, torch.bfloat16);  mul_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_23: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg26_1, convert_element_type_68);  arg26_1 = convert_element_type_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_73: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_23, [4096, 512])
        permute_34: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        mm_18: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_73, permute_34);  view_73 = permute_34 = None
        view_74: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_18, [32, 128, 1024]);  mm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_24: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_74, 0.5)
        pow_9: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_74, 3.0)
        mul_25: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_25: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_74, mul_25);  view_74 = mul_25 = None
        mul_26: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_25, 0.7978845608028654);  add_25 = None
        tanh_2: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.tanh.default(mul_26);  mul_26 = None
        add_26: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_27: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, add_26);  mul_24 = add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_75: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_23, [4096, 512]);  mul_23 = None
        permute_35: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        mm_19: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_75, permute_35);  view_75 = permute_35 = None
        view_76: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_19, [32, 128, 1024]);  mm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_28: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_27, view_76);  mul_27 = view_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_77: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_28, [4096, 1024]);  mul_28 = None
        permute_36: "bf16[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        mm_20: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_77, permute_36);  view_77 = permute_36 = None
        view_78: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_20, [32, 128, 512]);  mm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_27: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_23, view_78);  add_23 = view_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_75: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_27, torch.float32)
        pow_10: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_75, 2);  convert_element_type_75 = None
        mean_6: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_28: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_6, 1e-06);  mean_6 = None
        rsqrt_6: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_29: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_27, rsqrt_6);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_76: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_29, torch.bfloat16);  mul_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_30: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg30_1, convert_element_type_76);  arg30_1 = convert_element_type_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_79: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_30, [4096, 512])
        permute_37: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        mm_21: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_79, permute_37);  view_79 = permute_37 = None
        view_80: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_21, [32, 128, 384]);  mm_21 = None
        view_81: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_80, [32, 128, -1, 64]);  view_80 = None
        permute_38: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_81, [0, 2, 1, 3]);  view_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_13: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_38, [32, 6, 128, 64]);  permute_38 = None
        clone_25: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_88: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_25, [192, 128, 64]);  clone_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_82: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_30, [4096, 512])
        permute_39: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        mm_22: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_82, permute_39);  view_82 = permute_39 = None
        view_83: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_22, [32, 128, 384]);  mm_22 = None
        view_84: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_83, [32, 128, -1, 64]);  view_83 = None
        permute_40: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_43: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_40, [0, 1, 3, 2]);  permute_40 = None
        expand_14: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_43, [32, 6, 64, 128]);  permute_43 = None
        clone_26: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_89: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_26, [192, 64, 128]);  clone_26 = None
        bmm_6: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_88, view_89);  view_88 = view_89 = None
        view_90: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_6, [32, 6, 128, 128]);  bmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_29: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_90, add_7);  view_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_85: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_29, torch.float32);  add_29 = None
        amax_3: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_85, [-1], True)
        sub_4: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_85, amax_3);  convert_element_type_85 = amax_3 = None
        exp_3: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_4: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_5: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        convert_element_type_86: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_15: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_86, [32, 6, 128, 128]);  convert_element_type_86 = None
        view_93: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_15, [192, 128, 128]);  expand_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_85: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_30, [4096, 512]);  mul_30 = None
        permute_41: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        mm_23: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_85, permute_41);  view_85 = permute_41 = None
        view_86: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_23, [32, 128, 384]);  mm_23 = None
        view_87: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_86, [32, 128, -1, 64]);  view_86 = None
        permute_42: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_16: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_42, [32, 6, 128, 64]);  permute_42 = None
        clone_28: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_94: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_28, [192, 128, 64]);  clone_28 = None
        bmm_7: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_93, view_94);  view_93 = view_94 = None
        view_95: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_7, [32, 6, 128, 64]);  bmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_44: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
        clone_29: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_96: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_29, [32, 128, -1]);  clone_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_97: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_96, [4096, 384]);  view_96 = None
        permute_45: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        mm_24: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_97, permute_45);  view_97 = permute_45 = None
        view_98: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_24, [32, 128, 512]);  mm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_30: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_27, view_98);  add_27 = view_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_91: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_30, torch.float32)
        pow_11: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_91, 2);  convert_element_type_91 = None
        mean_7: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_31: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_7, 1e-06);  mean_7 = None
        rsqrt_7: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        mul_31: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_30, rsqrt_7);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_92: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_31, torch.bfloat16);  mul_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_32: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg35_1, convert_element_type_92);  arg35_1 = convert_element_type_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_99: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_32, [4096, 512])
        permute_46: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        mm_25: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_99, permute_46);  view_99 = permute_46 = None
        view_100: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_25, [32, 128, 1024]);  mm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_33: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_100, 0.5)
        pow_12: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_100, 3.0)
        mul_34: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_32: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_100, mul_34);  view_100 = mul_34 = None
        mul_35: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_32, 0.7978845608028654);  add_32 = None
        tanh_3: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.tanh.default(mul_35);  mul_35 = None
        add_33: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_36: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_33, add_33);  mul_33 = add_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_101: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_32, [4096, 512]);  mul_32 = None
        permute_47: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        mm_26: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_101, permute_47);  view_101 = permute_47 = None
        view_102: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_26, [32, 128, 1024]);  mm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_37: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_36, view_102);  mul_36 = view_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_103: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_37, [4096, 1024]);  mul_37 = None
        permute_48: "bf16[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        mm_27: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_103, permute_48);  view_103 = permute_48 = None
        view_104: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_27, [32, 128, 512]);  mm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_34: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_30, view_104);  add_30 = view_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_99: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_34, torch.float32)
        pow_13: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_99, 2);  convert_element_type_99 = None
        mean_8: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_35: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_8, 1e-06);  mean_8 = None
        rsqrt_8: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        mul_38: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_34, rsqrt_8);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_100: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_38, torch.bfloat16);  mul_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_39: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg39_1, convert_element_type_100);  arg39_1 = convert_element_type_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_105: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_39, [4096, 512])
        permute_49: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        mm_28: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_105, permute_49);  view_105 = permute_49 = None
        view_106: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_28, [32, 128, 384]);  mm_28 = None
        view_107: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_106, [32, 128, -1, 64]);  view_106 = None
        permute_50: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_107, [0, 2, 1, 3]);  view_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_17: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_50, [32, 6, 128, 64]);  permute_50 = None
        clone_33: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_114: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_33, [192, 128, 64]);  clone_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_108: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_39, [4096, 512])
        permute_51: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        mm_29: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_108, permute_51);  view_108 = permute_51 = None
        view_109: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_29, [32, 128, 384]);  mm_29 = None
        view_110: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_109, [32, 128, -1, 64]);  view_109 = None
        permute_52: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_110, [0, 2, 1, 3]);  view_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_55: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_52, [0, 1, 3, 2]);  permute_52 = None
        expand_18: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_55, [32, 6, 64, 128]);  permute_55 = None
        clone_34: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_115: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_34, [192, 64, 128]);  clone_34 = None
        bmm_8: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_114, view_115);  view_114 = view_115 = None
        view_116: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_8, [32, 6, 128, 128]);  bmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_36: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_116, add_7);  view_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_109: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_36, torch.float32);  add_36 = None
        amax_4: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_109, [-1], True)
        sub_5: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_109, amax_4);  convert_element_type_109 = amax_4 = None
        exp_4: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        sum_5: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_6: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        convert_element_type_110: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_19: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_110, [32, 6, 128, 128]);  convert_element_type_110 = None
        view_119: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_19, [192, 128, 128]);  expand_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_111: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_39, [4096, 512]);  mul_39 = None
        permute_53: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        mm_30: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_111, permute_53);  view_111 = permute_53 = None
        view_112: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_30, [32, 128, 384]);  mm_30 = None
        view_113: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_112, [32, 128, -1, 64]);  view_112 = None
        permute_54: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_20: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_54, [32, 6, 128, 64]);  permute_54 = None
        clone_36: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_120: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_36, [192, 128, 64]);  clone_36 = None
        bmm_9: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_119, view_120);  view_119 = view_120 = None
        view_121: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_9, [32, 6, 128, 64]);  bmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_56: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_121, [0, 2, 1, 3]);  view_121 = None
        clone_37: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_122: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_37, [32, 128, -1]);  clone_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_123: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_122, [4096, 384]);  view_122 = None
        permute_57: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        mm_31: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_123, permute_57);  view_123 = permute_57 = None
        view_124: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_31, [32, 128, 512]);  mm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_37: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_34, view_124);  add_34 = view_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_115: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_37, torch.float32)
        pow_14: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_115, 2);  convert_element_type_115 = None
        mean_9: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_14, [-1], True);  pow_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_38: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_9, 1e-06);  mean_9 = None
        rsqrt_9: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        mul_40: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_37, rsqrt_9);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_116: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_40, torch.bfloat16);  mul_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_41: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg44_1, convert_element_type_116);  arg44_1 = convert_element_type_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_125: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_41, [4096, 512])
        permute_58: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        mm_32: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_125, permute_58);  view_125 = permute_58 = None
        view_126: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_32, [32, 128, 1024]);  mm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_42: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_126, 0.5)
        pow_15: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_126, 3.0)
        mul_43: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
        add_39: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_126, mul_43);  view_126 = mul_43 = None
        mul_44: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_39, 0.7978845608028654);  add_39 = None
        tanh_4: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.tanh.default(mul_44);  mul_44 = None
        add_40: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_45: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, add_40);  mul_42 = add_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_127: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_41, [4096, 512]);  mul_41 = None
        permute_59: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        mm_33: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_127, permute_59);  view_127 = permute_59 = None
        view_128: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_33, [32, 128, 1024]);  mm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_46: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_45, view_128);  mul_45 = view_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_129: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_46, [4096, 1024]);  mul_46 = None
        permute_60: "bf16[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        mm_34: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_129, permute_60);  view_129 = permute_60 = None
        view_130: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_34, [32, 128, 512]);  mm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_41: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_37, view_130);  add_37 = view_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_123: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_41, torch.float32)
        pow_16: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_123, 2);  convert_element_type_123 = None
        mean_10: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_42: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_10, 1e-06);  mean_10 = None
        rsqrt_10: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_47: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_41, rsqrt_10);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_124: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_47, torch.bfloat16);  mul_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_48: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg48_1, convert_element_type_124);  arg48_1 = convert_element_type_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_131: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_48, [4096, 512])
        permute_61: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        mm_35: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_131, permute_61);  view_131 = permute_61 = None
        view_132: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_35, [32, 128, 384]);  mm_35 = None
        view_133: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_132, [32, 128, -1, 64]);  view_132 = None
        permute_62: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_21: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_62, [32, 6, 128, 64]);  permute_62 = None
        clone_41: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_140: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_41, [192, 128, 64]);  clone_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_134: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_48, [4096, 512])
        permute_63: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        mm_36: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_134, permute_63);  view_134 = permute_63 = None
        view_135: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_36, [32, 128, 384]);  mm_36 = None
        view_136: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_135, [32, 128, -1, 64]);  view_135 = None
        permute_64: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_67: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_64, [0, 1, 3, 2]);  permute_64 = None
        expand_22: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_67, [32, 6, 64, 128]);  permute_67 = None
        clone_42: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_141: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_42, [192, 64, 128]);  clone_42 = None
        bmm_10: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_140, view_141);  view_140 = view_141 = None
        view_142: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_10, [32, 6, 128, 128]);  bmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_43: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_142, add_7);  view_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_133: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_43, torch.float32);  add_43 = None
        amax_5: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_133, [-1], True)
        sub_6: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_133, amax_5);  convert_element_type_133 = amax_5 = None
        exp_5: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_6: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_7: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        convert_element_type_134: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_23: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_134, [32, 6, 128, 128]);  convert_element_type_134 = None
        view_145: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_23, [192, 128, 128]);  expand_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_137: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_48, [4096, 512]);  mul_48 = None
        permute_65: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        mm_37: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_137, permute_65);  view_137 = permute_65 = None
        view_138: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_37, [32, 128, 384]);  mm_37 = None
        view_139: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_138, [32, 128, -1, 64]);  view_138 = None
        permute_66: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_24: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_66, [32, 6, 128, 64]);  permute_66 = None
        clone_44: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_146: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_44, [192, 128, 64]);  clone_44 = None
        bmm_11: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_145, view_146);  view_145 = view_146 = None
        view_147: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_11, [32, 6, 128, 64]);  bmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_68: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_147, [0, 2, 1, 3]);  view_147 = None
        clone_45: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_148: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_45, [32, 128, -1]);  clone_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_149: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_148, [4096, 384]);  view_148 = None
        permute_69: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
        mm_38: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_149, permute_69);  view_149 = permute_69 = None
        view_150: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_38, [32, 128, 512]);  mm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_44: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_41, view_150);  add_41 = view_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_139: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_44, torch.float32)
        pow_17: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_139, 2);  convert_element_type_139 = None
        mean_11: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_45: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_11, 1e-06);  mean_11 = None
        rsqrt_11: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        mul_49: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_44, rsqrt_11);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_140: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_49, torch.bfloat16);  mul_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_50: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg53_1, convert_element_type_140);  arg53_1 = convert_element_type_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_151: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_50, [4096, 512])
        permute_70: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        mm_39: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_151, permute_70);  view_151 = permute_70 = None
        view_152: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_39, [32, 128, 1024]);  mm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_51: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_152, 0.5)
        pow_18: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_152, 3.0)
        mul_52: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_18, 0.044715);  pow_18 = None
        add_46: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_152, mul_52);  view_152 = mul_52 = None
        mul_53: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_46, 0.7978845608028654);  add_46 = None
        tanh_5: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.tanh.default(mul_53);  mul_53 = None
        add_47: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_54: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_51, add_47);  mul_51 = add_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_153: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_50, [4096, 512]);  mul_50 = None
        permute_71: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        mm_40: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_153, permute_71);  view_153 = permute_71 = None
        view_154: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_40, [32, 128, 1024]);  mm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_55: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_54, view_154);  mul_54 = view_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_155: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_55, [4096, 1024]);  mul_55 = None
        permute_72: "bf16[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        mm_41: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_155, permute_72);  view_155 = permute_72 = None
        view_156: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_41, [32, 128, 512]);  mm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_48: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_44, view_156);  add_44 = view_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_147: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_48, torch.float32)
        pow_19: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_147, 2);  convert_element_type_147 = None
        mean_12: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_49: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_12, 1e-06);  mean_12 = None
        rsqrt_12: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        mul_56: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_48, rsqrt_12);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_148: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_56, torch.bfloat16);  mul_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_57: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg57_1, convert_element_type_148);  arg57_1 = convert_element_type_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_157: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_57, [4096, 512])
        permute_73: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        mm_42: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_157, permute_73);  view_157 = permute_73 = None
        view_158: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_42, [32, 128, 384]);  mm_42 = None
        view_159: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_158, [32, 128, -1, 64]);  view_158 = None
        permute_74: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_159, [0, 2, 1, 3]);  view_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_25: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_74, [32, 6, 128, 64]);  permute_74 = None
        clone_49: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_166: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_49, [192, 128, 64]);  clone_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_160: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_57, [4096, 512])
        permute_75: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        mm_43: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_160, permute_75);  view_160 = permute_75 = None
        view_161: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_43, [32, 128, 384]);  mm_43 = None
        view_162: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_161, [32, 128, -1, 64]);  view_161 = None
        permute_76: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_79: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_76, [0, 1, 3, 2]);  permute_76 = None
        expand_26: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_79, [32, 6, 64, 128]);  permute_79 = None
        clone_50: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_167: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_50, [192, 64, 128]);  clone_50 = None
        bmm_12: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_166, view_167);  view_166 = view_167 = None
        view_168: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_12, [32, 6, 128, 128]);  bmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_50: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_168, add_7);  view_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_157: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_50, torch.float32);  add_50 = None
        amax_6: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_157, [-1], True)
        sub_7: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_157, amax_6);  convert_element_type_157 = amax_6 = None
        exp_6: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_7: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_8: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        convert_element_type_158: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_27: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_158, [32, 6, 128, 128]);  convert_element_type_158 = None
        view_171: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_27, [192, 128, 128]);  expand_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_163: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_57, [4096, 512]);  mul_57 = None
        permute_77: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        mm_44: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_163, permute_77);  view_163 = permute_77 = None
        view_164: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_44, [32, 128, 384]);  mm_44 = None
        view_165: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_164, [32, 128, -1, 64]);  view_164 = None
        permute_78: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_28: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_78, [32, 6, 128, 64]);  permute_78 = None
        clone_52: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_172: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_52, [192, 128, 64]);  clone_52 = None
        bmm_13: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_171, view_172);  view_171 = view_172 = None
        view_173: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_13, [32, 6, 128, 64]);  bmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_80: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_173, [0, 2, 1, 3]);  view_173 = None
        clone_53: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_174: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_53, [32, 128, -1]);  clone_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_175: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_174, [4096, 384]);  view_174 = None
        permute_81: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        mm_45: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_175, permute_81);  view_175 = permute_81 = None
        view_176: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_45, [32, 128, 512]);  mm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_51: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_48, view_176);  add_48 = view_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_163: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_51, torch.float32)
        pow_20: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_163, 2);  convert_element_type_163 = None
        mean_13: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_20, [-1], True);  pow_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_52: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_13, 1e-06);  mean_13 = None
        rsqrt_13: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_58: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_51, rsqrt_13);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_164: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_58, torch.bfloat16);  mul_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_59: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg62_1, convert_element_type_164);  arg62_1 = convert_element_type_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_177: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_59, [4096, 512])
        permute_82: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        mm_46: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_177, permute_82);  view_177 = permute_82 = None
        view_178: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_46, [32, 128, 1024]);  mm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_60: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_178, 0.5)
        pow_21: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_178, 3.0)
        mul_61: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
        add_53: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_178, mul_61);  view_178 = mul_61 = None
        mul_62: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_53, 0.7978845608028654);  add_53 = None
        tanh_6: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.tanh.default(mul_62);  mul_62 = None
        add_54: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_63: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_60, add_54);  mul_60 = add_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_179: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_59, [4096, 512]);  mul_59 = None
        permute_83: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        mm_47: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_179, permute_83);  view_179 = permute_83 = None
        view_180: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_47, [32, 128, 1024]);  mm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_64: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_63, view_180);  mul_63 = view_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_181: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_64, [4096, 1024]);  mul_64 = None
        permute_84: "bf16[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        mm_48: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_181, permute_84);  view_181 = permute_84 = None
        view_182: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_48, [32, 128, 512]);  mm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_55: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_51, view_182);  add_51 = view_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_171: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_55, torch.float32)
        pow_22: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_171, 2);  convert_element_type_171 = None
        mean_14: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_22, [-1], True);  pow_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_56: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_14, 1e-06);  mean_14 = None
        rsqrt_14: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        mul_65: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_55, rsqrt_14);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_172: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_65, torch.bfloat16);  mul_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_66: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg66_1, convert_element_type_172);  arg66_1 = convert_element_type_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_183: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_66, [4096, 512])
        permute_85: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        mm_49: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_183, permute_85);  view_183 = permute_85 = None
        view_184: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_49, [32, 128, 384]);  mm_49 = None
        view_185: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_184, [32, 128, -1, 64]);  view_184 = None
        permute_86: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_29: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_86, [32, 6, 128, 64]);  permute_86 = None
        clone_57: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_192: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_57, [192, 128, 64]);  clone_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_186: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_66, [4096, 512])
        permute_87: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        mm_50: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_186, permute_87);  view_186 = permute_87 = None
        view_187: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_50, [32, 128, 384]);  mm_50 = None
        view_188: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_187, [32, 128, -1, 64]);  view_187 = None
        permute_88: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_188, [0, 2, 1, 3]);  view_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_91: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_88, [0, 1, 3, 2]);  permute_88 = None
        expand_30: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_91, [32, 6, 64, 128]);  permute_91 = None
        clone_58: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_193: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_58, [192, 64, 128]);  clone_58 = None
        bmm_14: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_192, view_193);  view_192 = view_193 = None
        view_194: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_14, [32, 6, 128, 128]);  bmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_57: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_194, add_7);  view_194 = add_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_181: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_57, torch.float32);  add_57 = None
        amax_7: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_181, [-1], True)
        sub_8: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_181, amax_7);  convert_element_type_181 = amax_7 = None
        exp_7: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
        sum_8: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_9: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        convert_element_type_182: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_31: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_182, [32, 6, 128, 128]);  convert_element_type_182 = None
        view_197: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_31, [192, 128, 128]);  expand_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_189: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_66, [4096, 512]);  mul_66 = None
        permute_89: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        mm_51: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_189, permute_89);  view_189 = permute_89 = None
        view_190: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_51, [32, 128, 384]);  mm_51 = None
        view_191: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_190, [32, 128, -1, 64]);  view_190 = None
        permute_90: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_191, [0, 2, 1, 3]);  view_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_32: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_90, [32, 6, 128, 64]);  permute_90 = None
        clone_60: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_198: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_60, [192, 128, 64]);  clone_60 = None
        bmm_15: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_197, view_198);  view_197 = view_198 = None
        view_199: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_15, [32, 6, 128, 64]);  bmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_92: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_199, [0, 2, 1, 3]);  view_199 = None
        clone_61: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_200: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_61, [32, 128, -1]);  clone_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_201: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_200, [4096, 384]);  view_200 = None
        permute_93: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        mm_52: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_201, permute_93);  view_201 = permute_93 = None
        view_202: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_52, [32, 128, 512]);  mm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_58: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_55, view_202);  add_55 = view_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_187: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_58, torch.float32)
        pow_23: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_187, 2);  convert_element_type_187 = None
        mean_15: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_23, [-1], True);  pow_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_59: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_15, 1e-06);  mean_15 = None
        rsqrt_15: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
        mul_67: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_58, rsqrt_15);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_188: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_67, torch.bfloat16);  mul_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_68: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg71_1, convert_element_type_188);  arg71_1 = convert_element_type_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_203: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_68, [4096, 512])
        permute_94: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        mm_53: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_203, permute_94);  view_203 = permute_94 = None
        view_204: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_53, [32, 128, 1024]);  mm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_69: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_204, 0.5)
        pow_24: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_204, 3.0)
        mul_70: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_24, 0.044715);  pow_24 = None
        add_60: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_204, mul_70);  view_204 = mul_70 = None
        mul_71: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_60, 0.7978845608028654);  add_60 = None
        tanh_7: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.tanh.default(mul_71);  mul_71 = None
        add_61: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_72: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_69, add_61);  mul_69 = add_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_205: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_68, [4096, 512]);  mul_68 = None
        permute_95: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        mm_54: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_205, permute_95);  view_205 = permute_95 = None
        view_206: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_54, [32, 128, 1024]);  mm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_73: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_72, view_206);  mul_72 = view_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_207: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_73, [4096, 1024]);  mul_73 = None
        permute_96: "bf16[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        mm_55: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_207, permute_96);  view_207 = permute_96 = None
        view_208: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_55, [32, 128, 512]);  mm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_62: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_58, view_208);  add_58 = view_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_195: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_62, torch.float32)
        pow_25: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_195, 2);  convert_element_type_195 = None
        mean_16: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_63: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_16, 1e-06);  mean_16 = None
        rsqrt_16: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
        mul_74: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_62, rsqrt_16);  add_62 = rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_196: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_74, torch.bfloat16);  mul_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_75: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg75_1, convert_element_type_196);  arg75_1 = convert_element_type_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_233: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_75, [4096, 512])
        permute_109: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        mm_61: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_233, permute_109);  view_233 = permute_109 = None
        view_234: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_61, [32, 128, 384]);  mm_61 = None
        view_235: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_234, [32, 128, -1, 64]);  view_234 = None
        permute_110: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_235, [0, 2, 1, 3]);  view_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_113: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_110, [0, 1, 3, 2]);  permute_110 = None
        expand_40: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_113, [32, 6, 64, 128]);  permute_113 = None
        clone_74: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_240: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_74, [192, 64, 128]);  clone_74 = None
        bmm_18: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_239, view_240);  view_239 = view_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        view_241: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_18, [32, 6, 128, 128]);  bmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_225: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_241, torch.float32);  view_241 = None
        amax_9: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_225, [-1], True)
        sub_11: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_225, amax_9);  convert_element_type_225 = amax_9 = None
        exp_9: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        sum_10: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_13: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        convert_element_type_226: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_41: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_226, [32, 6, 128, 128]);  convert_element_type_226 = None
        view_244: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_41, [192, 128, 128]);  expand_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_236: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_75, [4096, 512])
        permute_111: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        mm_62: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_236, permute_111);  view_236 = permute_111 = None
        view_237: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_62, [32, 128, 384]);  mm_62 = None
        view_238: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_237, [32, 128, -1, 64]);  view_237 = None
        permute_112: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_238, [0, 2, 1, 3]);  view_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_42: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_112, [32, 6, 128, 64]);  permute_112 = None
        clone_76: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_245: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_76, [192, 128, 64]);  clone_76 = None
        bmm_19: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_244, view_245);  view_244 = view_245 = None
        view_246: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_19, [32, 6, 128, 64]);  bmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_114: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
        clone_77: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_247: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_77, [32, 128, -1]);  clone_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_248: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_247, [4096, 384]);  view_247 = None
        permute_115: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        mm_63: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_248, permute_115);  view_248 = permute_115 = None
        view_249: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_63, [32, 128, 512]);  mm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:400 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_78: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_74, view_249);  add_74 = view_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_231: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_78, torch.float32)
        pow_28: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_231, 2);  convert_element_type_231 = None
        mean_19: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_28, [-1], True);  pow_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_79: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_19, 1e-06);  mean_19 = None
        rsqrt_19: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
        mul_81: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_78, rsqrt_19);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_232: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_81, torch.bfloat16);  mul_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_82: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg88_1, convert_element_type_232);  arg88_1 = convert_element_type_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_250: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_82, [4096, 512])
        permute_116: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        mm_64: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_250, permute_116);  view_250 = permute_116 = None
        view_251: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_64, [32, 128, 1024]);  mm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_83: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_251, 0.5)
        pow_29: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_251, 3.0)
        mul_84: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_29, 0.044715);  pow_29 = None
        add_80: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_251, mul_84);  view_251 = mul_84 = None
        mul_85: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_80, 0.7978845608028654);  add_80 = None
        tanh_8: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.tanh.default(mul_85);  mul_85 = None
        add_81: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_86: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_83, add_81);  mul_83 = add_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_252: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_82, [4096, 512]);  mul_82 = None
        permute_117: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        mm_65: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_252, permute_117);  view_252 = permute_117 = None
        view_253: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_65, [32, 128, 1024]);  mm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_87: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_86, view_253);  mul_86 = view_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_254: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_87, [4096, 1024]);  mul_87 = None
        permute_118: "bf16[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        mm_66: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_254, permute_118);  view_254 = permute_118 = None
        view_255: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_66, [32, 128, 512]);  mm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_82: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_78, view_255);  add_78 = view_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_239: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_82, torch.float32)
        pow_30: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_239, 2);  convert_element_type_239 = None
        mean_20: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_30, [-1], True);  pow_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_83: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_20, 1e-06);  mean_20 = None
        rsqrt_20: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
        mul_88: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_82, rsqrt_20);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_240: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_88, torch.bfloat16);  mul_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_89: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg92_1, convert_element_type_240);  arg92_1 = convert_element_type_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_256: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_89, [4096, 512])
        permute_119: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        mm_67: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_256, permute_119);  view_256 = permute_119 = None
        view_257: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_67, [32, 128, 384]);  mm_67 = None
        view_258: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_257, [32, 128, -1, 64]);  view_257 = None
        permute_120: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_43: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_120, [32, 6, 128, 64]);  permute_120 = None
        clone_81: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_265: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_81, [192, 128, 64]);  clone_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_259: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_89, [4096, 512])
        permute_121: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        mm_68: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_259, permute_121);  view_259 = permute_121 = None
        view_260: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_68, [32, 128, 384]);  mm_68 = None
        view_261: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_260, [32, 128, -1, 64]);  view_260 = None
        permute_122: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_261, [0, 2, 1, 3]);  view_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_125: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_122, [0, 1, 3, 2]);  permute_122 = None
        expand_44: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_125, [32, 6, 64, 128]);  permute_125 = None
        clone_82: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_266: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_82, [192, 64, 128]);  clone_82 = None
        bmm_20: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_265, view_266);  view_265 = view_266 = None
        view_267: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_20, [32, 6, 128, 128]);  bmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_84: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_267, add_72);  view_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_249: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_84, torch.float32);  add_84 = None
        amax_10: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_249, [-1], True)
        sub_12: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_249, amax_10);  convert_element_type_249 = amax_10 = None
        exp_10: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_11: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_14: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        convert_element_type_250: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_14, torch.bfloat16);  div_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_45: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_250, [32, 6, 128, 128]);  convert_element_type_250 = None
        view_270: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_45, [192, 128, 128]);  expand_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_262: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_89, [4096, 512]);  mul_89 = None
        permute_123: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        mm_69: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_262, permute_123);  view_262 = permute_123 = None
        view_263: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_69, [32, 128, 384]);  mm_69 = None
        view_264: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_263, [32, 128, -1, 64]);  view_263 = None
        permute_124: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_264, [0, 2, 1, 3]);  view_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_46: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_124, [32, 6, 128, 64]);  permute_124 = None
        clone_84: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_271: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_84, [192, 128, 64]);  clone_84 = None
        bmm_21: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_270, view_271);  view_270 = view_271 = None
        view_272: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_21, [32, 6, 128, 64]);  bmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_126: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_272, [0, 2, 1, 3]);  view_272 = None
        clone_85: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_273: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_85, [32, 128, -1]);  clone_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_274: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_273, [4096, 384]);  view_273 = None
        permute_127: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        mm_70: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_274, permute_127);  view_274 = permute_127 = None
        view_275: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_70, [32, 128, 512]);  mm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_85: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_82, view_275);  add_82 = view_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_255: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_85, torch.float32)
        pow_31: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_255, 2);  convert_element_type_255 = None
        mean_21: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_31, [-1], True);  pow_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_86: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_21, 1e-06);  mean_21 = None
        rsqrt_21: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        mul_90: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_85, rsqrt_21);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_256: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_90, torch.bfloat16);  mul_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_91: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg97_1, convert_element_type_256);  arg97_1 = convert_element_type_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_276: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_91, [4096, 512]);  mul_91 = None
        permute_128: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        mm_71: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_276, permute_128);  view_276 = permute_128 = None
        view_277: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_71, [32, 128, 384]);  mm_71 = None
        view_278: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_277, [32, 128, -1, 64]);  view_277 = None
        permute_129: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_278, [0, 2, 1, 3]);  view_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_47: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_129, [32, 6, 128, 64]);  permute_129 = None
        clone_87: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_285: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_87, [192, 128, 64]);  clone_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_279: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_75, [4096, 512])
        permute_130: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        mm_72: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_279, permute_130);  view_279 = permute_130 = None
        view_280: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_72, [32, 128, 384]);  mm_72 = None
        view_281: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_280, [32, 128, -1, 64]);  view_280 = None
        permute_131: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_281, [0, 2, 1, 3]);  view_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_134: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_131, [0, 1, 3, 2]);  permute_131 = None
        expand_48: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_134, [32, 6, 64, 128]);  permute_134 = None
        clone_88: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_286: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_88, [192, 64, 128]);  clone_88 = None
        bmm_22: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_285, view_286);  view_285 = view_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        view_287: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_22, [32, 6, 128, 128]);  bmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_265: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_287, torch.float32);  view_287 = None
        amax_11: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_265, [-1], True)
        sub_13: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_265, amax_11);  convert_element_type_265 = amax_11 = None
        exp_11: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_12: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_15: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        convert_element_type_266: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_49: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_266, [32, 6, 128, 128]);  convert_element_type_266 = None
        view_290: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_49, [192, 128, 128]);  expand_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_282: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_75, [4096, 512])
        permute_132: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        mm_73: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_282, permute_132);  view_282 = permute_132 = None
        view_283: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_73, [32, 128, 384]);  mm_73 = None
        view_284: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_283, [32, 128, -1, 64]);  view_283 = None
        permute_133: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_284, [0, 2, 1, 3]);  view_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_50: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_133, [32, 6, 128, 64]);  permute_133 = None
        clone_90: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_291: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_90, [192, 128, 64]);  clone_90 = None
        bmm_23: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_290, view_291);  view_290 = view_291 = None
        view_292: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_23, [32, 6, 128, 64]);  bmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_135: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None
        clone_91: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_293: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_91, [32, 128, -1]);  clone_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_294: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_293, [4096, 384]);  view_293 = None
        permute_136: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        mm_74: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_294, permute_136);  view_294 = permute_136 = None
        view_295: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_74, [32, 128, 512]);  mm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:400 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_88: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_85, view_295);  add_85 = view_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_271: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_88, torch.float32)
        pow_32: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_271, 2);  convert_element_type_271 = None
        mean_22: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_32, [-1], True);  pow_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_89: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_22, 1e-06);  mean_22 = None
        rsqrt_22: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
        mul_92: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_88, rsqrt_22);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_272: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_92, torch.bfloat16);  mul_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_93: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg102_1, convert_element_type_272);  arg102_1 = convert_element_type_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_296: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_93, [4096, 512])
        permute_137: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        mm_75: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_296, permute_137);  view_296 = permute_137 = None
        view_297: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_75, [32, 128, 1024]);  mm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_94: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_297, 0.5)
        pow_33: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_297, 3.0)
        mul_95: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_33, 0.044715);  pow_33 = None
        add_90: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_297, mul_95);  view_297 = mul_95 = None
        mul_96: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_90, 0.7978845608028654);  add_90 = None
        tanh_9: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.tanh.default(mul_96);  mul_96 = None
        add_91: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_97: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_94, add_91);  mul_94 = add_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_298: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_93, [4096, 512]);  mul_93 = None
        permute_138: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        mm_76: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_298, permute_138);  view_298 = permute_138 = None
        view_299: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_76, [32, 128, 1024]);  mm_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_98: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_97, view_299);  mul_97 = view_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_300: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_98, [4096, 1024]);  mul_98 = None
        permute_139: "bf16[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        mm_77: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_300, permute_139);  view_300 = permute_139 = None
        view_301: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_77, [32, 128, 512]);  mm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_92: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_88, view_301);  add_88 = view_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_279: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_92, torch.float32)
        pow_34: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_279, 2);  convert_element_type_279 = None
        mean_23: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_34, [-1], True);  pow_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_93: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_23, 1e-06);  mean_23 = None
        rsqrt_23: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        mul_99: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_92, rsqrt_23);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_280: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_99, torch.bfloat16);  mul_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_100: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg106_1, convert_element_type_280);  arg106_1 = convert_element_type_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_302: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_100, [4096, 512])
        permute_140: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        mm_78: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_302, permute_140);  view_302 = permute_140 = None
        view_303: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_78, [32, 128, 384]);  mm_78 = None
        view_304: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_303, [32, 128, -1, 64]);  view_303 = None
        permute_141: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_304, [0, 2, 1, 3]);  view_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_51: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_141, [32, 6, 128, 64]);  permute_141 = None
        clone_95: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        view_311: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_95, [192, 128, 64]);  clone_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_305: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_100, [4096, 512])
        permute_142: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        mm_79: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_305, permute_142);  view_305 = permute_142 = None
        view_306: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_79, [32, 128, 384]);  mm_79 = None
        view_307: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_306, [32, 128, -1, 64]);  view_306 = None
        permute_143: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_307, [0, 2, 1, 3]);  view_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_146: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_143, [0, 1, 3, 2]);  permute_143 = None
        expand_52: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_146, [32, 6, 64, 128]);  permute_146 = None
        clone_96: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_312: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_96, [192, 64, 128]);  clone_96 = None
        bmm_24: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_311, view_312);  view_311 = view_312 = None
        view_313: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_24, [32, 6, 128, 128]);  bmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_94: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_313, add_72);  view_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_289: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_94, torch.float32);  add_94 = None
        amax_12: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_289, [-1], True)
        sub_14: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_289, amax_12);  convert_element_type_289 = amax_12 = None
        exp_12: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
        sum_13: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_16: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        convert_element_type_290: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_16, torch.bfloat16);  div_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_53: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_290, [32, 6, 128, 128]);  convert_element_type_290 = None
        view_316: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_53, [192, 128, 128]);  expand_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_308: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_100, [4096, 512]);  mul_100 = None
        permute_144: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        mm_80: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_308, permute_144);  view_308 = permute_144 = None
        view_309: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_80, [32, 128, 384]);  mm_80 = None
        view_310: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_309, [32, 128, -1, 64]);  view_309 = None
        permute_145: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_310, [0, 2, 1, 3]);  view_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_54: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_145, [32, 6, 128, 64]);  permute_145 = None
        clone_98: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_54, memory_format = torch.contiguous_format);  expand_54 = None
        view_317: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_98, [192, 128, 64]);  clone_98 = None
        bmm_25: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_316, view_317);  view_316 = view_317 = None
        view_318: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_25, [32, 6, 128, 64]);  bmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_147: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_318, [0, 2, 1, 3]);  view_318 = None
        clone_99: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_319: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_99, [32, 128, -1]);  clone_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_320: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_319, [4096, 384]);  view_319 = None
        permute_148: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        mm_81: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_320, permute_148);  view_320 = permute_148 = None
        view_321: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_81, [32, 128, 512]);  mm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_95: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_92, view_321);  add_92 = view_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_295: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_95, torch.float32)
        pow_35: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_295, 2);  convert_element_type_295 = None
        mean_24: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_35, [-1], True);  pow_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_96: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_24, 1e-06);  mean_24 = None
        rsqrt_24: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        mul_101: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_95, rsqrt_24);  rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_296: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_101, torch.bfloat16);  mul_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_102: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg111_1, convert_element_type_296);  arg111_1 = convert_element_type_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_322: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_102, [4096, 512]);  mul_102 = None
        permute_149: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
        mm_82: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_322, permute_149);  view_322 = permute_149 = None
        view_323: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_82, [32, 128, 384]);  mm_82 = None
        view_324: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_323, [32, 128, -1, 64]);  view_323 = None
        permute_150: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_324, [0, 2, 1, 3]);  view_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_55: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_150, [32, 6, 128, 64]);  permute_150 = None
        clone_101: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
        view_331: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_101, [192, 128, 64]);  clone_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_325: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_75, [4096, 512])
        permute_151: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        mm_83: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_325, permute_151);  view_325 = permute_151 = None
        view_326: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_83, [32, 128, 384]);  mm_83 = None
        view_327: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_326, [32, 128, -1, 64]);  view_326 = None
        permute_152: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_327, [0, 2, 1, 3]);  view_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_155: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_152, [0, 1, 3, 2]);  permute_152 = None
        expand_56: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_155, [32, 6, 64, 128]);  permute_155 = None
        clone_102: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_332: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_102, [192, 64, 128]);  clone_102 = None
        bmm_26: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_331, view_332);  view_331 = view_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        view_333: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_26, [32, 6, 128, 128]);  bmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_305: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_333, torch.float32);  view_333 = None
        amax_13: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_305, [-1], True)
        sub_15: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_305, amax_13);  convert_element_type_305 = amax_13 = None
        exp_13: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
        sum_14: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_17: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        convert_element_type_306: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_17, torch.bfloat16);  div_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_57: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_306, [32, 6, 128, 128]);  convert_element_type_306 = None
        view_336: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_57, [192, 128, 128]);  expand_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_328: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_75, [4096, 512])
        permute_153: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        mm_84: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_328, permute_153);  view_328 = permute_153 = None
        view_329: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_84, [32, 128, 384]);  mm_84 = None
        view_330: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_329, [32, 128, -1, 64]);  view_329 = None
        permute_154: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_58: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_154, [32, 6, 128, 64]);  permute_154 = None
        clone_104: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
        view_337: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_104, [192, 128, 64]);  clone_104 = None
        bmm_27: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_336, view_337);  view_336 = view_337 = None
        view_338: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_27, [32, 6, 128, 64]);  bmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_156: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None
        clone_105: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_156, memory_format = torch.contiguous_format);  permute_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_339: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_105, [32, 128, -1]);  clone_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_340: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_339, [4096, 384]);  view_339 = None
        permute_157: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        mm_85: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_340, permute_157);  view_340 = permute_157 = None
        view_341: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_85, [32, 128, 512]);  mm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:400 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_98: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_95, view_341);  add_95 = view_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_311: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_98, torch.float32)
        pow_36: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_311, 2);  convert_element_type_311 = None
        mean_25: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_36, [-1], True);  pow_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_99: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_25, 1e-06);  mean_25 = None
        rsqrt_25: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
        mul_103: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_98, rsqrt_25);  rsqrt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_312: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_103, torch.bfloat16);  mul_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_104: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg116_1, convert_element_type_312);  arg116_1 = convert_element_type_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_342: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_104, [4096, 512])
        permute_158: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        mm_86: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_342, permute_158);  view_342 = permute_158 = None
        view_343: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_86, [32, 128, 1024]);  mm_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_105: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_343, 0.5)
        pow_37: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_343, 3.0)
        mul_106: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_37, 0.044715);  pow_37 = None
        add_100: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_343, mul_106);  view_343 = mul_106 = None
        mul_107: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_100, 0.7978845608028654);  add_100 = None
        tanh_10: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.tanh.default(mul_107);  mul_107 = None
        add_101: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_108: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_105, add_101);  mul_105 = add_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_344: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_104, [4096, 512]);  mul_104 = None
        permute_159: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        mm_87: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_344, permute_159);  view_344 = permute_159 = None
        view_345: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_87, [32, 128, 1024]);  mm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_109: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_108, view_345);  mul_108 = view_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_346: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_109, [4096, 1024]);  mul_109 = None
        permute_160: "bf16[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        mm_88: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_346, permute_160);  view_346 = permute_160 = None
        view_347: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_88, [32, 128, 512]);  mm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_102: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_98, view_347);  add_98 = view_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_319: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_102, torch.float32)
        pow_38: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_319, 2);  convert_element_type_319 = None
        mean_26: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_38, [-1], True);  pow_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_103: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_26, 1e-06);  mean_26 = None
        rsqrt_26: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
        mul_110: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_102, rsqrt_26);  rsqrt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_320: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_110, torch.bfloat16);  mul_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_111: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg120_1, convert_element_type_320);  arg120_1 = convert_element_type_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_348: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_111, [4096, 512])
        permute_161: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        mm_89: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_348, permute_161);  view_348 = permute_161 = None
        view_349: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_89, [32, 128, 384]);  mm_89 = None
        view_350: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_349, [32, 128, -1, 64]);  view_349 = None
        permute_162: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_350, [0, 2, 1, 3]);  view_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_59: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_162, [32, 6, 128, 64]);  permute_162 = None
        clone_109: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
        view_357: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_109, [192, 128, 64]);  clone_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_351: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_111, [4096, 512])
        permute_163: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        mm_90: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_351, permute_163);  view_351 = permute_163 = None
        view_352: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_90, [32, 128, 384]);  mm_90 = None
        view_353: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_352, [32, 128, -1, 64]);  view_352 = None
        permute_164: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_353, [0, 2, 1, 3]);  view_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_167: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_164, [0, 1, 3, 2]);  permute_164 = None
        expand_60: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_167, [32, 6, 64, 128]);  permute_167 = None
        clone_110: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_358: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_110, [192, 64, 128]);  clone_110 = None
        bmm_28: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_357, view_358);  view_357 = view_358 = None
        view_359: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_28, [32, 6, 128, 128]);  bmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_104: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_359, add_72);  view_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_329: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_104, torch.float32);  add_104 = None
        amax_14: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_329, [-1], True)
        sub_16: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_329, amax_14);  convert_element_type_329 = amax_14 = None
        exp_14: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_15: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_18: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        convert_element_type_330: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_18, torch.bfloat16);  div_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_61: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_330, [32, 6, 128, 128]);  convert_element_type_330 = None
        view_362: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_61, [192, 128, 128]);  expand_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_354: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_111, [4096, 512]);  mul_111 = None
        permute_165: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        mm_91: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_354, permute_165);  view_354 = permute_165 = None
        view_355: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_91, [32, 128, 384]);  mm_91 = None
        view_356: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_355, [32, 128, -1, 64]);  view_355 = None
        permute_166: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_356, [0, 2, 1, 3]);  view_356 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_62: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_166, [32, 6, 128, 64]);  permute_166 = None
        clone_112: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_62, memory_format = torch.contiguous_format);  expand_62 = None
        view_363: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_112, [192, 128, 64]);  clone_112 = None
        bmm_29: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_362, view_363);  view_362 = view_363 = None
        view_364: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_29, [32, 6, 128, 64]);  bmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_168: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_364, [0, 2, 1, 3]);  view_364 = None
        clone_113: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_365: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_113, [32, 128, -1]);  clone_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_366: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_365, [4096, 384]);  view_365 = None
        permute_169: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
        mm_92: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_366, permute_169);  view_366 = permute_169 = None
        view_367: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_92, [32, 128, 512]);  mm_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_105: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_102, view_367);  add_102 = view_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_335: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_105, torch.float32)
        pow_39: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_335, 2);  convert_element_type_335 = None
        mean_27: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_39, [-1], True);  pow_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_106: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_27, 1e-06);  mean_27 = None
        rsqrt_27: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        mul_112: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_105, rsqrt_27);  rsqrt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_336: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_112, torch.bfloat16);  mul_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_113: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg125_1, convert_element_type_336);  arg125_1 = convert_element_type_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_368: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_113, [4096, 512]);  mul_113 = None
        permute_170: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        mm_93: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_368, permute_170);  view_368 = permute_170 = None
        view_369: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_93, [32, 128, 384]);  mm_93 = None
        view_370: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_369, [32, 128, -1, 64]);  view_369 = None
        permute_171: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_370, [0, 2, 1, 3]);  view_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_63: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_171, [32, 6, 128, 64]);  permute_171 = None
        clone_115: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
        view_377: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_115, [192, 128, 64]);  clone_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_371: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_75, [4096, 512])
        permute_172: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        mm_94: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_371, permute_172);  view_371 = permute_172 = None
        view_372: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_94, [32, 128, 384]);  mm_94 = None
        view_373: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_372, [32, 128, -1, 64]);  view_372 = None
        permute_173: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_373, [0, 2, 1, 3]);  view_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_176: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_173, [0, 1, 3, 2]);  permute_173 = None
        expand_64: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_176, [32, 6, 64, 128]);  permute_176 = None
        clone_116: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
        view_378: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_116, [192, 64, 128]);  clone_116 = None
        bmm_30: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_377, view_378);  view_377 = view_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        view_379: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_30, [32, 6, 128, 128]);  bmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_345: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_379, torch.float32);  view_379 = None
        amax_15: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_345, [-1], True)
        sub_17: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_345, amax_15);  convert_element_type_345 = amax_15 = None
        exp_15: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
        sum_16: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_19: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        convert_element_type_346: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16);  div_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_65: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_346, [32, 6, 128, 128]);  convert_element_type_346 = None
        view_382: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_65, [192, 128, 128]);  expand_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_374: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_75, [4096, 512])
        permute_174: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
        mm_95: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_374, permute_174);  view_374 = permute_174 = None
        view_375: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_95, [32, 128, 384]);  mm_95 = None
        view_376: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_375, [32, 128, -1, 64]);  view_375 = None
        permute_175: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_376, [0, 2, 1, 3]);  view_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_66: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_175, [32, 6, 128, 64]);  permute_175 = None
        clone_118: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
        view_383: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_118, [192, 128, 64]);  clone_118 = None
        bmm_31: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_382, view_383);  view_382 = view_383 = None
        view_384: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_31, [32, 6, 128, 64]);  bmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_177: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_384, [0, 2, 1, 3]);  view_384 = None
        clone_119: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_385: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_119, [32, 128, -1]);  clone_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_386: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_385, [4096, 384]);  view_385 = None
        permute_178: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        mm_96: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_386, permute_178);  view_386 = permute_178 = None
        view_387: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_96, [32, 128, 512]);  mm_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:400 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_108: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_105, view_387);  add_105 = view_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_351: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_108, torch.float32)
        pow_40: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_351, 2);  convert_element_type_351 = None
        mean_28: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_40, [-1], True);  pow_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_109: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_28, 1e-06);  mean_28 = None
        rsqrt_28: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
        mul_114: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_108, rsqrt_28);  rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_352: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_114, torch.bfloat16);  mul_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_115: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg130_1, convert_element_type_352);  arg130_1 = convert_element_type_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_388: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_115, [4096, 512])
        permute_179: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        mm_97: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_388, permute_179);  view_388 = permute_179 = None
        view_389: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_97, [32, 128, 1024]);  mm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_116: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_389, 0.5)
        pow_41: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_389, 3.0)
        mul_117: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_41, 0.044715);  pow_41 = None
        add_110: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_389, mul_117);  view_389 = mul_117 = None
        mul_118: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_110, 0.7978845608028654);  add_110 = None
        tanh_11: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.tanh.default(mul_118);  mul_118 = None
        add_111: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_119: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_116, add_111);  mul_116 = add_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_390: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_115, [4096, 512]);  mul_115 = None
        permute_180: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
        mm_98: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_390, permute_180);  view_390 = permute_180 = None
        view_391: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_98, [32, 128, 1024]);  mm_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_120: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_119, view_391);  mul_119 = view_391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_392: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_120, [4096, 1024]);  mul_120 = None
        permute_181: "bf16[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        mm_99: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_392, permute_181);  view_392 = permute_181 = None
        view_393: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_99, [32, 128, 512]);  mm_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_112: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_108, view_393);  add_108 = view_393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_359: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_112, torch.float32)
        pow_42: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_359, 2);  convert_element_type_359 = None
        mean_29: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_42, [-1], True);  pow_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_113: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_29, 1e-06);  mean_29 = None
        rsqrt_29: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
        mul_121: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_112, rsqrt_29);  rsqrt_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_360: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_121, torch.bfloat16);  mul_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_122: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg134_1, convert_element_type_360);  arg134_1 = convert_element_type_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_394: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_122, [4096, 512])
        permute_182: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        mm_100: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_394, permute_182);  view_394 = permute_182 = None
        view_395: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_100, [32, 128, 384]);  mm_100 = None
        view_396: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_395, [32, 128, -1, 64]);  view_395 = None
        permute_183: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_396, [0, 2, 1, 3]);  view_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_67: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_183, [32, 6, 128, 64]);  permute_183 = None
        clone_123: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
        view_403: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_123, [192, 128, 64]);  clone_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_397: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_122, [4096, 512])
        permute_184: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        mm_101: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_397, permute_184);  view_397 = permute_184 = None
        view_398: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_101, [32, 128, 384]);  mm_101 = None
        view_399: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_398, [32, 128, -1, 64]);  view_398 = None
        permute_185: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_399, [0, 2, 1, 3]);  view_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_188: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_185, [0, 1, 3, 2]);  permute_185 = None
        expand_68: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_188, [32, 6, 64, 128]);  permute_188 = None
        clone_124: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
        view_404: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_124, [192, 64, 128]);  clone_124 = None
        bmm_32: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_403, view_404);  view_403 = view_404 = None
        view_405: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_32, [32, 6, 128, 128]);  bmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_114: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_405, add_72);  view_405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_369: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_114, torch.float32);  add_114 = None
        amax_16: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_369, [-1], True)
        sub_18: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_369, amax_16);  convert_element_type_369 = amax_16 = None
        exp_16: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
        sum_17: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_20: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        convert_element_type_370: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_20, torch.bfloat16);  div_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_69: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_370, [32, 6, 128, 128]);  convert_element_type_370 = None
        view_408: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_69, [192, 128, 128]);  expand_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_400: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_122, [4096, 512]);  mul_122 = None
        permute_186: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        mm_102: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_400, permute_186);  view_400 = permute_186 = None
        view_401: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_102, [32, 128, 384]);  mm_102 = None
        view_402: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_401, [32, 128, -1, 64]);  view_401 = None
        permute_187: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_70: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_187, [32, 6, 128, 64]);  permute_187 = None
        clone_126: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_70, memory_format = torch.contiguous_format);  expand_70 = None
        view_409: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_126, [192, 128, 64]);  clone_126 = None
        bmm_33: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_408, view_409);  view_408 = view_409 = None
        view_410: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_33, [32, 6, 128, 64]);  bmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_189: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_410, [0, 2, 1, 3]);  view_410 = None
        clone_127: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_411: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_127, [32, 128, -1]);  clone_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_412: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_411, [4096, 384]);  view_411 = None
        permute_190: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
        mm_103: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_412, permute_190);  view_412 = permute_190 = None
        view_413: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_103, [32, 128, 512]);  mm_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_115: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_112, view_413);  add_112 = view_413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_375: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_115, torch.float32)
        pow_43: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_375, 2);  convert_element_type_375 = None
        mean_30: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_43, [-1], True);  pow_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_116: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_30, 1e-06);  mean_30 = None
        rsqrt_30: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        mul_123: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_115, rsqrt_30);  rsqrt_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_376: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_123, torch.bfloat16);  mul_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_124: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg139_1, convert_element_type_376);  arg139_1 = convert_element_type_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_414: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_124, [4096, 512]);  mul_124 = None
        permute_191: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
        mm_104: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_414, permute_191);  view_414 = permute_191 = None
        view_415: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_104, [32, 128, 384]);  mm_104 = None
        view_416: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_415, [32, 128, -1, 64]);  view_415 = None
        permute_192: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_416, [0, 2, 1, 3]);  view_416 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_71: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_192, [32, 6, 128, 64]);  permute_192 = None
        clone_129: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
        view_423: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_129, [192, 128, 64]);  clone_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_417: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_75, [4096, 512])
        permute_193: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        mm_105: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_417, permute_193);  view_417 = permute_193 = None
        view_418: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_105, [32, 128, 384]);  mm_105 = None
        view_419: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_418, [32, 128, -1, 64]);  view_418 = None
        permute_194: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_419, [0, 2, 1, 3]);  view_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_197: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_194, [0, 1, 3, 2]);  permute_194 = None
        expand_72: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_197, [32, 6, 64, 128]);  permute_197 = None
        clone_130: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
        view_424: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_130, [192, 64, 128]);  clone_130 = None
        bmm_34: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_423, view_424);  view_423 = view_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        view_425: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_34, [32, 6, 128, 128]);  bmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_385: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_425, torch.float32);  view_425 = None
        amax_17: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_385, [-1], True)
        sub_19: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_385, amax_17);  convert_element_type_385 = amax_17 = None
        exp_17: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_18: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_21: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
        convert_element_type_386: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16);  div_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_73: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_386, [32, 6, 128, 128]);  convert_element_type_386 = None
        view_428: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_73, [192, 128, 128]);  expand_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_420: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_75, [4096, 512])
        permute_195: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
        mm_106: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_420, permute_195);  view_420 = permute_195 = None
        view_421: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_106, [32, 128, 384]);  mm_106 = None
        view_422: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_421, [32, 128, -1, 64]);  view_421 = None
        permute_196: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_422, [0, 2, 1, 3]);  view_422 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_74: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_196, [32, 6, 128, 64]);  permute_196 = None
        clone_132: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
        view_429: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_132, [192, 128, 64]);  clone_132 = None
        bmm_35: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_428, view_429);  view_428 = view_429 = None
        view_430: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_35, [32, 6, 128, 64]);  bmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_198: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_430, [0, 2, 1, 3]);  view_430 = None
        clone_133: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_198, memory_format = torch.contiguous_format);  permute_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_431: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_133, [32, 128, -1]);  clone_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_432: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_431, [4096, 384]);  view_431 = None
        permute_199: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        mm_107: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_432, permute_199);  view_432 = permute_199 = None
        view_433: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_107, [32, 128, 512]);  mm_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:400 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_118: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_115, view_433);  add_115 = view_433 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_391: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_118, torch.float32)
        pow_44: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_391, 2);  convert_element_type_391 = None
        mean_31: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_44, [-1], True);  pow_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_119: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_31, 1e-06);  mean_31 = None
        rsqrt_31: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
        mul_125: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_118, rsqrt_31);  rsqrt_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_392: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_125, torch.bfloat16);  mul_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_126: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg144_1, convert_element_type_392);  arg144_1 = convert_element_type_392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_434: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_126, [4096, 512])
        permute_200: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        mm_108: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_434, permute_200);  view_434 = permute_200 = None
        view_435: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_108, [32, 128, 1024]);  mm_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_127: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_435, 0.5)
        pow_45: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_435, 3.0)
        mul_128: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_45, 0.044715);  pow_45 = None
        add_120: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_435, mul_128);  view_435 = mul_128 = None
        mul_129: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_120, 0.7978845608028654);  add_120 = None
        tanh_12: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.tanh.default(mul_129);  mul_129 = None
        add_121: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_12, 1.0);  tanh_12 = None
        mul_130: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_127, add_121);  mul_127 = add_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_436: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_126, [4096, 512]);  mul_126 = None
        permute_201: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        mm_109: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_436, permute_201);  view_436 = permute_201 = None
        view_437: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_109, [32, 128, 1024]);  mm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_131: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_130, view_437);  mul_130 = view_437 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_438: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_131, [4096, 1024]);  mul_131 = None
        permute_202: "bf16[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        mm_110: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_438, permute_202);  view_438 = permute_202 = None
        view_439: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_110, [32, 128, 512]);  mm_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_122: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_118, view_439);  add_118 = view_439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_399: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_122, torch.float32)
        pow_46: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_399, 2);  convert_element_type_399 = None
        mean_32: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_46, [-1], True);  pow_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_123: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_32, 1e-06);  mean_32 = None
        rsqrt_32: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
        mul_132: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_122, rsqrt_32);  rsqrt_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_400: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_132, torch.bfloat16);  mul_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_133: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg148_1, convert_element_type_400);  arg148_1 = convert_element_type_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_440: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_133, [4096, 512])
        permute_203: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        mm_111: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_440, permute_203);  view_440 = permute_203 = None
        view_441: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_111, [32, 128, 384]);  mm_111 = None
        view_442: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_441, [32, 128, -1, 64]);  view_441 = None
        permute_204: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_442, [0, 2, 1, 3]);  view_442 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_75: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_204, [32, 6, 128, 64]);  permute_204 = None
        clone_137: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
        view_449: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_137, [192, 128, 64]);  clone_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_443: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_133, [4096, 512])
        permute_205: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
        mm_112: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_443, permute_205);  view_443 = permute_205 = None
        view_444: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_112, [32, 128, 384]);  mm_112 = None
        view_445: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_444, [32, 128, -1, 64]);  view_444 = None
        permute_206: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_445, [0, 2, 1, 3]);  view_445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_209: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_206, [0, 1, 3, 2]);  permute_206 = None
        expand_76: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_209, [32, 6, 64, 128]);  permute_209 = None
        clone_138: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
        view_450: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_138, [192, 64, 128]);  clone_138 = None
        bmm_36: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_449, view_450);  view_449 = view_450 = None
        view_451: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_36, [32, 6, 128, 128]);  bmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_124: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_451, add_72);  view_451 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_409: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_124, torch.float32);  add_124 = None
        amax_18: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_409, [-1], True)
        sub_20: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_409, amax_18);  convert_element_type_409 = amax_18 = None
        exp_18: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
        sum_19: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_22: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
        convert_element_type_410: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_22, torch.bfloat16);  div_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_77: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_410, [32, 6, 128, 128]);  convert_element_type_410 = None
        view_454: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_77, [192, 128, 128]);  expand_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_446: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_133, [4096, 512]);  mul_133 = None
        permute_207: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        mm_113: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_446, permute_207);  view_446 = permute_207 = None
        view_447: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_113, [32, 128, 384]);  mm_113 = None
        view_448: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_447, [32, 128, -1, 64]);  view_447 = None
        permute_208: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_78: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_208, [32, 6, 128, 64]);  permute_208 = None
        clone_140: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_78, memory_format = torch.contiguous_format);  expand_78 = None
        view_455: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_140, [192, 128, 64]);  clone_140 = None
        bmm_37: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_454, view_455);  view_454 = view_455 = None
        view_456: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_37, [32, 6, 128, 64]);  bmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_210: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_456, [0, 2, 1, 3]);  view_456 = None
        clone_141: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_457: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_141, [32, 128, -1]);  clone_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_458: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_457, [4096, 384]);  view_457 = None
        permute_211: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        mm_114: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_458, permute_211);  view_458 = permute_211 = None
        view_459: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_114, [32, 128, 512]);  mm_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_125: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_122, view_459);  add_122 = view_459 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_415: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_125, torch.float32)
        pow_47: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_415, 2);  convert_element_type_415 = None
        mean_33: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_47, [-1], True);  pow_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_126: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_33, 1e-06);  mean_33 = None
        rsqrt_33: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        mul_134: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_125, rsqrt_33);  rsqrt_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_416: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_134, torch.bfloat16);  mul_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_135: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg153_1, convert_element_type_416);  arg153_1 = convert_element_type_416 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_460: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_135, [4096, 512]);  mul_135 = None
        permute_212: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        mm_115: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_460, permute_212);  view_460 = permute_212 = None
        view_461: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_115, [32, 128, 384]);  mm_115 = None
        view_462: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_461, [32, 128, -1, 64]);  view_461 = None
        permute_213: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_462, [0, 2, 1, 3]);  view_462 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_79: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_213, [32, 6, 128, 64]);  permute_213 = None
        clone_143: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_79, memory_format = torch.contiguous_format);  expand_79 = None
        view_469: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_143, [192, 128, 64]);  clone_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_463: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_75, [4096, 512])
        permute_214: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        mm_116: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_463, permute_214);  view_463 = permute_214 = None
        view_464: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_116, [32, 128, 384]);  mm_116 = None
        view_465: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_464, [32, 128, -1, 64]);  view_464 = None
        permute_215: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_465, [0, 2, 1, 3]);  view_465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_218: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_215, [0, 1, 3, 2]);  permute_215 = None
        expand_80: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_218, [32, 6, 64, 128]);  permute_218 = None
        clone_144: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
        view_470: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_144, [192, 64, 128]);  clone_144 = None
        bmm_38: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_469, view_470);  view_469 = view_470 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        view_471: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_38, [32, 6, 128, 128]);  bmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_425: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_471, torch.float32);  view_471 = None
        amax_19: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_425, [-1], True)
        sub_21: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_425, amax_19);  convert_element_type_425 = amax_19 = None
        exp_19: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
        sum_20: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_23: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
        convert_element_type_426: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_23, torch.bfloat16);  div_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_81: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_426, [32, 6, 128, 128]);  convert_element_type_426 = None
        view_474: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_81, [192, 128, 128]);  expand_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_466: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_75, [4096, 512])
        permute_216: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
        mm_117: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_466, permute_216);  view_466 = permute_216 = None
        view_467: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_117, [32, 128, 384]);  mm_117 = None
        view_468: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_467, [32, 128, -1, 64]);  view_467 = None
        permute_217: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_468, [0, 2, 1, 3]);  view_468 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_82: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_217, [32, 6, 128, 64]);  permute_217 = None
        clone_146: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_82, memory_format = torch.contiguous_format);  expand_82 = None
        view_475: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_146, [192, 128, 64]);  clone_146 = None
        bmm_39: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_474, view_475);  view_474 = view_475 = None
        view_476: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_39, [32, 6, 128, 64]);  bmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_219: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_476, [0, 2, 1, 3]);  view_476 = None
        clone_147: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_219, memory_format = torch.contiguous_format);  permute_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_477: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_147, [32, 128, -1]);  clone_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_478: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_477, [4096, 384]);  view_477 = None
        permute_220: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        mm_118: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_478, permute_220);  view_478 = permute_220 = None
        view_479: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_118, [32, 128, 512]);  mm_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:400 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_128: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_125, view_479);  add_125 = view_479 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_431: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_128, torch.float32)
        pow_48: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_431, 2);  convert_element_type_431 = None
        mean_34: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_48, [-1], True);  pow_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_129: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_34, 1e-06);  mean_34 = None
        rsqrt_34: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
        mul_136: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_128, rsqrt_34);  rsqrt_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_432: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_136, torch.bfloat16);  mul_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_137: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg158_1, convert_element_type_432);  arg158_1 = convert_element_type_432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_480: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_137, [4096, 512])
        permute_221: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        mm_119: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_480, permute_221);  view_480 = permute_221 = None
        view_481: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_119, [32, 128, 1024]);  mm_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_138: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_481, 0.5)
        pow_49: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_481, 3.0)
        mul_139: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_49, 0.044715);  pow_49 = None
        add_130: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_481, mul_139);  view_481 = mul_139 = None
        mul_140: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_130, 0.7978845608028654);  add_130 = None
        tanh_13: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.tanh.default(mul_140);  mul_140 = None
        add_131: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_13, 1.0);  tanh_13 = None
        mul_141: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_138, add_131);  mul_138 = add_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_482: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_137, [4096, 512]);  mul_137 = None
        permute_222: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
        mm_120: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_482, permute_222);  view_482 = permute_222 = None
        view_483: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_120, [32, 128, 1024]);  mm_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_142: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_141, view_483);  mul_141 = view_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_484: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_142, [4096, 1024]);  mul_142 = None
        permute_223: "bf16[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        mm_121: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_484, permute_223);  view_484 = permute_223 = None
        view_485: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_121, [32, 128, 512]);  mm_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_132: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_128, view_485);  add_128 = view_485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_439: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_132, torch.float32)
        pow_50: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_439, 2);  convert_element_type_439 = None
        mean_35: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_50, [-1], True);  pow_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_133: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_35, 1e-06);  mean_35 = None
        rsqrt_35: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
        mul_143: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_132, rsqrt_35);  rsqrt_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_440: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_143, torch.bfloat16);  mul_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_144: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg162_1, convert_element_type_440);  arg162_1 = convert_element_type_440 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_486: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_144, [4096, 512])
        permute_224: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        mm_122: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_486, permute_224);  view_486 = permute_224 = None
        view_487: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_122, [32, 128, 384]);  mm_122 = None
        view_488: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_487, [32, 128, -1, 64]);  view_487 = None
        permute_225: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_488, [0, 2, 1, 3]);  view_488 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_83: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_225, [32, 6, 128, 64]);  permute_225 = None
        clone_151: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_83, memory_format = torch.contiguous_format);  expand_83 = None
        view_495: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_151, [192, 128, 64]);  clone_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_489: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_144, [4096, 512])
        permute_226: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
        mm_123: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_489, permute_226);  view_489 = permute_226 = None
        view_490: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_123, [32, 128, 384]);  mm_123 = None
        view_491: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_490, [32, 128, -1, 64]);  view_490 = None
        permute_227: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_491, [0, 2, 1, 3]);  view_491 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_230: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_227, [0, 1, 3, 2]);  permute_227 = None
        expand_84: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_230, [32, 6, 64, 128]);  permute_230 = None
        clone_152: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
        view_496: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_152, [192, 64, 128]);  clone_152 = None
        bmm_40: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_495, view_496);  view_495 = view_496 = None
        view_497: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_40, [32, 6, 128, 128]);  bmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_134: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_497, add_72);  view_497 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_449: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_134, torch.float32);  add_134 = None
        amax_20: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_449, [-1], True)
        sub_22: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_449, amax_20);  convert_element_type_449 = amax_20 = None
        exp_20: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_21: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_24: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
        convert_element_type_450: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_24, torch.bfloat16);  div_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_85: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_450, [32, 6, 128, 128]);  convert_element_type_450 = None
        view_500: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_85, [192, 128, 128]);  expand_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_492: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_144, [4096, 512]);  mul_144 = None
        permute_228: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        mm_124: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_492, permute_228);  view_492 = permute_228 = None
        view_493: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_124, [32, 128, 384]);  mm_124 = None
        view_494: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_493, [32, 128, -1, 64]);  view_493 = None
        permute_229: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_494, [0, 2, 1, 3]);  view_494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_86: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_229, [32, 6, 128, 64]);  permute_229 = None
        clone_154: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_86, memory_format = torch.contiguous_format);  expand_86 = None
        view_501: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_154, [192, 128, 64]);  clone_154 = None
        bmm_41: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_500, view_501);  view_500 = view_501 = None
        view_502: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_41, [32, 6, 128, 64]);  bmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_231: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_502, [0, 2, 1, 3]);  view_502 = None
        clone_155: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_231, memory_format = torch.contiguous_format);  permute_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_503: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_155, [32, 128, -1]);  clone_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_504: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_503, [4096, 384]);  view_503 = None
        permute_232: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
        mm_125: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_504, permute_232);  view_504 = permute_232 = None
        view_505: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_125, [32, 128, 512]);  mm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_135: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_132, view_505);  add_132 = view_505 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_455: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_135, torch.float32)
        pow_51: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_455, 2);  convert_element_type_455 = None
        mean_36: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_51, [-1], True);  pow_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_136: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_36, 1e-06);  mean_36 = None
        rsqrt_36: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
        mul_145: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_135, rsqrt_36);  rsqrt_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_456: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_145, torch.bfloat16);  mul_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_146: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg167_1, convert_element_type_456);  arg167_1 = convert_element_type_456 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_506: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_146, [4096, 512]);  mul_146 = None
        permute_233: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
        mm_126: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_506, permute_233);  view_506 = permute_233 = None
        view_507: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_126, [32, 128, 384]);  mm_126 = None
        view_508: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_507, [32, 128, -1, 64]);  view_507 = None
        permute_234: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_508, [0, 2, 1, 3]);  view_508 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_87: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_234, [32, 6, 128, 64]);  permute_234 = None
        clone_157: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_87, memory_format = torch.contiguous_format);  expand_87 = None
        view_515: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_157, [192, 128, 64]);  clone_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_509: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_75, [4096, 512])
        permute_235: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        mm_127: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_509, permute_235);  view_509 = permute_235 = None
        view_510: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_127, [32, 128, 384]);  mm_127 = None
        view_511: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_510, [32, 128, -1, 64]);  view_510 = None
        permute_236: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_511, [0, 2, 1, 3]);  view_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_239: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_236, [0, 1, 3, 2]);  permute_236 = None
        expand_88: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_239, [32, 6, 64, 128]);  permute_239 = None
        clone_158: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
        view_516: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_158, [192, 64, 128]);  clone_158 = None
        bmm_42: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_515, view_516);  view_515 = view_516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        view_517: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_42, [32, 6, 128, 128]);  bmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_465: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_517, torch.float32);  view_517 = None
        amax_21: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_465, [-1], True)
        sub_23: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_465, amax_21);  convert_element_type_465 = amax_21 = None
        exp_21: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_23);  sub_23 = None
        sum_22: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_25: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
        convert_element_type_466: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_25, torch.bfloat16);  div_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_89: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_466, [32, 6, 128, 128]);  convert_element_type_466 = None
        view_520: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_89, [192, 128, 128]);  expand_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_512: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_75, [4096, 512])
        permute_237: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        mm_128: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_512, permute_237);  view_512 = permute_237 = None
        view_513: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_128, [32, 128, 384]);  mm_128 = None
        view_514: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_513, [32, 128, -1, 64]);  view_513 = None
        permute_238: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_514, [0, 2, 1, 3]);  view_514 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_90: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_238, [32, 6, 128, 64]);  permute_238 = None
        clone_160: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_90, memory_format = torch.contiguous_format);  expand_90 = None
        view_521: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_160, [192, 128, 64]);  clone_160 = None
        bmm_43: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_520, view_521);  view_520 = view_521 = None
        view_522: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_43, [32, 6, 128, 64]);  bmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_240: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_522, [0, 2, 1, 3]);  view_522 = None
        clone_161: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_240, memory_format = torch.contiguous_format);  permute_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_523: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_161, [32, 128, -1]);  clone_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_524: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_523, [4096, 384]);  view_523 = None
        permute_241: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        mm_129: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_524, permute_241);  view_524 = permute_241 = None
        view_525: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_129, [32, 128, 512]);  mm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:400 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_138: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_135, view_525);  add_135 = view_525 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_471: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_138, torch.float32)
        pow_52: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_471, 2);  convert_element_type_471 = None
        mean_37: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_52, [-1], True);  pow_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_139: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_37, 1e-06);  mean_37 = None
        rsqrt_37: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
        mul_147: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_138, rsqrt_37);  rsqrt_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_472: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_147, torch.bfloat16);  mul_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_148: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg172_1, convert_element_type_472);  arg172_1 = convert_element_type_472 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_526: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_148, [4096, 512])
        permute_242: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        mm_130: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_526, permute_242);  view_526 = permute_242 = None
        view_527: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_130, [32, 128, 1024]);  mm_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_149: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_527, 0.5)
        pow_53: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_527, 3.0)
        mul_150: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_53, 0.044715);  pow_53 = None
        add_140: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_527, mul_150);  view_527 = mul_150 = None
        mul_151: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_140, 0.7978845608028654);  add_140 = None
        tanh_14: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.tanh.default(mul_151);  mul_151 = None
        add_141: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_14, 1.0);  tanh_14 = None
        mul_152: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_149, add_141);  mul_149 = add_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_528: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_148, [4096, 512]);  mul_148 = None
        permute_243: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
        mm_131: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_528, permute_243);  view_528 = permute_243 = None
        view_529: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_131, [32, 128, 1024]);  mm_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_153: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_152, view_529);  mul_152 = view_529 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_530: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_153, [4096, 1024]);  mul_153 = None
        permute_244: "bf16[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        mm_132: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_530, permute_244);  view_530 = permute_244 = None
        view_531: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_132, [32, 128, 512]);  mm_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_142: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_138, view_531);  add_138 = view_531 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_479: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_142, torch.float32)
        pow_54: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_479, 2);  convert_element_type_479 = None
        mean_38: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_54, [-1], True);  pow_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_143: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_38, 1e-06);  mean_38 = None
        rsqrt_38: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
        mul_154: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_142, rsqrt_38);  rsqrt_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_480: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_154, torch.bfloat16);  mul_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_155: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg176_1, convert_element_type_480);  arg176_1 = convert_element_type_480 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_532: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_155, [4096, 512])
        permute_245: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        mm_133: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_532, permute_245);  view_532 = permute_245 = None
        view_533: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_133, [32, 128, 384]);  mm_133 = None
        view_534: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_533, [32, 128, -1, 64]);  view_533 = None
        permute_246: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_534, [0, 2, 1, 3]);  view_534 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_91: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_246, [32, 6, 128, 64]);  permute_246 = None
        clone_165: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_91, memory_format = torch.contiguous_format);  expand_91 = None
        view_541: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_165, [192, 128, 64]);  clone_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_535: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_155, [4096, 512])
        permute_247: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
        mm_134: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_535, permute_247);  view_535 = permute_247 = None
        view_536: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_134, [32, 128, 384]);  mm_134 = None
        view_537: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_536, [32, 128, -1, 64]);  view_536 = None
        permute_248: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_537, [0, 2, 1, 3]);  view_537 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_251: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_248, [0, 1, 3, 2]);  permute_248 = None
        expand_92: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_251, [32, 6, 64, 128]);  permute_251 = None
        clone_166: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
        view_542: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_166, [192, 64, 128]);  clone_166 = None
        bmm_44: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_541, view_542);  view_541 = view_542 = None
        view_543: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_44, [32, 6, 128, 128]);  bmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_144: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_543, add_72);  view_543 = add_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_489: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_144, torch.float32);  add_144 = None
        amax_22: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_489, [-1], True)
        sub_24: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_489, amax_22);  convert_element_type_489 = amax_22 = None
        exp_22: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_24);  sub_24 = None
        sum_23: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_26: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
        convert_element_type_490: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_26, torch.bfloat16);  div_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_93: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_490, [32, 6, 128, 128]);  convert_element_type_490 = None
        view_546: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_93, [192, 128, 128]);  expand_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_538: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_155, [4096, 512]);  mul_155 = None
        permute_249: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        mm_135: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_538, permute_249);  view_538 = permute_249 = None
        view_539: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_135, [32, 128, 384]);  mm_135 = None
        view_540: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_539, [32, 128, -1, 64]);  view_539 = None
        permute_250: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_540, [0, 2, 1, 3]);  view_540 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_94: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_250, [32, 6, 128, 64]);  permute_250 = None
        clone_168: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_94, memory_format = torch.contiguous_format);  expand_94 = None
        view_547: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_168, [192, 128, 64]);  clone_168 = None
        bmm_45: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_546, view_547);  view_546 = view_547 = None
        view_548: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_45, [32, 6, 128, 64]);  bmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_252: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_548, [0, 2, 1, 3]);  view_548 = None
        clone_169: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_252, memory_format = torch.contiguous_format);  permute_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_549: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_169, [32, 128, -1]);  clone_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_550: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_549, [4096, 384]);  view_549 = None
        permute_253: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
        mm_136: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_550, permute_253);  view_550 = permute_253 = None
        view_551: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_136, [32, 128, 512]);  mm_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_145: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_142, view_551);  add_142 = view_551 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_495: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_145, torch.float32)
        pow_55: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_495, 2);  convert_element_type_495 = None
        mean_39: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_55, [-1], True);  pow_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_146: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_39, 1e-06);  mean_39 = None
        rsqrt_39: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
        mul_156: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_145, rsqrt_39);  rsqrt_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_496: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_156, torch.bfloat16);  mul_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_157: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg181_1, convert_element_type_496);  arg181_1 = convert_element_type_496 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        view_552: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_157, [4096, 512]);  mul_157 = None
        permute_254: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
        mm_137: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_552, permute_254);  view_552 = permute_254 = None
        view_553: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_137, [32, 128, 384]);  mm_137 = None
        view_554: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_553, [32, 128, -1, 64]);  view_553 = None
        permute_255: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_554, [0, 2, 1, 3]);  view_554 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_95: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_255, [32, 6, 128, 64]);  permute_255 = None
        clone_171: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_95, memory_format = torch.contiguous_format);  expand_95 = None
        view_561: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_171, [192, 128, 64]);  clone_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        view_555: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_75, [4096, 512])
        permute_256: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        mm_138: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_555, permute_256);  view_555 = permute_256 = None
        view_556: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_138, [32, 128, 384]);  mm_138 = None
        view_557: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_556, [32, 128, -1, 64]);  view_556 = None
        permute_257: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_557, [0, 2, 1, 3]);  view_557 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_260: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.permute.default(permute_257, [0, 1, 3, 2]);  permute_257 = None
        expand_96: "bf16[32, 6, 64, 128][49152, 64, 1, 384]cuda:0" = torch.ops.aten.expand.default(permute_260, [32, 6, 64, 128]);  permute_260 = None
        clone_172: "bf16[32, 6, 64, 128][49152, 8192, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_96, memory_format = torch.contiguous_format);  expand_96 = None
        view_562: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_172, [192, 64, 128]);  clone_172 = None
        bmm_46: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_561, view_562);  view_561 = view_562 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        view_563: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_46, [32, 6, 128, 128]);  bmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_505: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_563, torch.float32);  view_563 = None
        amax_23: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_505, [-1], True)
        sub_25: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_505, amax_23);  convert_element_type_505 = amax_23 = None
        exp_23: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_24: "f32[32, 6, 128, 1][768, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_27: "f32[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
        convert_element_type_506: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_27, torch.bfloat16);  div_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_97: "bf16[32, 6, 128, 128][98304, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_506, [32, 6, 128, 128]);  convert_element_type_506 = None
        view_566: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_97, [192, 128, 128]);  expand_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        view_558: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_75, [4096, 512])
        permute_258: "bf16[512, 384][1, 512]cuda:0" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
        mm_139: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_558, permute_258);  view_558 = permute_258 = None
        view_559: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_139, [32, 128, 384]);  mm_139 = None
        view_560: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_559, [32, 128, -1, 64]);  view_559 = None
        permute_259: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_560, [0, 2, 1, 3]);  view_560 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_98: "bf16[32, 6, 128, 64][49152, 64, 384, 1]cuda:0" = torch.ops.aten.expand.default(permute_259, [32, 6, 128, 64]);  permute_259 = None
        clone_174: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(expand_98, memory_format = torch.contiguous_format);  expand_98 = None
        view_567: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_174, [192, 128, 64]);  clone_174 = None
        bmm_47: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_566, view_567);  view_566 = view_567 = None
        view_568: "bf16[32, 6, 128, 64][49152, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_47, [32, 6, 128, 64]);  bmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_261: "bf16[32, 128, 6, 64][49152, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_568, [0, 2, 1, 3]);  view_568 = None
        clone_175: "bf16[32, 128, 6, 64][49152, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_261, memory_format = torch.contiguous_format);  permute_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_569: "bf16[32, 128, 384][49152, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_175, [32, 128, -1]);  clone_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_570: "bf16[4096, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_569, [4096, 384]);  view_569 = None
        permute_262: "bf16[384, 512][1, 384]cuda:0" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        mm_140: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_570, permute_262);  view_570 = permute_262 = None
        view_571: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_140, [32, 128, 512]);  mm_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:400 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_148: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_145, view_571);  add_145 = view_571 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_511: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_148, torch.float32)
        pow_56: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_511, 2);  convert_element_type_511 = None
        mean_40: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_56, [-1], True);  pow_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_149: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_40, 1e-06);  mean_40 = None
        rsqrt_40: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
        mul_158: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_148, rsqrt_40);  rsqrt_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_512: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_158, torch.bfloat16);  mul_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_159: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg186_1, convert_element_type_512);  arg186_1 = convert_element_type_512 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_572: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_159, [4096, 512])
        permute_263: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        mm_141: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_572, permute_263);  view_572 = permute_263 = None
        view_573: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_141, [32, 128, 1024]);  mm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_160: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_573, 0.5)
        pow_57: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(view_573, 3.0)
        mul_161: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_57, 0.044715);  pow_57 = None
        add_150: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_573, mul_161);  view_573 = mul_161 = None
        mul_162: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_150, 0.7978845608028654);  add_150 = None
        tanh_15: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.tanh.default(mul_162);  mul_162 = None
        add_151: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_15, 1.0);  tanh_15 = None
        mul_163: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_160, add_151);  mul_160 = add_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_574: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_159, [4096, 512]);  mul_159 = None
        permute_264: "bf16[512, 1024][1, 512]cuda:0" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
        mm_142: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_574, permute_264);  view_574 = permute_264 = None
        view_575: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_142, [32, 128, 1024]);  mm_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_164: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_163, view_575);  mul_163 = view_575 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_576: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_164, [4096, 1024]);  mul_164 = None
        permute_265: "bf16[1024, 512][1, 1024]cuda:0" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        mm_143: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_576, permute_265);  view_576 = permute_265 = None
        view_577: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_143, [32, 128, 512]);  mm_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_152: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_148, view_577);  add_148 = view_577 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_519: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_152, torch.float32)
        pow_58: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_519, 2);  convert_element_type_519 = None
        mean_41: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_58, [-1], True);  pow_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_153: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_41, 1e-06);  mean_41 = None
        rsqrt_41: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
        mul_165: "f32[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_152, rsqrt_41);  add_152 = rsqrt_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:67 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_520: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_165, torch.bfloat16);  mul_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_166: "bf16[32, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(arg190_1, convert_element_type_520);  arg190_1 = convert_element_type_520 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:1143 in forward, code: lm_logits = self.lm_head(sequence_output)
        view_578: "bf16[4096, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_166, [4096, 512]);  mul_166 = None
        permute_266: "bf16[512, 250112][1, 512]cuda:0" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        mm_144: "bf16[4096, 250112][250112, 1]cuda:0" = torch.ops.aten.mm.default(view_578, permute_266);  view_578 = permute_266 = None
        view_579: "bf16[32, 128, 250112][32014336, 250112, 1]cuda:0" = torch.ops.aten.reshape.default(mm_144, [32, 128, 250112]);  mm_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:1150 in forward, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        view_580: "bf16[4096, 250112][250112, 1]cuda:0" = torch.ops.aten.reshape.default(view_579, [-1, 250112])
        convert_element_type_523: "f32[4096, 250112][250112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_580, torch.float32);  view_580 = None
        amax_24: "f32[4096, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_523, [1], True)
        sub_26: "f32[4096, 250112][250112, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_523, amax_24);  convert_element_type_523 = amax_24 = None
        exp_24: "f32[4096, 250112][250112, 1]cuda:0" = torch.ops.aten.exp.default(sub_26)
        sum_25: "f32[4096, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log_2: "f32[4096, 1][1, 1]cuda:0" = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_27: "f32[4096, 250112][250112, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_26, log_2);  sub_26 = log_2 = None
        convert_element_type_524: "bf16[4096, 250112][250112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_27, torch.bfloat16);  sub_27 = None
        ne: "b8[4096][1]cuda:0" = torch.ops.aten.ne.Scalar(view_581, -100)
        full_default_4: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5: "i64[4096][1]cuda:0" = torch.ops.aten.where.self(ne, view_581, full_default_4);  ne = full_default_4 = None
        unsqueeze_18: "i64[4096, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(where_5, 1);  where_5 = None
        gather: "bf16[4096, 1][1, 1]cuda:0" = torch.ops.aten.gather.default(convert_element_type_524, 1, unsqueeze_18);  convert_element_type_524 = unsqueeze_18 = None
        squeeze: "bf16[4096][1]cuda:0" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg_1: "bf16[4096][1]cuda:0" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_5: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "bf16[4096][1]cuda:0" = torch.ops.aten.where.self(ne_1, neg_1, full_default_5);  ne_1 = neg_1 = full_default_5 = None
        sum_27: "bf16[][]cuda:0" = torch.ops.aten.sum.default(where_6);  where_6 = None
        ne_2: "b8[4096][1]cuda:0" = torch.ops.aten.ne.Scalar(view_581, -100);  view_581 = None
        sum_26: "i64[][]cuda:0" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_525: "bf16[][]cuda:0" = torch.ops.prims.convert_element_type.default(sum_26, torch.bfloat16);  sum_26 = None
        div_28: "bf16[][]cuda:0" = torch.ops.aten.div.Tensor(sum_27, convert_element_type_525);  sum_27 = convert_element_type_525 = None
        return (div_28, view_579, mul_75)
